"""
BTC XGBoost Strategy — Shadow Forward Tester (Multi-Timeframe)
==============================================================
ENTRY  : Daily timeframe — ML model generates signal once per day at 00:05 UTC
         after the daily candle closes on Hyperliquid.
EXIT   : 1-minute timeframe — a continuous monitor loop checks price every
         1 minutes against SL / TP / horizon and logs every check.

No real orders are ever placed. All trades are virtual (paper money).

ARCHITECTURE
------------
Two independent loops run concurrently via Python's threading module:

  ┌─────────────────────────────────────────────────────┐
  │  DAILY LOOP  (runs once at 00:05 UTC)               │
  │  • Fetch latest closed daily candle from HL         │
  │  • Rebuild features (200-day MA, RSI, BB, etc.)     │
  │  • Run XGBoost → prediction                         │
  │  • If no position AND prediction > threshold → ENTER│
  └─────────────────────────────────────────────────────┘
                          │
                          │ writes entry to state file
                          ▼
  ┌─────────────────────────────────────────────────────┐
  │  1-MIN MONITOR LOOP  (runs every 1 minutes)       │
  │  • Fetch latest 1-min candle from HL               │
  │  • If position open → check SL / TP / horizon       │
  │  • Log every check to monitor_log.csv               │
  │  • If exit triggered → record trade & update wallet │
  └─────────────────────────────────────────────────────┘

HOW TO RUN
----------
Run this single script — it starts both loops automatically:
    python claude_forwardtesting_hyperliquid.py

View report at any time:
    python claude_forwardtesting_hyperliquid.py report

Run in background (Linux/Mac):
    nohup python claude_forwardtesting_hyperliquid.py > shadow.log 2>&1 &

Stop it:
    kill $(cat shadow.pid)

REQUIREMENTS
------------
    pip install ccxt yfinance xgboost scikit-learn pandas numpy
"""

import os
import sys
import json
import time
import pickle
import logging
import warnings
import threading
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import yfinance as yf
import ccxt

from datetime import datetime, timezone, timedelta

# ── Logging ────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


def _yf_download_with_retry(tickers, max_attempts=3, base_delay_sec=5, **kwargs):
    """
    Wrapper around yf.download() with two fixes for unattended/CI environments:

    1. threads=False — yfinance's internal SQLite cache (used for Yahoo's
       auth cookie/crumb handling) is not safe for concurrent writes. The
       default threads=True fetches multiple tickers in parallel and can
       trigger 'sqlite3.OperationalError: database is locked', which is
       exactly what a fresh GitHub Actions runner hits on multi-ticker
       downloads. Forcing sequential fetches avoids this entirely.

    2. Retry with backoff — an unattended bot running on a schedule has no
       human watching for a one-off network blip or Yahoo rate-limit. A
       single failed download currently skips the entire cycle silently;
       this retries a couple of times before giving up.
    """
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            result = yf.download(tickers, threads=False, **kwargs)
            if result is None or result.empty:
                raise ValueError("yfinance returned an empty result")
            return result
        except Exception as e:
            last_error = e
            log.warning(
                f"yf.download attempt {attempt}/{max_attempts} failed: {e}"
            )
            if attempt < max_attempts:
                time.sleep(base_delay_sec * attempt)
    raise RuntimeError(
        f"yf.download failed after {max_attempts} attempts. Last error: {last_error}"
    )

# ═══════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════
CONFIG = {
    # Hyperliquid
    "symbol":            "BTC/USDC:USDC",
    "daily_tf":          "1d",
    "intraday_tf":       "1m",           # exit monitoring timeframe

    # Strategy
    "horizon_days":      7,               # max hold in calendar days
    "quantile_entry":    0.60,            # top 40% predictions trigger entry

    # Portfolio (paper money)
    "initial_capital":   1_000.0,
    "position_size_pct": 100.0,
    "trade_fee_pct":     0.045,
    "leverage":          1,

    # Training data
    "yf_train_start":    "2018-01-01",
    "yf_train_end":      "2024-11-01",

    # Lookback for feature warm-up
    "hl_lookback_days":  400,

    # Model retraining
    "retrain_every_days": 7,

    # Loop intervals
    "monitor_interval_sec": 60,          # 1 minute = 60 seconds
    "daily_check_hour_utc": 0,            # hour to run daily loop (midnight UTC)
    "daily_check_minute_utc": 5,          # minute offset after candle close

    # File paths
    "trade_log_file":    "shadow_trade_log.json",
    "equity_log_file":   "shadow_equity_log.csv",
    "monitor_log_file":  "shadow_monitor_log.csv",
    "state_file":        "shadow_state.json",
    "model_file":        "shadow_model.pkl",
}

FEATURE_COLS = [
    "btc_ret_1d", "spy_ret_1d", "qqq_ret_1d", "gold_ret_1d",
    "btc_log_ret_7d", "btc_log_ret_14d", "btc_log_ret_21d", "btc_log_ret_28d",
    "btc_log_vol_7d", "btc_log_vol_14d", "btc_log_vol_21d",
    "btc_log_vol_28d", "btc_log_vol_56d",
    "price_vs_ma_7", "price_vs_ma_14", "price_vs_ma_21",
    "price_vs_ma_28", "price_vs_ma_50",
    "rsi_14", "stoch_kd_diff", "adi", "mfi_14",
    "bb_width", "bb_pct_b", "kc_width", "kc_pct", "psar_dist", "regime",
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Thread lock — both loops share the state file; this prevents race conditions
_state_lock = threading.Lock()


# ═══════════════════════════════════════════════════════════════════════════
# STATE & FILE HELPERS
# ═══════════════════════════════════════════════════════════════════════════
def _path(filename):
    return os.path.join(SCRIPT_DIR, filename)


def load_state() -> dict:
    fp = _path(CONFIG["state_file"])
    if os.path.exists(fp):
        with open(fp) as f:
            return json.load(f)
    return {
        "in_position":         False,
        "entry_price":         None,
        "entry_date":          None,
        "entry_datetime_utc":  None,
        "target_return":       None,
        "sl_magnitude":        None,
        "position_usdc":       None,
        "quantity_btc":        None,
        "fee_entry":           None,
        "wallet":              CONFIG["initial_capital"],
        "last_ath":            CONFIG["initial_capital"],
        "prediction_history":  [],
        "model_trained_date":  None,
        "daily_loop_last_run": None,      # ISO date string — prevents double-firing
    }


def save_state(state: dict):
    fp = _path(CONFIG["state_file"])
    with open(fp, "w") as f:
        json.dump(state, f, indent=2, default=str)


def append_trade(event: dict):
    fp = _path(CONFIG["trade_log_file"])
    trades = []
    if os.path.exists(fp):
        with open(fp) as f:
            trades = json.load(f)
    trades.append(event)
    with open(fp, "w") as f:
        json.dump(trades, f, indent=2, default=str)


def append_monitor_log(row: dict):
    fp    = _path(CONFIG["monitor_log_file"])
    write_header = not os.path.exists(fp)
    pd.DataFrame([row]).to_csv(fp, mode="a", header=write_header, index=False)


def append_equity_log(row: dict):
    fp    = _path(CONFIG["equity_log_file"])
    write_header = not os.path.exists(fp)
    pd.DataFrame([row]).to_csv(fp, mode="a", header=write_header, index=False)


# ═══════════════════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════════════════
def _get_exchange():
    """
    Create a Hyperliquid CCXT exchange instance.

    The 'loadSpotMarkets' option is set to False to work around a bug in the
    CCXT Hyperliquid connector where some spot tokens have None as their base
    currency, causing a TypeError when building the symbol string.
    We only need perp markets (BTC/USDC:USDC) so this is safe to disable.
    """
    ex = ccxt.hyperliquid({
        "enableRateLimit": True,
        "options": {
            "loadSpotMarkets": False,   # skip spot — avoids None base-currency bug
        },
    })
    return ex


def fetch_daily_ohlcv(lookback_days: int) -> pd.DataFrame:
    since_ms = int(
        (datetime.now(timezone.utc).timestamp() - lookback_days * 86400) * 1000
    )
    ex        = _get_exchange()
    all_ohlcv = []
    cursor    = since_ms
    while True:
        batch = ex.fetch_ohlcv(CONFIG["symbol"], CONFIG["daily_tf"],
                                since=cursor, limit=500)
        if not batch:
            break
        all_ohlcv.extend(batch)
        if len(batch) < 500:
            break
        cursor = batch[-1][0] + 1

    df = _ohlcv_to_df(all_ohlcv)
    # Drop today's incomplete candle
    now_utc = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    df = df[df["timestamp"] < now_utc]
    return df


def fetch_latest_1min_candle() -> dict | None:
    """Fetch the single most-recently-closed 1-min candle."""
    ex    = _get_exchange()
    ohlcv = ex.fetch_ohlcv(CONFIG["symbol"], CONFIG["intraday_tf"],
                             limit=2)         # last 2 to ensure one is closed
    if not ohlcv or len(ohlcv) < 1:
        return None
    # The second-to-last candle is guaranteed closed
    c = ohlcv[-2]
    return {
        "timestamp": pd.to_datetime(c[0], unit="ms"),
        "open":  float(c[1]),
        "high":  float(c[2]),
        "low":   float(c[3]),
        "close": float(c[4]),
    }


def _ohlcv_to_df(raw: list) -> pd.DataFrame:
    df = pd.DataFrame(raw, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates("timestamp").sort_values("timestamp").reset_index(drop=True)
    for col in ["open","high","low","close","volume"]:
        df[col] = pd.to_numeric(df[col])
    return df


def fetch_yf_context(start: str) -> pd.DataFrame:
    raw = _yf_download_with_retry(["SPY","QQQ","GC=F"], start=start,
                                  progress=False, auto_adjust=True, group_by="ticker")
    ctx = pd.DataFrame()
    ctx["spy_close"]  = raw["SPY"]["Close"]
    ctx["qqq_close"]  = raw["QQQ"]["Close"]
    ctx["gold_close"] = raw["GC=F"]["Close"]
    ctx.index = pd.to_datetime(ctx.index).tz_localize(None)
    return ctx


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING  (identical to backtest)
# ═══════════════════════════════════════════════════════════════════════════
def _parabolic_sar(high, low, step=0.02, max_step=0.20):
    n   = len(high)
    sar = np.full(n, np.nan)
    ep  = np.full(n, np.nan)
    bull, af = True, step
    sar[0], ep[0] = low.iloc[0], high.iloc[0]
    for i in range(1, n):
        ps, pe = sar[i-1], ep[i-1]
        if bull:
            sar[i] = ps + af * (pe - ps)
            sar[i] = min(sar[i], low.iloc[i-1],
                         low.iloc[i-2] if i >= 2 else low.iloc[i-1])
            if low.iloc[i] < sar[i]:
                bull, sar[i], ep[i], af = False, pe, low.iloc[i], step
            else:
                ep[i] = high.iloc[i] if high.iloc[i] > pe else pe
                if high.iloc[i] > pe: af = min(af + step, max_step)
        else:
            sar[i] = ps + af * (pe - ps)
            sar[i] = max(sar[i], high.iloc[i-1],
                         high.iloc[i-2] if i >= 2 else high.iloc[i-1])
            if high.iloc[i] > sar[i]:
                bull, sar[i], ep[i], af = True, pe, high.iloc[i], step
            else:
                ep[i] = low.iloc[i] if low.iloc[i] < pe else pe
                if low.iloc[i] < pe: af = min(af + step, max_step)
    return sar


def build_features(hl_df, ctx_df, horizon=7):
    df = pd.DataFrame()
    df.index = pd.to_datetime(hl_df["timestamp"].values)
    df.index.name = "Date"
    df["btc_close"]  = hl_df["close"].values
    df["btc_high"]   = hl_df["high"].values
    df["btc_low"]    = hl_df["low"].values
    df["btc_volume"] = hl_df["volume"].values
    df["spy_close"]  = ctx_df["spy_close"].reindex(df.index, method="ffill")
    df["qqq_close"]  = ctx_df["qqq_close"].reindex(df.index, method="ffill")
    df["gold_close"] = ctx_df["gold_close"].reindex(df.index, method="ffill")

    df["btc_ret_1d"]  = df["btc_close"].pct_change()
    df["spy_ret_1d"]  = df["spy_close"].pct_change()
    df["qqq_ret_1d"]  = df["qqq_close"].pct_change()
    df["gold_ret_1d"] = df["gold_close"].pct_change()
    df["btc_log_ret"] = np.log(df["btc_close"]).diff()

    for w in [7,14,21,28]:
        df[f"btc_log_ret_{w}d"] = df["btc_log_ret"].rolling(w).sum()
    for w in [7,14,21,28,56]:
        df[f"btc_log_vol_{w}d"] = df["btc_log_ret"].rolling(w).std()
    for w in [7,14,21,28]:
        df[f"btc_ret_{w}d"] = df["btc_close"].pct_change(w)
        df[f"btc_vol_{w}d"] = df["btc_ret_1d"].rolling(w).std()
    for w in [7,14,21,28,50,200]:
        df[f"ma_{w}"] = df["btc_close"].rolling(w).mean()
    df["regime"] = (df["btc_close"] > df["ma_200"]).astype(int)
    for w in [7,14,21,28,50]:
        df[f"price_vs_ma_{w}"] = df["btc_close"] / df[f"ma_{w}"] - 1
    for w in [7,14,21,28]:
        vma = df["btc_volume"].rolling(w).mean()
        df[f"vol_ma_{w}"]       = vma
        df[f"volume_ratio_{w}"] = df["btc_volume"] / vma

    delta    = df["btc_close"].diff()
    avg_gain = delta.clip(lower=0).rolling(14).mean()
    avg_loss = (-delta.clip(upper=0)).rolling(14).mean()
    df["rsi_14"] = 100 - (100 / (1 + avg_gain / avg_loss))

    low_min  = df["btc_low"].rolling(5).min()
    high_max = df["btc_high"].rolling(5).max()
    raw_k    = 100*(df["btc_close"]-low_min)/(high_max-low_min).replace(0,np.nan)
    df["stoch_k"]       = raw_k.rolling(3).mean()
    df["stoch_d"]       = df["stoch_k"].rolling(3).mean()
    df["stoch_kd_diff"] = df["stoch_k"] - df["stoch_d"]

    clv = ((df["btc_close"]-df["btc_low"])-(df["btc_high"]-df["btc_close"])) / \
          (df["btc_high"]-df["btc_low"]).replace(0,np.nan)
    df["adi"] = (clv * df["btc_volume"]).rolling(30).sum()

    tp     = (df["btc_high"]+df["btc_low"]+df["btc_close"])/3
    raw_mf = tp*df["btc_volume"]
    tp_d   = tp.diff()
    pos_mf = raw_mf.where(tp_d>0,0.0).rolling(14).sum()
    neg_mf = raw_mf.where(tp_d<0,0.0).rolling(14).sum().replace(0,np.nan)
    df["mfi_14"] = (100-(100/(1+pos_mf/neg_mf))).astype("float64")

    bb_ma  = df["btc_close"].rolling(20).mean()
    bb_std = df["btc_close"].rolling(20).std()
    bb_up  = bb_ma+2*bb_std
    bb_lo  = bb_ma-2*bb_std
    df["bb_width"] = (bb_up-bb_lo)/bb_ma
    df["bb_pct_b"] = (df["btc_close"]-bb_lo)/(bb_up-bb_lo).replace(0,np.nan)

    ema20  = df["btc_close"].ewm(span=20,adjust=False).mean()
    tr     = pd.concat([
                 df["btc_high"]-df["btc_low"],
                 (df["btc_high"]-df["btc_close"].shift()).abs(),
                 (df["btc_low"] -df["btc_close"].shift()).abs(),
             ],axis=1).max(axis=1)
    atr10  = tr.rolling(10).mean()
    kc_up  = ema20+2*atr10; kc_lo = ema20-2*atr10
    df["kc_width"] = (kc_up-kc_lo)/ema20
    df["kc_pct"]   = (df["btc_close"]-kc_lo)/(kc_up-kc_lo).replace(0,np.nan)

    df["psar"]      = _parabolic_sar(df["btc_high"],df["btc_low"])
    df["psar_dist"] = (df["btc_close"]-df["psar"])/df["btc_close"]

    df["raw_target"]  = (df["btc_close"].shift(-horizon)/df["btc_close"])-1
    df["rolling_vol"] = df["btc_ret_1d"].shift(1).rolling(30).std()
    df["target_norm"] = df["raw_target"]/df["rolling_vol"]

    return df.dropna(subset=FEATURE_COLS+["rolling_vol"]).copy()


# ═══════════════════════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════════════════════
def train_model(ctx_df) -> object:
    log.info("Downloading YF BTC-USD for model training...")
    yf_raw = _yf_download_with_retry(
        ["BTC-USD","SPY","QQQ","GC=F"],
        start=CONFIG["yf_train_start"], end=CONFIG["yf_train_end"],
        auto_adjust=True, progress=False, group_by="ticker",
    )
    btc = yf_raw["BTC-USD"]
    idx = pd.to_datetime(btc.index).tz_localize(None)
    hl_mock = pd.DataFrame({
        "timestamp": idx, "open": btc["Open"].values,
        "high": btc["High"].values, "low": btc["Low"].values,
        "close": btc["Close"].values, "volume": btc["Volume"].values,
    })
    train_ctx = ctx_df[ctx_df.index <= pd.Timestamp(CONFIG["yf_train_end"])]
    train_df  = build_features(hl_mock, train_ctx, horizon=CONFIG["horizon_days"])
    train_df  = train_df.dropna(subset=["target_norm"])

    from xgboost import XGBRegressor
    model = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.03,
                          subsample=0.8, colsample_bytree=0.8,
                          objective="reg:squarederror", random_state=42)
    model.fit(train_df[FEATURE_COLS], train_df["target_norm"])
    log.info(f"Model trained on {len(train_df)} rows.")
    return model


def load_or_train_model(ctx_df, state) -> object:
    fp         = _path(CONFIG["model_file"])
    last_train = state.get("model_trained_date")
    retrain_n  = CONFIG["retrain_every_days"]
    needs_train = True
    if os.path.exists(fp) and last_train:
        days_since = (datetime.now() - datetime.fromisoformat(last_train)).days
        if retrain_n == 0 or days_since < retrain_n:
            needs_train = False
    if needs_train:
        model = train_model(ctx_df)
        with open(fp, "wb") as f:
            pickle.dump(model, f)
        state["model_trained_date"] = datetime.now().isoformat()
        log.info("Model saved.")
    else:
        with open(fp, "rb") as f:
            model = pickle.load(f)
        log.info(f"Cached model loaded (trained {last_train[:10]}).")
    return model


def compute_threshold(state, new_pred):
    history = state.get("prediction_history", [])
    history.append(float(new_pred))
    history = history[-90:]
    state["prediction_history"] = history
    if len(history) < 20:
        threshold = 0.013
        log.warning(f"Prediction history only {len(history)} rows — using fallback threshold.")
    else:
        threshold = float(np.quantile(history, CONFIG["quantile_entry"]))
    sl_magnitude = abs(threshold) / 2
    return threshold, sl_magnitude


# ═══════════════════════════════════════════════════════════════════════════
# ── LOOP 1 : DAILY ENTRY LOOP ─────────────────────────────────────────────
# Runs once per day at 00:05 UTC. Checks for new entry signal only.
# Exit is handled entirely by the 1-min monitor loop.
# ═══════════════════════════════════════════════════════════════════════════
def daily_entry_loop():
    log.info("Daily entry loop started.")
    while True:
        now_utc = datetime.now(timezone.utc)

        # Wait until 00:05 UTC
        target_h = CONFIG["daily_check_hour_utc"]
        target_m = CONFIG["daily_check_minute_utc"]
        is_trigger_time = (now_utc.hour == target_h and now_utc.minute == target_m)

        if is_trigger_time:
            today_str = now_utc.strftime("%Y-%m-%d")

            with _state_lock:
                state = load_state()
                already_ran = (state.get("daily_loop_last_run") == today_str)

            if not already_ran:
                log.info(f"[DAILY] Running entry check for {today_str}")
                try:
                    _run_daily_entry(today_str)
                except Exception as e:
                    log.error(f"[DAILY] Error: {e}", exc_info=True)

        # Sleep 50 seconds, then re-check — tight enough to never miss the minute
        time.sleep(50)


def _run_daily_entry(today_str: str):
    # ── fetch data ─────────────────────────────────────────────────────────
    ctx_df = fetch_yf_context(start=CONFIG["yf_train_start"])
    hl_df  = fetch_daily_ohlcv(CONFIG["hl_lookback_days"])

    if len(hl_df) < 60:
        log.error("[DAILY] Not enough daily candles. Skipping.")
        return

    feat_df = build_features(hl_df, ctx_df, horizon=CONFIG["horizon_days"])
    if feat_df.empty:
        log.error("[DAILY] Feature frame empty. Skipping.")
        return

    latest        = feat_df.iloc[-1]
    current_price = float(latest["btc_close"])
    latest_date   = feat_df.index[-1]
    rolling_vol   = float(latest["rolling_vol"])

    # ── model ──────────────────────────────────────────────────────────────
    with _state_lock:
        state = load_state()

    model      = load_or_train_model(ctx_df, state)
    pred_norm  = float(model.predict(feat_df[FEATURE_COLS].iloc[[-1]])[0])
    prediction = pred_norm * rolling_vol
    threshold, sl_magnitude = compute_threshold(state, prediction)

    log.info(f"[DAILY] {latest_date.date()}  BTC=${current_price:,.0f}  "
             f"pred={prediction:.4f}  threshold={threshold:.4f}")

    with _state_lock:
        state = load_state()   # re-read in case monitor loop modified it

        # ── only enter if no open position ─────────────────────────────────
        if not state["in_position"]:
            if prediction > threshold:
                wallet        = state["wallet"]
                position_usdc = wallet * CONFIG["position_size_pct"] / 100
                amount        = position_usdc * CONFIG["leverage"]
                fee_entry     = amount * CONFIG["trade_fee_pct"] / 100
                quantity_btc  = (amount - fee_entry) / current_price
                wallet       -= position_usdc

                state.update({
                    "in_position":        True,
                    "entry_price":        current_price,
                    "entry_date":         str(latest_date.date()),
                    "entry_datetime_utc": datetime.now(timezone.utc).isoformat(),
                    "target_return":      prediction,
                    "sl_magnitude":       sl_magnitude,
                    "position_usdc":      position_usdc,
                    "quantity_btc":       quantity_btc,
                    "fee_entry":          fee_entry,
                    "wallet":             wallet,
                })
                if wallet > state["last_ath"]:
                    state["last_ath"] = wallet

                event = {
                    "event":          "long_entry",
                    "datetime_utc":   datetime.now(timezone.utc).isoformat(),
                    "entry_date":     str(latest_date.date()),
                    "price":          round(current_price, 2),
                    "prediction":     round(prediction, 6),
                    "threshold":      round(threshold, 6),
                    "sl_magnitude":   round(sl_magnitude, 6),
                    "sl_price":       round(current_price * (1 - sl_magnitude), 2),
                    "tp_price":       round(current_price * (1 + prediction), 2),
                    "horizon_exit_date": str(
                        (datetime.now(timezone.utc) +
                         timedelta(days=CONFIG["horizon_days"])).date()
                    ),
                    "position_usdc":  round(position_usdc, 4),
                    "quantity_btc":   round(quantity_btc, 8),
                    "fee_entry":      round(fee_entry, 4),
                    "wallet_after":   round(wallet, 4),
                }
                append_trade(event)
                log.info(
                    f"[DAILY] ✅ ENTERED LONG  price=${current_price:,.0f}  "
                    f"qty={quantity_btc:.6f} BTC  "
                    f"SL=${current_price*(1-sl_magnitude):,.0f}  "
                    f"TP=${current_price*(1+prediction):,.0f}  "
                    f"wallet={wallet:.2f}"
                )
            else:
                log.info(f"[DAILY] No signal. pred={prediction:.4f} ≤ "
                         f"threshold={threshold:.4f}")
        else:
            log.info(f"[DAILY] Already in position. Skipping entry check.")

        state["daily_loop_last_run"] = today_str
        # Persisted regardless of entry decision, purely so an external
        # notifier (e.g. run_once.py's Telegram integration) can report
        # what the model saw today even on a "no signal" day.
        state["last_daily_check_date"]  = today_str
        state["last_daily_price"]       = current_price
        state["last_daily_prediction"]  = prediction
        state["last_daily_threshold"]   = threshold
        save_state(state)


# ═══════════════════════════════════════════════════════════════════════════
# ── LOOP 2 : 1-MINUTE EXIT MONITOR ───────────────────────────────────────
# Runs every 1 minutes. Checks exit criteria against latest 1-min close.
# Only acts if there is an open position.
# ═══════════════════════════════════════════════════════════════════════════
def monitor_exit_loop():
    log.info("1-min exit monitor loop started.")
    while True:
        try:
            _run_monitor_check()
        except Exception as e:
            log.error(f"[MONITOR] Error: {e}", exc_info=True)
        time.sleep(CONFIG["monitor_interval_sec"])


def _run_monitor_check():
    with _state_lock:
        state = load_state()

    # Nothing to monitor if no open position
    if not state["in_position"]:
        log.debug("[MONITOR] No open position. Skipping.")
        return

    # ── Fetch latest closed 1-min candle ─────────────────────────────────
    candle = fetch_latest_1min_candle()
    if candle is None:
        log.warning("[MONITOR] Could not fetch 1-min candle.")
        return

    candle_ts     = candle["timestamp"]
    current_price = candle["close"]
    candle_low    = candle["low"]

    entry_price   = state["entry_price"]
    target_return = state["target_return"]
    sl_magnitude  = state["sl_magnitude"]
    entry_dt      = datetime.fromisoformat(state["entry_datetime_utc"])
    hours_held    = (datetime.now(timezone.utc) - entry_dt.replace(tzinfo=timezone.utc)).total_seconds() / 3600
    days_held     = hours_held / 24

    # ── Compute returns ────────────────────────────────────────────────────
    current_return     = (current_price / entry_price) - 1
    intraday_low_ret   = (candle_low    / entry_price) - 1

    # ── Exit criteria (checked in priority order) ──────────────────────────
    exit_reason = None
    exit_price  = current_price

    if intraday_low_ret < -sl_magnitude:
        # Price wicked below SL within this 1-min candle
        exit_price  = entry_price * (1 - sl_magnitude)
        exit_reason = "stop_loss"
    elif current_return >= target_return:
        exit_reason = "target_reached"
    elif days_held >= CONFIG["horizon_days"]:
        exit_reason = "horizon_expired"

    # ── Log this check ─────────────────────────────────────────────────────
    monitor_row = {
        "datetime_utc":      datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        "candle_close_utc":  str(candle_ts),
        "current_price":     round(current_price, 2),
        "candle_low":        round(candle_low, 2),
        "entry_price":       round(entry_price, 2),
        "current_return_pct": round(current_return * 100, 4),
        "intraday_low_ret_pct": round(intraday_low_ret * 100, 4),
        "target_return_pct": round(target_return * 100, 4),
        "sl_pct":            round(-sl_magnitude * 100, 4),
        "sl_price":          round(entry_price * (1 - sl_magnitude), 2),
        "tp_price":          round(entry_price * (1 + target_return), 2),
        "days_held":         round(days_held, 3),
        "horizon_days":      CONFIG["horizon_days"],
        "exit_triggered":    exit_reason or "none",
    }
    append_monitor_log(monitor_row)

    # ── Print a clean status line ──────────────────────────────────────────
    status_symbol = "🔴 EXIT" if exit_reason else "🟡 HOLD"
    log.info(
        f"[MONITOR] {status_symbol}  "
        f"price=${current_price:,.0f}  "
        f"ret={current_return*100:+.2f}%  "
        f"SL={-sl_magnitude*100:.2f}%  "
        f"TP={target_return*100:.2f}%  "
        f"held={days_held:.1f}d  "
        + (f"→ {exit_reason}" if exit_reason else "")
    )

    # ── Execute exit if triggered ──────────────────────────────────────────
    if exit_reason:
        with _state_lock:
            state = load_state()
            wallet        = state["wallet"]
            position_usdc = state["position_usdc"]
            quantity_btc  = state["quantity_btc"]
            fee_entry     = state["fee_entry"]
            last_ath      = state["last_ath"]

            pnl_gross = (exit_price - entry_price) * quantity_btc
            fee_exit  = quantity_btc * exit_price * CONFIG["trade_fee_pct"] / 100
            net_pnl   = pnl_gross - fee_entry - fee_exit
            wallet   += position_usdc + pnl_gross - fee_exit

            # ── Equity snapshot at close ───────────────────────────────────
            hodl_qty    = CONFIG["initial_capital"] / entry_price   # approx
            hodl_value  = hodl_qty * current_price
            if wallet > last_ath:
                last_ath = wallet
            drawdown = (wallet - last_ath) / last_ath if last_ath else 0

            equity_row = {
                "datetime_utc":  datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
                "event":         "trade_closed",
                "btc_price":     round(current_price, 2),
                "net_pnl":       round(net_pnl, 4),
                "wallet":        round(wallet, 4),
                "hodl_value":    round(hodl_value, 4),
                "bot_pct":       round((wallet/CONFIG["initial_capital"]-1)*100, 3),
                "hodl_pct":      round((hodl_value/CONFIG["initial_capital"]-1)*100, 3),
                "drawdown_pct":  round(drawdown*100, 3),
                "exit_reason":   exit_reason,
            }
            append_equity_log(equity_row)

            trade_event = {
                "event":            "long_exit",
                "datetime_utc":     datetime.now(timezone.utc).isoformat(),
                "exit_reason":      exit_reason,
                "entry_price":      round(entry_price, 2),
                "exit_price":       round(exit_price, 2),
                "entry_date":       state["entry_date"],
                "entry_datetime":   state["entry_datetime_utc"],
                "days_held":        round(days_held, 3),
                "current_return_pct": round(current_return*100, 3),
                "pnl_gross":        round(pnl_gross, 4),
                "fee_entry":        round(fee_entry, 4),
                "fee_exit":         round(fee_exit, 4),
                "net_pnl":          round(net_pnl, 4),
                "wallet":           round(wallet, 4),
            }
            append_trade(trade_event)

            log.info(
                f"[MONITOR] ✅ CLOSED  [{exit_reason}]  "
                f"${entry_price:,.0f} → ${exit_price:,.0f}  "
                f"pnl={net_pnl:+.2f} USDC  wallet={wallet:.2f}"
            )

            state.update({
                "in_position":        False,
                "entry_price":        None,
                "entry_date":         None,
                "entry_datetime_utc": None,
                "target_return":      None,
                "sl_magnitude":       None,
                "position_usdc":      None,
                "quantity_btc":       None,
                "fee_entry":          None,
                "wallet":             wallet,
                "last_ath":           last_ath,
            })
            save_state(state)


# ═══════════════════════════════════════════════════════════════════════════
# REPORT
# ═══════════════════════════════════════════════════════════════════════════
def print_report():
    trades_fp = _path(CONFIG["trade_log_file"])
    equity_fp = _path(CONFIG["equity_log_file"])
    monitor_fp = _path(CONFIG["monitor_log_file"])

    print(f"\n{'='*60}")
    print("  SHADOW FORWARD TEST — REPORT")
    print(f"{'='*60}")

    if os.path.exists(trades_fp):
        with open(trades_fp) as f:
            trades = json.load(f)
        exits   = [t for t in trades if t["event"] == "long_exit"]
        entries = [t for t in trades if t["event"] == "long_entry"]
        print(f"\n  Total entries : {len(entries)}")
        print(f"  Total exits   : {len(exits)}")
        if exits:
            pnls    = [e["net_pnl"] for e in exits]
            wins    = [p for p in pnls if p > 0]
            losses  = [p for p in pnls if p <= 0]
            reasons = {}
            for e in exits:
                r = e.get("exit_reason","?")
                reasons[r] = reasons.get(r, 0) + 1
            print(f"  Win rate      : {len(wins)/len(pnls)*100:.1f}%")
            print(f"  Avg win       : {np.mean(wins):.2f} USDC" if wins else "  Avg win       : n/a")
            print(f"  Avg loss      : {np.mean(losses):.2f} USDC" if losses else "  Avg loss      : n/a")
            print(f"  Best trade    : {max(pnls):.2f} USDC")
            print(f"  Worst trade   : {min(pnls):.2f} USDC")
            print(f"  Exit reasons  : {reasons}")

    if os.path.exists(equity_fp):
        eq = pd.read_csv(equity_fp)
        print(f"\n  Current wallet : {eq['wallet'].iloc[-1]:.2f} USDC  "
              f"({(eq['wallet'].iloc[-1]/CONFIG['initial_capital']-1)*100:+.2f}%)")
        print(f"  HODL value     : {eq['hodl_value'].iloc[-1]:.2f} USDC  "
              f"({(eq['hodl_value'].iloc[-1]/CONFIG['initial_capital']-1)*100:+.2f}%)")
        print(f"  Max drawdown   : {eq['drawdown_pct'].min():.2f}%")

    if os.path.exists(monitor_fp):
        mon = pd.read_csv(monitor_fp)
        print(f"\n  Monitor checks run : {len(mon)}")
        print(f"  Last check         : {mon['datetime_utc'].iloc[-1]}")
        print(f"  Last price         : ${mon['current_price'].iloc[-1]:,.0f}")
        print(f"\n  Last 5 monitor checks:")
        print(mon[["datetime_utc","current_price","current_return_pct",
                    "sl_pct","target_return_pct","days_held",
                    "exit_triggered"]].tail(5).to_string(index=False))

    # Show state of current open position
    fp = _path(CONFIG["state_file"])
    if os.path.exists(fp):
        with open(fp) as f:
            state = json.load(f)
        print(f"\n{'='*60}")
        print(f"  CURRENT POSITION")
        print(f"{'='*60}")
        if state["in_position"]:
            ep    = state["entry_price"]
            sl    = state["sl_magnitude"]
            tp    = state["target_return"]
            print(f"  IN POSITION since {state['entry_date']}")
            print(f"  Entry price : ${ep:,.2f}")
            print(f"  SL price    : ${ep*(1-sl):,.2f}  ({-sl*100:.2f}%)")
            print(f"  TP price    : ${ep*(1+tp):,.2f}  ({tp*100:.2f}%)")
            print(f"  Wallet (excl. position): ${state['wallet']:.2f}")
        else:
            print(f"  No open position.  Wallet: ${state['wallet']:.2f}")


# ═══════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        print_report()
        sys.exit(0)

    # Save PID so you can stop it easily
    with open(_path("shadow.pid"), "w") as f:
        f.write(str(os.getpid()))
    log.info(f"PID {os.getpid()} written to shadow.pid")
    log.info("Starting shadow forward tester (multi-timeframe)...")
    log.info("  Entry  : daily    @ 00:05 UTC")
    log.info("  Exit   : 1-min   every 1 minutes")
    log.info("  Stop   : kill $(cat shadow.pid)")

    # Start both loops as daemon threads
    t_daily   = threading.Thread(target=daily_entry_loop,  daemon=True, name="DailyEntry")
    t_monitor = threading.Thread(target=monitor_exit_loop, daemon=True, name="MonitorExit")

    t_daily.start()
    t_monitor.start()

    # Keep main thread alive
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("Stopped by user.")
