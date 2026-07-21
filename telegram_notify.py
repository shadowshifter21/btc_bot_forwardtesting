"""
Telegram Notifications — Optional, Safe No-Op if Unconfigured
================================================================
Sends a Telegram message whenever:
  1. The daily entry check runs (once/day) — reports whether a trade was
     opened, no signal fired, or the bot was already holding a position.
  2. A trade closes (checked every monitor tick) — reports exit reason,
     entry/exit price, P&L, and updated wallet balance.

SETUP (one-time):
  1. Open Telegram, message @BotFather, send /newbot, follow the prompts.
     BotFather gives you a bot token like: 123456789:AAExampleTokenHere
  2. Message your new bot anything (e.g. "hi") so it can see your chat.
  3. Visit https://api.telegram.org/bot<YOUR_TOKEN>/getUpdates in a browser
     — find "chat":{"id": ...} in the JSON response. That number is your
     chat ID.
  4. In your GitHub repo: Settings -> Secrets and variables -> Actions
     -> New repository secret. Add two secrets:
       TELEGRAM_BOT_TOKEN = <token from step 1>
       TELEGRAM_CHAT_ID   = <chat id from step 3>

If these two environment variables aren't set, every function below
no-ops silently (with a log line) — the bot runs exactly as before,
just without notifications. Nothing else depends on this module.
"""

import os
import logging

import requests

log = logging.getLogger("telegram_notify")

TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID    = os.environ.get("TELEGRAM_CHAT_ID")

API_BASE = "https://api.telegram.org"


def _configured() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram_message(text: str) -> bool:
    """
    Send a message to the configured Telegram chat. Returns True on success,
    False on any failure (network issue, bad token, etc.) — never raises,
    since a notification failure should never take down the trading bot.
    """
    if not _configured():
        log.info("Telegram not configured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID "
                 "missing) — skipping notification.")
        return False

    url = f"{API_BASE}/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {
        "chat_id": TELEGRAM_CHAT_ID,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        resp = requests.post(url, json=payload, timeout=15)
        if resp.status_code == 200:
            log.info("Telegram notification sent.")
            return True
        else:
            log.warning(f"Telegram send failed: HTTP {resp.status_code} — {resp.text[:200]}")
            return False
    except Exception as e:
        log.warning(f"Telegram send error: {e}")
        return False


def get_new_updates(offset: int | None = None) -> list:
    """
    Fetch new incoming messages since `offset` (Telegram's manual-polling
    pattern — an alternative to webhooks that fits our ephemeral,
    scheduled-run architecture perfectly: each run does one quick,
    non-blocking fetch of whatever arrived since last time).

    timeout=0 makes this return immediately with whatever's available,
    rather than long-polling and holding the GitHub Actions job open.

    Returns an empty list (not an error) if unconfigured or on any failure —
    a failed poll should never break the trading logic that runs alongside it.
    """
    if not _configured():
        return []

    url = f"{API_BASE}/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
    params = {"timeout": 0}
    if offset is not None:
        params["offset"] = offset

    try:
        resp = requests.get(url, params=params, timeout=15)
        if resp.status_code != 200:
            log.warning(f"getUpdates failed: HTTP {resp.status_code}")
            return []
        data = resp.json()
        return data.get("result", [])
    except Exception as e:
        log.warning(f"getUpdates error: {e}")
        return []


def extract_commands(updates: list) -> list:
    """
    Filter raw Telegram updates down to text messages from OUR configured
    chat only (defense in depth — ignores messages from any other chat even
    if something unexpected ever messages this bot), returning a simple list
    of {"update_id":..., "text":...} dicts.
    """
    commands = []
    for update in updates:
        message = update.get("message", {})
        chat_id = str(message.get("chat", {}).get("id", ""))
        text = message.get("text", "")

        if not text:
            continue
        if TELEGRAM_CHAT_ID and chat_id != str(TELEGRAM_CHAT_ID):
            log.warning(f"Ignoring message from unrecognized chat_id={chat_id}")
            continue

        commands.append({"update_id": update["update_id"], "text": text.strip()})
    return commands


# ── Message formatters — pure functions, easy to test/tweak independently ──

def format_trade_opened(entry_event: dict) -> str:
    return (
        f"🟢 <b>Trade Opened</b>\n"
        f"Date: {entry_event.get('entry_date')}\n"
        f"Entry Price: ${entry_event.get('price', 0):,.2f}\n"
        f"Quantity: {entry_event.get('quantity_btc', 0):.6f} BTC\n"
        f"Stop Loss: ${entry_event.get('sl_price', 0):,.2f}\n"
        f"Take Profit: ${entry_event.get('tp_price', 0):,.2f}\n"
        f"Prediction: {entry_event.get('prediction', 0):.4f} "
        f"(threshold {entry_event.get('threshold', 0):.4f})\n"
        f"Horizon exit by: {entry_event.get('horizon_exit_date')}\n"
        f"Wallet after entry: ${entry_event.get('wallet_after', 0):,.2f}"
    )


def format_no_signal(check_date: str, price: float, prediction: float, threshold: float) -> str:
    return (
        f"⚪ <b>No Trade Today</b>\n"
        f"Date: {check_date}\n"
        f"BTC Price: ${price:,.2f}\n"
        f"Prediction: {prediction:.4f} (threshold {threshold:.4f})\n"
        f"Signal did not clear the entry threshold — staying in cash."
    )


def format_already_in_position(check_date: str, entry_date: str, price: float) -> str:
    return (
        f"🔵 <b>Daily Check — Already In Position</b>\n"
        f"Date: {check_date}\n"
        f"BTC Price: ${price:,.2f}\n"
        f"Holding since: {entry_date}"
    )


def format_trade_closed(exit_event: dict) -> str:
    pnl = exit_event.get("net_pnl", 0)
    pnl_symbol = "🟢" if pnl >= 0 else "🔴"
    return (
        f"🔴 <b>Trade Closed</b>\n"
        f"Reason: {exit_event.get('exit_reason')}\n"
        f"Entry: ${exit_event.get('entry_price', 0):,.2f} → "
        f"Exit: ${exit_event.get('exit_price', 0):,.2f}\n"
        f"Held: {exit_event.get('days_held', 0):.2f} days\n"
        f"Return: {exit_event.get('current_return_pct', 0):+.2f}%\n"
        f"{pnl_symbol} P&amp;L: ${pnl:+,.2f}\n"
        f"Wallet: ${exit_event.get('wallet', 0):,.2f}"
    )


def format_status_position(state: dict, current_price: float | None) -> str:
    """
    Formats the "ongoing trade" section of /status: entry details and
    unrealized P&L if a position is currently open, or a plain "flat"
    message otherwise. `current_price` should come from the most recent
    monitor tick (at most ~5 minutes stale, matching the bot's own polling
    cadence) — pass None if unavailable and this will note that plainly
    rather than guessing.
    """
    if not state.get("in_position"):
        return "📭 <b>No open position</b> — currently flat."

    entry_price = state.get("entry_price", 0) or 0
    quantity    = state.get("quantity_btc", 0) or 0
    sl_mag      = state.get("sl_magnitude", 0) or 0
    target_ret  = state.get("target_return", 0) or 0

    lines = [
        "📈 <b>Open Position</b>",
        f"Entry: ${entry_price:,.2f} on {state.get('entry_date', '?')}",
        f"Quantity: {quantity:.6f} BTC",
        f"Stop Loss: ${entry_price * (1 - sl_mag):,.2f} ({-sl_mag*100:.2f}%)",
        f"Take Profit: ${entry_price * (1 + target_ret):,.2f} ({target_ret*100:.2f}%)",
    ]

    if current_price is not None and entry_price:
        unrealized_pct = (current_price / entry_price - 1) * 100
        unrealized_symbol = "🟢" if unrealized_pct >= 0 else "🔴"
        lines.append(f"Current Price: ${current_price:,.2f}")
        lines.append(f"{unrealized_symbol} Unrealized: {unrealized_pct:+.2f}%")
    else:
        lines.append("Current price: unavailable this check")

    return "\n".join(lines)


def format_status_overall(trades: list, equity_rows, initial_capital: float) -> str:
    """
    Formats the "overall bot" section of /status: wallet, HODL comparison,
    trade counts and win rate — the same figures print_report() shows,
    reformatted for a Telegram message.
    `equity_rows` is a pandas DataFrame read from shadow_equity_log.csv
    (or None if the file doesn't exist yet).
    """
    exits = [t for t in trades if t.get("event") == "long_exit"]
    entries = [t for t in trades if t.get("event") == "long_entry"]

    lines = ["📊 <b>Overall Status</b>"]

    if equity_rows is not None and len(equity_rows) > 0:
        wallet = equity_rows["wallet"].iloc[-1]
        hodl   = equity_rows["hodl_value"].iloc[-1]
        max_dd = equity_rows["drawdown_pct"].min()
        wallet_pct = (wallet / initial_capital - 1) * 100
        hodl_pct   = (hodl / initial_capital - 1) * 100
        lines.append(f"Wallet: ${wallet:,.2f} ({wallet_pct:+.2f}%)")
        lines.append(f"HODL benchmark: ${hodl:,.2f} ({hodl_pct:+.2f}%)")
        lines.append(f"Max drawdown: {max_dd:.2f}%")
    else:
        lines.append("Wallet: no equity history yet")

    lines.append(f"Total entries: {len(entries)}")
    lines.append(f"Total exits: {len(exits)}")

    if exits:
        pnls = [e.get("net_pnl", 0) for e in exits]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        win_rate = len(wins) / len(pnls) * 100
        lines.append(f"Win rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
        if wins:
            lines.append(f"Avg win: ${sum(wins)/len(wins):+,.2f}")
        if losses:
            lines.append(f"Avg loss: ${sum(losses)/len(losses):+,.2f}")

    return "\n".join(lines)
