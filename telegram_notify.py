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
