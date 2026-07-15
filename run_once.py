"""
GitHub Actions Single-Shot Entry Point (+ Telegram Notifications)
====================================================================
Replaces the infinite while-loop architecture (daily_entry_loop / monitor_exit_loop
in claude_forwardtesting_hyperliquid.py) with a single "do one check" pass,
designed to be invoked repeatedly by a scheduled GitHub Actions workflow
(triggered externally, e.g. by cron-job.org every 5 minutes) instead of
running as a continuously-alive process.

WHY THIS EXISTS
---------------
GitHub Actions runners are ephemeral: each scheduled run spins up a fresh
container, runs to completion, and is destroyed. There is no way to keep an
infinite while-loop with time.sleep() alive between runs. Instead:

  - This script runs ONCE per invocation.
  - It always performs one MONITOR check (cheap — internally a no-op if no
    position is currently open, exactly matching the original loop's logic).
  - It performs the DAILY entry check only if today's daily check hasn't
    already run (tracked via state["daily_loop_last_run"], the same field
    the original while-loop used to prevent double-firing).
  - All state/log files are written locally by the unchanged bot functions;
    the GitHub Actions workflow YAML then commits and pushes them back to
    the repo, so the next scheduled run picks up exactly where this one
    left off.

NONE OF YOUR TRADING LOGIC IS MODIFIED. This file only calls the existing
_run_daily_entry() and _run_monitor_check() functions from
claude_forwardtesting_hyperliquid.py — it does not reimplement them.

TELEGRAM NOTIFICATIONS
-----------------------
After the daily check runs (once/day), a message is sent reporting one of:
  - a trade was opened (with entry/SL/TP/prediction details)
  - no signal fired (with the day's prediction vs threshold)
  - the bot was already holding a position (so you know the check ran)
After any monitor tick that closes a trade, a separate message reports the
exit reason, entry/exit price, P&L, and updated wallet balance.
See telegram_notify.py's docstring for one-time setup (BotFather + chat ID
+ GitHub repo secrets). If unconfigured, notifications silently no-op —
the bot's trading logic is completely unaffected either way.

SCHEDULING NOTE
---------------
GitHub Actions' own `schedule:` cron is unreliable at high frequency (runs
can be delayed by hours, not just minutes, on free/personal-tier repos).
This deployment relies on an EXTERNAL scheduler (cron-job.org) calling
`workflow_dispatch` on a real schedule instead — see the workflow YAML's
comments. The daily-check gate below is still written as "hasn't run yet
today AND we're at or past the target time" rather than "exactly at HH:MM",
so any residual delay still fires the daily check correctly rather than
silently missing its window.

MODEL RETRAINING NOTE
----------------------
The original script caches the trained model (shadow_model.pkl) locally and
only retrains every CONFIG["retrain_every_days"] (default 7) days. Because
GitHub Actions runners are ephemeral and this deployment intentionally does
NOT commit the (binary, repo-bloating) .pkl file back to git, that cache
does not persist between runs — so under this deployment, the model is
effectively retrained every time the daily entry check fires (once a day),
not once a week. This costs a bit more compute time per run (still well
within GitHub Actions' free minutes) but keeps the repo clean and the model
always fresh.
"""

import sys
import json
from datetime import datetime, timezone

import claude_forwardtesting_hyperliquid as bot
import telegram_notify as tg


def _get_last_trade_event() -> dict | None:
    """Read the most recently appended event from the trade log JSON file."""
    fp = bot._path(bot.CONFIG["trade_log_file"])
    try:
        with open(fp) as f:
            trades = json.load(f)
        return trades[-1] if trades else None
    except (FileNotFoundError, json.JSONDecodeError, IndexError):
        return None


def maybe_run_daily_entry():
    """
    Run the daily entry check if (a) it hasn't already run today AND
    (b) we're at or past the configured target time. Safe to call from
    every invocation — it's a fast no-op once today's check has completed.

    Sends a Telegram notification reporting the outcome (trade opened / no
    signal / already in position) whenever the check actually executes.
    """
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")

    target_h = bot.CONFIG["daily_check_hour_utc"]
    target_m = bot.CONFIG["daily_check_minute_utc"]
    target_minutes_of_day = target_h * 60 + target_m
    now_minutes_of_day    = now_utc.hour * 60 + now_utc.minute

    state_before = bot.load_state()
    already_ran_today = (state_before.get("daily_loop_last_run") == today_str)

    if already_ran_today:
        bot.log.info(f"[DAILY] Already ran today ({today_str}). Skipping.")
        return

    if now_minutes_of_day < target_minutes_of_day:
        bot.log.info(
            f"[DAILY] Not yet time (target {target_h:02d}:{target_m:02d} UTC, "
            f"now {now_utc.hour:02d}:{now_utc.minute:02d} UTC). Skipping."
        )
        return

    was_in_position = bool(state_before.get("in_position"))

    bot.log.info(f"[DAILY] Running entry check for {today_str}")
    try:
        bot._run_daily_entry(today_str)
    except Exception as e:
        bot.log.error(f"[DAILY] Error: {e}", exc_info=True)
        tg.send_telegram_message(
            f"⚠️ <b>Daily Entry Check Error</b>\nDate: {today_str}\n{e}"
        )
        return

    # ── Determine outcome and notify ────────────────────────────────────
    state_after = bot.load_state()
    is_in_position_now = bool(state_after.get("in_position"))

    if was_in_position:
        # Entry logic is skipped entirely when already holding a position —
        # state won't have changed on this path.
        msg = tg.format_already_in_position(
            check_date=today_str,
            entry_date=state_after.get("entry_date", "unknown"),
            price=state_after.get("last_daily_price", 0.0),
        )
    elif is_in_position_now:
        # Flipped from no-position to in-position: a trade was just opened.
        entry_event = _get_last_trade_event()
        if entry_event and entry_event.get("event") == "long_entry":
            msg = tg.format_trade_opened(entry_event)
        else:
            # Defensive fallback if the trade log couldn't be read for some reason.
            msg = (f"🟢 <b>Trade Opened</b>\nDate: {today_str}\n"
                   f"(Details unavailable — check shadow_trade_log.json)")
    else:
        # Stayed flat: no signal fired.
        msg = tg.format_no_signal(
            check_date=today_str,
            price=state_after.get("last_daily_price", 0.0),
            prediction=state_after.get("last_daily_prediction", 0.0),
            threshold=state_after.get("last_daily_threshold", 0.0),
        )

    tg.send_telegram_message(msg)


def run_monitor_check():
    """
    Always attempt one monitor tick. The underlying function is already a
    no-op internally if no position is open. Sends a Telegram notification
    if this tick closes a trade.
    """
    state_before = bot.load_state()
    was_in_position = bool(state_before.get("in_position"))

    try:
        bot._run_monitor_check()
    except Exception as e:
        bot.log.error(f"[MONITOR] Error: {e}", exc_info=True)
        return

    if not was_in_position:
        return   # nothing was open, nothing could have closed

    state_after = bot.load_state()
    is_in_position_now = bool(state_after.get("in_position"))

    if was_in_position and not is_in_position_now:
        # Position flipped from open to closed on this tick — a trade just exited.
        exit_event = _get_last_trade_event()
        if exit_event and exit_event.get("event") == "long_exit":
            tg.send_telegram_message(tg.format_trade_closed(exit_event))
        else:
            tg.send_telegram_message(
                "🔴 <b>Trade Closed</b>\n(Details unavailable — check shadow_trade_log.json)"
            )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "report":
        bot.print_report()
        sys.exit(0)

    bot.log.info("=" * 60)
    bot.log.info("GitHub Actions single-shot run starting")
    bot.log.info("=" * 60)

    maybe_run_daily_entry()
    run_monitor_check()

    bot.log.info("Single-shot run complete.")

    # Print a full status report at the end of EVERY run (not just when
    # invoked with the "report" argument), so results are always visible
    # directly in this run's GitHub Actions log — no separate step needed.
    print()
    bot.print_report()
