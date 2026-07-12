"""
GitHub Actions Single-Shot Entry Point
========================================
Replaces the infinite while-loop architecture (daily_entry_loop / monitor_exit_loop
in claude_forwardtesting_hyperliquid.py) with a single "do one check" pass,
designed to be invoked repeatedly by a scheduled GitHub Actions workflow
(e.g. every 5 or 10 minutes) instead of running as a continuously-alive process.

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

SCHEDULING NOTE
---------------
GitHub Actions cron does not guarantee exact-minute precision — runs can be
delayed, especially during periods of high platform load. The daily-check
gate below is deliberately written as "hasn't run yet today AND we're at or
past the target time" rather than "exactly at HH:MM", so a delayed run still
fires the daily check correctly instead of silently missing its only window.

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
always fresh. If you'd rather restore weekly-only retraining, see the note
in the README about committing the .pkl file instead.
"""

import sys
from datetime import datetime, timezone

import claude_forwardtesting_hyperliquid as bot


def maybe_run_daily_entry():
    """
    Run the daily entry check if (a) it hasn't already run today AND
    (b) we're at or past the configured target time. Safe to call from
    every invocation — it's a fast no-op once today's check has completed.
    """
    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")

    target_h = bot.CONFIG["daily_check_hour_utc"]
    target_m = bot.CONFIG["daily_check_minute_utc"]
    target_minutes_of_day = target_h * 60 + target_m
    now_minutes_of_day    = now_utc.hour * 60 + now_utc.minute

    state = bot.load_state()
    already_ran_today = (state.get("daily_loop_last_run") == today_str)

    if already_ran_today:
        bot.log.info(f"[DAILY] Already ran today ({today_str}). Skipping.")
        return

    if now_minutes_of_day < target_minutes_of_day:
        bot.log.info(
            f"[DAILY] Not yet time (target {target_h:02d}:{target_m:02d} UTC, "
            f"now {now_utc.hour:02d}:{now_utc.minute:02d} UTC). Skipping."
        )
        return

    bot.log.info(f"[DAILY] Running entry check for {today_str}")
    try:
        bot._run_daily_entry(today_str)
    except Exception as e:
        bot.log.error(f"[DAILY] Error: {e}", exc_info=True)


def run_monitor_check():
    """Always attempt one monitor tick. The underlying function is already
    a no-op internally if no position is open."""
    try:
        bot._run_monitor_check()
    except Exception as e:
        bot.log.error(f"[MONITOR] Error: {e}", exc_info=True)


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
