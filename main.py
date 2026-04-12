"""
ATG (Alpha Trading Guru) — FastAPI entry point v3.0

Endpoints:
  GET /         — service banner
  GET /health   — full system health + bandit stats
  POST /scan    — trigger manual EOD scan
  POST /monitor — trigger manual position monitor cycle
  GET /positions— list open positions
  GET /stats    — trade statistics

Background scheduler (APScheduler):
  - Every 15 min (market hours, Mon–Fri): position monitor
  - 15:30 ET weekdays: EOD scan
  - 16:30 ET weekdays: daily summary + snapshot
  - 16:00 ET Fridays:  weekly report
  - 09:30 ET weekdays: reset daily circuit breaker state
"""
import logging
import os
import sys
from contextlib import asynccontextmanager
from datetime import datetime

import pytz
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from fastapi import FastAPI

from src.self_evolving_orchestrator import ATGOrchestrator
from src.database import get_open_positions, get_trade_stats
from src.position_watchdog import start_position_watchdog
from config.settings import ALPACA_BASE_URL, ALPACA_API_KEY, ALPACA_SECRET_KEY
from src.startup_guard import validate as _startup_validate

_startup_validate()  # Crash fast if env vars missing


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)
ET  = pytz.timezone("America/New_York")

# ── Global orchestrator ───────────────────────────────────────────────────────
orchestrator = ATGOrchestrator()
scheduler    = AsyncIOScheduler(timezone=ET)


# ── Scheduled task wrappers ───────────────────────────────────────────────────

async def _task_monitor() -> None:
    """Async wrapper for position monitor (every 15 min)."""
    try:
        orchestrator.run_position_monitor()
    except Exception as e:
        log.error("Position monitor task error: %s", e)


async def _task_intraday_scan() -> None:
    """Async wrapper for intraday scan (every 30 min, 09:30–15:00 ET weekdays)."""
    # Step 66: reset scan loop freeze watchdog timer
    try:
        from src.scan_watchdog import heartbeat as _wdbeat
        _wdbeat()
    except Exception:
        pass
    try:
        orchestrator.run_eod_scan()
    except Exception as e:
        log.error("Intraday scan task error: %s", e)


async def _task_eod_scan() -> None:
    """Async wrapper for EOD scan (15:30 ET weekdays)."""
    try:
        orchestrator.run_eod_scan()
    except Exception as e:
        log.error("EOD scan task error: %s", e)


async def _task_daily_summary() -> None:
    """Async wrapper for daily summary (16:30 ET weekdays)."""
    try:
        orchestrator.run_daily_summary()
    except Exception as e:
        log.error("Daily summary task error: %s", e)


async def _task_weekly_report() -> None:
    """Async wrapper for weekly report (16:00 ET Fridays)."""
    try:
        orchestrator.run_weekly_report()
    except Exception as e:
        log.error("Weekly report task error: %s", e)


async def _task_reset_daily() -> None:
    """Async wrapper to reset daily circuit breaker state (09:30 ET weekdays)."""
    try:
        orchestrator.reset_daily_state()
    except Exception as e:
        log.error("Daily reset task error: %s", e)


# ── App lifespan (replaces deprecated startup/shutdown events) ────────────────



def _verify_capital_router_reachable(max_retries: int = 5, delay: int = 10) -> bool:
    """Boot gate: verify Capital Router is reachable before starting."""
    import requests, time
    cr_url = os.getenv("CAPITAL_ROUTER_URL", "https://capital-router-production.up.railway.app")
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(f"{cr_url}/health", timeout=5)
            if resp.status_code == 200:
                log.info("✅ Capital Router reachable at %s", cr_url)
                return True
        except Exception as e:
            log.warning("Capital Router check %d/%d failed: %s", attempt, max_retries, e)
        if attempt < max_retries:
            time.sleep(delay)
    log.error("❌ Capital Router unreachable after %d attempts — refusing to start", max_retries)
    return False

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init orchestrator + scheduler.  Shutdown: stop scheduler."""
    log.info("ATG v3.0 starting …")

    # Capital Router boot gate
    if not _verify_capital_router_reachable():
        sys.exit(1)

    orchestrator.startup()

    # ── Startup Reconciliation — align DB, CR, Alpaca before trading ──────────
    try:
        from src.startup_reconciler import StartupReconciler
        from src.database import get_open_positions
        # FIX [F3]: Wire real DB functions — stubs were preventing ghost trade cleanup
        from src.database import void_position as _void_pos, confirm_position_open as _confirm_pos, get_pending_positions as _get_pending
        _reconciler = StartupReconciler(
            alpaca_base_url=ALPACA_BASE_URL,
            alpaca_key=ALPACA_API_KEY,
            alpaca_secret=ALPACA_SECRET_KEY,
            cr_client=None,
            db_get_open_fn=get_open_positions,
            db_get_pending_fn=_get_pending,       # FIX [F3]: real pending query
            db_void_fn=_void_pos,                 # FIX [F3]: real void (was no-op stub)
            db_confirm_fn=_confirm_pos,           # FIX [F3]: real confirm (was no-op stub)
            system_id="ATG_SWING",
            notify_fn=None,
        )
        _rpt = _reconciler.run()
        log.info(
            "Startup reconciliation complete — ghosts=%d orphans=%d confirmed=%d voided=%d",
            len(_rpt["ghosts"]), len(_rpt["orphans"]),
            len(_rpt["pending_confirmed"]), len(_rpt["pending_voided"]),
        )
    except Exception as _exc:
        log.warning("Startup reconciliation failed (non-fatal): %s", _exc)

    # ── Step 30: OrderLifecycleWatcher — PENDING → OPEN/VOID tracker ────────
    try:
        from src.order_lifecycle import OrderLifecycleWatcher
        from src.database import get_pending_positions, void_position, confirm_position_open
        _order_watcher = OrderLifecycleWatcher(
            alpaca_base_url=ALPACA_BASE_URL,
            alpaca_key=ALPACA_API_KEY,
            alpaca_secret=ALPACA_SECRET_KEY,
            db_get_pending_fn=get_pending_positions,
            db_void_fn=void_position,
            db_confirm_fn=confirm_position_open,
            cr_client=None,
            system_id="ATG_SWING",
        )
        _order_watcher.start()
        log.info("✅ OrderLifecycleWatcher started (poll=30s)")
    except Exception as _olw_exc:
        log.warning("OrderLifecycleWatcher init failed (non-fatal): %s", _olw_exc)

    watchdog_thread = start_position_watchdog()
    log.info("Position Watchdog started (threshold=8%, poll=60s)")

    # Position monitor — every 15 min Mon–Fri 09:30–16:15 ET
    scheduler.add_job(
        _task_monitor, "cron",
        day_of_week="mon-fri", hour="9-16", minute="*/15",
        id="position_monitor",
    )
    # Intraday scan — every 30 min Mon–Fri 09:30–15:00 ET
    # Swing scanner uses daily/weekly bars that update throughout the day;
    # scanning every 30 min catches setups as they form rather than waiting for EOD.
    scheduler.add_job(
        _task_intraday_scan, "cron",
        day_of_week="mon-fri", hour="9-14", minute="0,30",
        id="intraday_scan",
    )
    # Also fire at 15:00 to catch final pre-EOD setups
    scheduler.add_job(
        _task_intraday_scan, "cron",
        day_of_week="mon-fri", hour=15, minute=0,
        id="intraday_scan_1500",
    )
    # EOD scan — 15:30 ET weekdays (final scan of the day)
    scheduler.add_job(
        _task_eod_scan, "cron",
        day_of_week="mon-fri", hour=15, minute=30,
        id="eod_scan",
    )
    # Daily summary — 16:30 ET weekdays
    scheduler.add_job(
        _task_daily_summary, "cron",
        day_of_week="mon-fri", hour=16, minute=30,
        id="daily_summary",
    )
    # Weekly report — Friday 16:00 ET
    scheduler.add_job(
        _task_weekly_report, "cron",
        day_of_week="fri", hour=16, minute=0,
        id="weekly_report",
    )
    # Daily circuit-breaker reset — 09:30 ET weekdays
    scheduler.add_job(
        _task_reset_daily, "cron",
        day_of_week="mon-fri", hour=9, minute=30,
        id="daily_reset",
    )
    # Weekly circuit breaker reset — Monday 09:30 ET
    async def _task_reset_weekly() -> None:
        try:
            orchestrator.reset_weekly_state()
        except Exception as e:
            log.error("Weekly reset task error: %s", e)
    scheduler.add_job(
        _task_reset_weekly, "cron",
        day_of_week="mon", hour=9, minute=30,
        id="weekly_reset",
    )

    scheduler.start()
    log.info("ATG scheduler started — %d jobs registered", len(scheduler.get_jobs()))

    # Step 66: Scan loop freeze watchdog (30-min interval, freeze at 60 min)
    try:
        import os as _os
        from src.scan_watchdog import start as _sw_start
        _sw_start(
            interval_s=1800,
            service="ATG_SWING",
            bot_token=_os.getenv("TELEGRAM_BOT_TOKEN", ""),
            chat_id=_os.getenv("TELEGRAM_CHAT_ID", "-5130564161"),
        )
    except Exception as _sw_e:
        log.warning("Scan watchdog failed to start: %s", _sw_e)

    yield

    scheduler.shutdown(wait=False)
    log.info("ATG scheduler stopped")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Alpha Trading Guru — Swing System",
    version="3.0.0",
    lifespan=lifespan,
)


@app.get("/")
async def root() -> dict:
    """Service identification banner."""
    return {
        "service": "Alpha Trading Guru — ATG Swing",
        "version": "3.0.0",
        "system":  "ATG_SWING",
        "status":  "running",
        "ts":      datetime.now(ET).isoformat(),
    }


@app.get("/health")
async def health() -> dict:
    """Full system health check with bandit + trade stats."""
    return orchestrator.health_data()


@app.post("/scan")
async def trigger_scan() -> dict:
    """Manually trigger an EOD scan cycle."""
    log.info("Manual scan triggered via /scan")
    try:
        orchestrator.run_eod_scan()
        return {"status": "ok", "message": "EOD scan completed"}
    except Exception as e:
        log.error("Manual scan failed: %s", e)
        return {"status": "error", "message": str(e)}


@app.post("/monitor")
async def trigger_monitor() -> dict:
    """Manually trigger a position monitor cycle."""
    log.info("Manual monitor triggered via /monitor")
    try:
        orchestrator.run_position_monitor()
        return {"status": "ok", "message": "Position monitor completed"}
    except Exception as e:
        log.error("Manual monitor failed: %s", e)
        return {"status": "error", "message": str(e)}


@app.get("/positions")
async def get_positions() -> dict:
    """Return all currently open positions."""
    positions = get_open_positions()
    return {"count": len(positions), "positions": positions}


@app.get("/stats")
async def get_stats() -> dict:
    """Return aggregate trade statistics."""
    stats  = get_trade_stats()
    best   = orchestrator.bandit.best_setup() if orchestrator.bandit else {}
    return {**stats, "best_setup": best, "phase": orchestrator.phase}


if __name__ == "__main__":
    import uvicorn
    import os
    import signal
    import sys

    def _handle_sigterm(signum, frame):
        log.info("SIGTERM received — shutting down gracefully")
        try:
            scheduler.shutdown(wait=False)
        except Exception:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _handle_sigterm)
    signal.signal(signal.SIGINT, _handle_sigterm)

    port = int(os.getenv("PORT", "8080"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, log_level="warning", access_log=False)
