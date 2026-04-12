"""
Step 66: Scan loop watchdog.
Detects a frozen scan loop (no heartbeat() call within 2× scan interval).
On freeze: log CRITICAL → Telegram alert → SIGTERM for graceful restart.

Market-hours guard: freeze check is SKIPPED outside market hours (weekdays 09:30–16:00 ET)
and on weekends — prevents false-positive restarts when the scanner is intentionally idle.
"""
import os
import signal
import threading
import time
import logging
import requests
from datetime import datetime, time as dtime
import pytz

logger = logging.getLogger(__name__)

_last_scan_ts: float = time.time()
_watchdog_started: bool = False
_lock = threading.Lock()

_ET = pytz.timezone("America/New_York")
_MARKET_OPEN  = dtime(9, 25)   # 5-min buffer before open
_MARKET_CLOSE = dtime(16, 5)   # 5-min buffer after close


def _is_market_hours() -> bool:
    """Returns True if current ET time is within trading hours on a weekday."""
    now_et = datetime.now(_ET)
    # Weekend → never market hours
    if now_et.weekday() >= 5:
        return False
    t = now_et.time()
    return _MARKET_OPEN <= t <= _MARKET_CLOSE


def heartbeat() -> None:
    """Call at the start of each scan cycle to reset the freeze timer."""
    global _last_scan_ts
    _last_scan_ts = time.time()


def _watchdog_loop(interval_s: int, service: str, bot_token: str, chat_id: str) -> None:
    """Daemon thread: fires every interval_s, raises alarm at 2× if no heartbeat.
    
    Only triggers freeze alert during market hours — outside hours the scanner
    is intentionally idle and should not be flagged as frozen.
    """
    deadline = 2 * interval_s
    logger.info("Scan watchdog running — freeze_limit=%ds (market-hours guard active)", deadline)
    while True:
        time.sleep(interval_s)

        # ── Market-hours guard: skip freeze check outside trading hours ──
        if not _is_market_hours():
            # Reset the timestamp so we don't accumulate stale elapsed time
            heartbeat()
            logger.debug("Watchdog: market closed — resetting timer, skip freeze check")
            continue

        elapsed = time.time() - _last_scan_ts
        if elapsed > deadline:
            msg = (
                f"\u26a0\ufe0f {service}: SCAN LOOP FROZEN \u2014 no cycle for {elapsed:.0f}s "
                f"(limit {deadline}s). Triggering restart."
            )
            logger.critical(msg)
            try:
                if bot_token and bot_token not in ("", "DISABLED"):
                    requests.post(
                        f"https://api.telegram.org/bot{bot_token}/sendMessage",
                        json={"chat_id": chat_id, "text": msg},
                        timeout=5,
                    )
            except Exception:
                pass
            try:
                os.kill(os.getpid(), signal.SIGTERM)
            except Exception:
                pass


def start(interval_s: int, service: str, bot_token: str = "", chat_id: str = "-5130564161") -> threading.Thread:
    """Launch scan watchdog daemon thread. Idempotent — safe to call multiple times."""
    global _watchdog_started
    with _lock:
        if _watchdog_started:
            return None  # type: ignore[return-value]
        _watchdog_started = True
    t = threading.Thread(
        target=_watchdog_loop,
        args=(interval_s, service, bot_token, chat_id),
        daemon=True,
        name="scan-watchdog",
    )
    t.start()
    logger.info(
        "\u2705 Scan loop watchdog started (interval=%ds, freeze_limit=%ds, market-hours guard=ON)",
        interval_s, 2 * interval_s,
    )
    return t
