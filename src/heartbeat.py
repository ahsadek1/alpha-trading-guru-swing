"""
src/heartbeat.py — Heartbeat tracker for scan loops.
Records last-scan timestamp so Guardian can detect stale loops.
"""
import time
import logging

logger = logging.getLogger(__name__)

_last_scan_ts: float = 0.0
_scan_count: int = 0


def record_scan():
    """Call at the end of every scan cycle."""
    global _last_scan_ts, _scan_count
    _last_scan_ts = time.time()
    _scan_count += 1
    logger.debug("Heartbeat scan #%d recorded", _scan_count)


def last_scan_age_seconds() -> float:
    """Seconds since the last scan completed. 0 if never scanned."""
    if _last_scan_ts == 0:
        return 0.0
    return time.time() - _last_scan_ts


def get_scan_count() -> int:
    return _scan_count


def health_fields() -> dict:
    """Inject into /health response."""
    return {
        "scan_count":        _scan_count,
        "last_scan_age_s":   round(last_scan_age_seconds(), 1),
        "scanner_healthy":   last_scan_age_seconds() < 900 or _scan_count == 0,
    }
