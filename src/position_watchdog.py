"""
Position Watchdog — Block 1 Safety Net
=======================================
Daemon thread monitoring all open positions every POLL_INTERVAL_S seconds.
If ANY position exceeds STOP_LOSS_PCT loss → force-close at market.

Permanent Fixes (A17):
- Market hours gate: completely silent outside NYSE hours — no close attempts, no alerts
- Alert dedup: 30-minute cooldown per symbol — prevents message floods
- Token read from env var (not hardcoded)
"""
import os
import json
import logging
import threading
import time
import urllib.request
import urllib.error
from datetime import datetime

log = logging.getLogger("position_watchdog")

ALPACA_KEY    = os.getenv("ALPACA_API_KEY",    "")  # FIX [F12]: no hardcoded fallback
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "")  # FIX [F12]: no hardcoded fallback
ALPACA_BASE   = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")

BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
AHMED_DM     = os.getenv("TELEGRAM_AHMED_ID",  "8573754783")
GROUP_ID     = os.getenv("TELEGRAM_GROUP_ID",  "-5130564161")

STOP_LOSS_PCT   = float(os.getenv("WATCHDOG_STOP_LOSS_PCT",  "8.0"))
CAPITAL_ROUTER_URL = os.getenv("CAPITAL_ROUTER_URL", "")
SYSTEM_ID          = os.getenv("SYSTEM_ID", "ATG_SWING")
_CR_PNL_COOLDOWN_S = 60
_last_cr_pnl_report: float = 0.0

POLL_INTERVAL_S = int(os.getenv("WATCHDOG_POLL_INTERVAL_S",  "60"))
RESTART_DELAY_S = int(os.getenv("WATCHDOG_RESTART_DELAY_S",  "10"))

_last_alert: dict = {}
_ALERT_COOLDOWN_S = 1800  # 30 minutes


def _is_market_open() -> bool:
    """True only during NYSE regular hours Mon-Fri 09:30-16:00 ET."""
    try:
        import zoneinfo
        et = zoneinfo.ZoneInfo("America/New_York")
        now_et = datetime.now(et)
    except Exception:
        try:
            import pytz
            now_et = datetime.now(pytz.timezone("America/New_York"))
        except Exception:
            return False
    if now_et.weekday() >= 5:
        return False
    market_open  = now_et.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now_et < market_close


def _alpaca_get(path: str):
    url = f"{ALPACA_BASE}{path}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("APCA-API-KEY-ID",     ALPACA_KEY)
    req.add_header("APCA-API-SECRET-KEY", ALPACA_SECRET)
    req.add_header("Content-Type",        "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except Exception as e:
        log.error("Alpaca GET %s error: %s", path, e)
        return None


def _alpaca_delete(path: str):
    url = f"{ALPACA_BASE}{path}"
    req = urllib.request.Request(url, method="DELETE")
    req.add_header("APCA-API-KEY-ID",     ALPACA_KEY)
    req.add_header("APCA-API-SECRET-KEY", ALPACA_SECRET)
    req.add_header("Content-Type",        "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except Exception as e:
        log.error("Alpaca DELETE %s error: %s", path, e)
        return None


def _telegram_send(message: str):
    if not BOT_TOKEN:
        log.debug("Telegram not configured (BOT_TOKEN not set)")
        return
    targets = [AHMED_DM, GROUP_ID] if GROUP_ID else [AHMED_DM]
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in targets:
        payload = json.dumps({"chat_id": chat_id, "text": message, "parse_mode": "Markdown"}).encode()
        try:
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                result = json.loads(r.read())
                if not result.get("ok"):
                    log.warning("Telegram send to %s failed: %s", chat_id, result)
        except Exception as e:
            log.warning("Telegram send error: %s", e)


def _report_pnl_to_cr(total_unrealized_pnl: float) -> None:
    """Step 54: report unrealized P&L to Capital Router every watchdog cycle (best-effort)."""
    global _last_cr_pnl_report
    now = time.time()
    if not CAPITAL_ROUTER_URL or now - _last_cr_pnl_report < _CR_PNL_COOLDOWN_S:
        return
    try:
        payload = json.dumps({"system": SYSTEM_ID, "pnl": round(total_unrealized_pnl, 2)}).encode()
        req = urllib.request.Request(
            f"{CAPITAL_ROUTER_URL.rstrip('/')}/report_pnl",
            data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=4):
            _last_cr_pnl_report = now
            log.debug("CR pnl synced: unrealized=$%.2f", total_unrealized_pnl)
    except Exception as e:
        log.debug("CR pnl sync non-critical error: %s", e)


def _check_and_enforce():
    """Check positions. Only attempts close during market hours."""
    if not _is_market_open():
        log.debug("Watchdog: market closed — skipping")
        return

    positions = _alpaca_get("/v2/positions")
    if not positions:
        return

    total_unrealized_pnl = 0.0  # Step 54
    for pos in positions:
        try:
            symbol      = pos.get("symbol", "UNKNOWN").upper()
            qty         = float(pos.get("qty", 0))
            side        = pos.get("side", "long")
            avg_cost    = float(pos.get("avg_entry_price", 0))
            cur_price   = float(pos.get("current_price", 0))
            unreal_pl   = float(pos.get("unrealized_pl", 0))
            unreal_plpc = float(pos.get("unrealized_plpc", 0)) * 100

            total_unrealized_pnl += unreal_pl  # Step 54

            if unreal_plpc > -STOP_LOSS_PCT:
                continue

            log.warning("WATCHDOG: %s loss %.2f%% — FORCE CLOSING", symbol, unreal_plpc)
            result = _alpaca_delete(f"/v2/positions/{symbol}")
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            if result is not None:
                alert = (f"\U0001f6d1 *WATCHDOG_FORCE_CLOSE*\n"
                         f"Symbol: `{symbol}` | Loss: `{unreal_plpc:.2f}%`\n"
                         f"Entry: ${avg_cost:.2f} | Now: ${cur_price:.2f} | P&L: ${unreal_pl:+.2f}\n"
                         f"Time: {ts}")
                log.warning("WATCHDOG_FORCE_CLOSE: %s loss=%.2f%%", symbol, unreal_plpc)
            else:
                alert = (f"\u274c *WATCHDOG_CLOSE_FAILED*\n"
                         f"Symbol: `{symbol}` | Loss: `{unreal_plpc:.2f}%`\n"
                         f"\u26a0\ufe0f Manual close required | Time: {ts}")
                log.error("WATCHDOG_CLOSE_FAILED: %s loss=%.2f%%", symbol, unreal_plpc)

            now_ts = time.time()
            if now_ts - _last_alert.get(symbol, 0) >= _ALERT_COOLDOWN_S:
                _telegram_send(alert)
                _last_alert[symbol] = now_ts
            else:
                log.info("Alert for %s suppressed (cooldown)", symbol)

        except Exception as e:
            log.error("Watchdog error on %s: %s", pos.get("symbol", "?"), e)

    _report_pnl_to_cr(total_unrealized_pnl)  # Step 54


def _watchdog_loop():
    log.info("Position Watchdog started | threshold=%.0f%% | poll=%ds",
             STOP_LOSS_PCT, POLL_INTERVAL_S)
    consecutive_errors = 0
    while True:
        try:
            _check_and_enforce()
            consecutive_errors = 0
            time.sleep(POLL_INTERVAL_S)
        except Exception as e:
            consecutive_errors += 1
            delay = min(RESTART_DELAY_S * (2 ** min(consecutive_errors - 1, 5)), 300)
            log.error("Watchdog error #%d: %s — retry in %ds", consecutive_errors, e, delay)
            time.sleep(delay)


def start_position_watchdog() -> threading.Thread:
    t = threading.Thread(target=_watchdog_loop, name="position-watchdog", daemon=True)
    t.start()
    log.info("Position Watchdog launched (id=%s)", t.ident)
    return t


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
    _check_and_enforce()
