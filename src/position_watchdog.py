"""
Position Watchdog — NEXUS Step 3 (enhanced from Block 1 Component 3)
======================================================================
Daemon thread that monitors all open options positions every 60 seconds.

Two complementary stop-loss checks:

1. PRICE-BASED STOP TARGETS (primary, NEXUS Step 3):
   Reads stop_targets table (SQLite). Each entry records:
     - osi_symbol: the specific options contract
     - entry_price: credit received (per share)
     - stop_price: computed at entry as entry_price × (1 + stop_loss_pct/100)
       e.g. entry=1.00, stop_loss_pct=200 → stop_price=3.00 (close if option mark > 3×)
     - target_price: 25% of original = 75% profit capture
   If current mark >= stop_price → force-close + log STOP_HIT
   If current mark <= target_price → force-close + log TARGET_HIT

2. UNREALIZED P&L FALLBACK (legacy):
   If a position is NOT in stop_targets (equity positions, reconciliation gap):
   Uses Alpaca's unrealized_plpc. Threshold: WATCHDOG_STOP_LOSS_PCT (default 8%).

Thread is self-healing: restarts automatically after any crash.

Usage:
    from src.position_watchdog import start_position_watchdog
    t = start_position_watchdog()
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

# ── Config ────────────────────────────────────────────────────────────────────
ALPACA_KEY    = os.getenv("ALPACA_API_KEY",    "PKPGM3BRNYPGCF5Z56IAUZCZJL")
ALPACA_SECRET = os.getenv("ALPACA_SECRET_KEY", "5uVVmmB2dYnpA1SsTbkde8V2wixocBfAvGBsnrWSnJDs")
ALPACA_BASE   = os.getenv("ALPACA_BASE_URL",   "https://paper-api.alpaca.markets")

BOT_TOKEN    = os.getenv("TELEGRAM_BOT_TOKEN", "")
AHMED_DM     = os.getenv("TELEGRAM_AHMED_ID",  "8573754783")
GROUP_ID     = os.getenv("TELEGRAM_GROUP_ID",  "-5130564161")

STOP_LOSS_PCT   = float(os.getenv("WATCHDOG_STOP_LOSS_PCT",  "8.0"))   # 8% loss threshold
POLL_INTERVAL_S = int(os.getenv("WATCHDOG_POLL_INTERVAL_S",  "60"))    # check every 60s
RESTART_DELAY_S = int(os.getenv("WATCHDOG_RESTART_DELAY_S",  "10"))    # delay before restart
ALERT_COOLDOWN_S = int(os.getenv("WATCHDOG_ALERT_COOLDOWN_S", "600"))  # 10 min between same-symbol alerts

# ── Alert dedup: track last alert time per symbol ─────────────────────────────
_last_alert_ts: dict = {}  # symbol → float (epoch)


# ── Alpaca helpers ────────────────────────────────────────────────────────────

def _alpaca_get(path: str):
    """GET from Alpaca. Returns parsed JSON or None on error."""
    url = f"{ALPACA_BASE}{path}"
    req = urllib.request.Request(url, method="GET")
    req.add_header("APCA-API-KEY-ID",     ALPACA_KEY)
    req.add_header("APCA-API-SECRET-KEY", ALPACA_SECRET)
    req.add_header("Content-Type",        "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        log.error("Alpaca GET %s → HTTP %d: %s", path, e.code, e.read().decode()[:200])
        return None
    except Exception as e:
        log.error("Alpaca GET %s error: %s", path, e)
        return None


def _alpaca_delete(path: str):
    """DELETE on Alpaca. Returns parsed JSON or None on error."""
    url = f"{ALPACA_BASE}{path}"
    req = urllib.request.Request(url, method="DELETE")
    req.add_header("APCA-API-KEY-ID",     ALPACA_KEY)
    req.add_header("APCA-API-SECRET-KEY", ALPACA_SECRET)
    req.add_header("Content-Type",        "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            body = r.read()
            return json.loads(body) if body else {}
    except urllib.error.HTTPError as e:
        log.error("Alpaca DELETE %s → HTTP %d: %s", path, e.code, e.read().decode()[:200])
        return None
    except Exception as e:
        log.error("Alpaca DELETE %s error: %s", path, e)
        return None


# ── Telegram helper ───────────────────────────────────────────────────────────

def _telegram_send(message: str, target: str = None):
    """Send message to Ahmed DM (default) and group."""
    targets = [AHMED_DM, GROUP_ID] if target is None else [target]
    url     = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"
    for chat_id in targets:
        payload = json.dumps({
            "chat_id":    chat_id,
            "text":       message,
            "parse_mode": "Markdown",
        }).encode()
        try:
            req = urllib.request.Request(url, data=payload,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=10) as r:
                result = json.loads(r.read())
                if not result.get("ok"):
                    log.warning("Telegram send to %s failed: %s", chat_id, result)
        except Exception as e:
            log.warning("Telegram send to %s error: %s", chat_id, e)


# ── Core watchdog logic ───────────────────────────────────────────────────────



# ── Market hours gate ─────────────────────────────────────────────────────────
def _market_is_open() -> bool:
    """Returns True only during regular market hours Mon-Fri 09:30-16:00 ET."""
    try:
        import zoneinfo
        tz = zoneinfo.ZoneInfo("America/New_York")
        from datetime import datetime
        now = datetime.now(tz)
    except Exception:
        from datetime import datetime, timezone, timedelta
        now = datetime.now(timezone(timedelta(hours=-4)))
    if now.weekday() >= 5:
        return False
    market_open  = now.replace(hour=9,  minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0,  second=0, microsecond=0)
    return market_open <= now <= market_close

def _get_option_mark(osi_symbol: str) -> float | None:
    """
    Fetch current mark price for an options contract via Alpaca.
    Returns mark (float) or None on failure.
    """
    # Alpaca options positions endpoint
    positions = _alpaca_get("/v2/positions")
    if not positions:
        return None
    for p in positions:
        if p.get("symbol", "").upper() == osi_symbol.upper():
            try:
                return float(p.get("current_price", 0))
            except Exception:
                return None
    return None


def _alpaca_post(path: str, payload: dict):
    """POST to Alpaca. Returns parsed JSON or None on error."""
    url  = f"{ALPACA_BASE}{path}"
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(url, data=data, method="POST")
    req.add_header("APCA-API-KEY-ID",     ALPACA_KEY)
    req.add_header("APCA-API-SECRET-KEY", ALPACA_SECRET)
    req.add_header("Content-Type",        "application/json")
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        log.error("Alpaca POST %s → HTTP %d: %s", path, e.code, e.read().decode()[:200])
        return None
    except Exception as e:
        log.error("Alpaca POST %s error: %s", path, e)
        return None


def _get_underlying(osi_symbol: str) -> str:
    """Extract underlying ticker from OSI symbol. E.g. NVDA260529C00190000 → NVDA."""
    import re
    m = re.match(r"^([A-Z]+)\d{6}[CP]\d{8}$", osi_symbol.upper())
    return m.group(1) if m else osi_symbol[:4]


def _close_short_legs_for_underlying(underlying: str) -> list:
    """
    Close any SHORT option positions on the same underlying before closing the long leg.
    Prevents 'uncovered option contracts' rejection from Alpaca on spread positions.
    Returns list of OSI symbols successfully closed.
    """
    import urllib.parse
    positions = _alpaca_get("/v2/positions")
    if not positions:
        return []
    closed = []
    for pos in positions:
        sym  = pos.get("symbol", "").upper()
        side = pos.get("side", "").lower()
        qty  = float(pos.get("qty", 0))
        if _get_underlying(sym) != underlying.upper():
            continue
        if len(sym) < 10 or side != "short":
            continue
        encoded = urllib.parse.quote(sym, safe="")
        log.info("Spread-close: buy_to_close short leg %s before closing long", sym)
        result = _alpaca_delete(f"/v2/positions/{encoded}")
        if result is not None:
            closed.append(sym)
            continue
        order = _alpaca_post("/v2/orders", {
            "symbol": sym, "qty": str(int(abs(qty))), "side": "buy",
            "type": "market", "time_in_force": "day",
            "position_intent": "buy_to_close",
        })
        if order and order.get("id"):
            closed.append(sym)
        else:
            log.error("Failed to close spread short leg %s — manual intervention needed", sym)
    return closed


def _close_option_position(osi_symbol: str, underlying: str, position_side: str = "long") -> bool:
    """
    Close an options position by its OSI symbol.
    Strategy:
      0. If closing a LONG leg, first close any SHORT legs on the same underlying
         (spread short-leg-first protocol — prevents 'uncovered option contracts' rejection)
      1. Try DELETE /v2/positions/{osi_symbol} (standard close)
      2. If that fails, place explicit sell_to_close / buy_to_close market order
      3. Last resort: DELETE /v2/positions/{underlying}
    Returns True if any attempt succeeded.
    """
    import urllib.parse

    if position_side == "long":
        short_legs_closed = _close_short_legs_for_underlying(underlying)
        if short_legs_closed:
            log.info("Spread-close: closed %d short leg(s) for %s before closing long",
                     len(short_legs_closed), underlying)
            import time as _time
            _time.sleep(1)

    encoded = urllib.parse.quote(osi_symbol, safe="")

    # Attempt 1: standard DELETE
    result = _alpaca_delete(f"/v2/positions/{encoded}")
    if result is not None:
        return True

    # Attempt 2: explicit order with position_intent (handles Alpaca uncovered-call rejection)
    side   = "sell" if position_side == "long" else "buy"
    intent = "sell_to_close" if position_side == "long" else "buy_to_close"
    log.warning("DELETE failed for %s — trying explicit %s order", osi_symbol, intent)
    order = _alpaca_post("/v2/orders", {
        "symbol":          osi_symbol,
        "qty":             "1",
        "side":            side,
        "type":            "market",
        "time_in_force":   "day",
        "position_intent": intent,
    })
    if order and order.get("id"):
        log.info("Explicit close order placed for %s: order_id=%s", osi_symbol, order["id"][:8])
        return True

    # Attempt 3: close all positions on underlying (last resort)
    log.warning("Explicit order failed for %s — retrying via underlying %s", osi_symbol, underlying)
    result = _alpaca_delete(f"/v2/positions/{underlying}")
    return result is not None


def _check_price_based_stops() -> None:
    """
    NEXUS Step 3 primary check: compare current option mark vs stop_price / target_price
    in the stop_targets table.
    """
    try:
        from src.database import get_open_stop_targets, close_stop_target
    except ImportError:
        log.warning("stop_targets module not available — skipping price-based check")
        return

    targets = get_open_stop_targets()
    if not targets:
        return

    ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    for row in targets:
        osi   = row.get("osi_symbol", "")
        under = row.get("underlying", "")
        entry = float(row.get("entry_price", 0))
        stop  = float(row.get("stop_price", 9999))
        tgt   = float(row.get("target_price", 0))
        side  = row.get("position_side", "")
        row_id = row.get("id")

        if not osi:
            continue

        mark = _get_option_mark(osi)
        if mark is None:
            log.debug("Watchdog: could not fetch mark for %s — skipping", osi)
            continue

        close_reason = None
        if mark >= stop:
            close_reason = "STOP_HIT"
            log.warning(
                "STOP_HIT: %s mark=%.4f >= stop=%.4f (entry=%.4f loss=%.0f%%)",
                osi, mark, stop, entry, (mark/entry - 1)*100,
            )
        elif mark <= tgt and tgt > 0:
            close_reason = "TARGET_HIT"
            log.info(
                "TARGET_HIT: %s mark=%.4f <= target=%.4f (profit=%.0f%%)",
                osi, mark, tgt, (1 - mark/entry)*100,
            )

        if close_reason:
            success = _close_option_position(osi, under, position_side=side)
            try:
                close_stop_target(row_id, close_reason)
            except Exception as e:
                log.error("close_stop_target DB error: %s", e)

            pnl_pct = round((1 - mark / entry) * 100, 2) if entry else 0  # + = profit, - = loss
            icon = "✅" if close_reason == "TARGET_HIT" else "🛑"
            alert_msg = (
                f"{icon} *WATCHDOG_{close_reason}*\n"
                f"Contract:  `{osi}`\n"
                f"Side:      {side}\n"
                f"Entry:     ${entry:.4f}\n"
                f"Mark:      ${mark:.4f}\n"
                f"P&L:       `{pnl_pct:+.1f}%`\n"
                f"Action:    {'Market close submitted' if success else '⚠️ Close FAILED — manual required'}\n"
                f"Time:      {ts}"
            )
            # Dedup: only alert once per ALERT_COOLDOWN_S per symbol
            now = time.time()
            if now - _last_alert_ts.get(osi, 0) >= ALERT_COOLDOWN_S:
                _last_alert_ts[osi] = now
                _telegram_send(alert_msg)
            else:
                log.debug("Alert dedup: suppressing repeat alert for %s", osi)


def _check_and_enforce():
    """
    Single iteration: two-layer stop enforcement.
    Layer 1 (primary): price-based stop_targets from SQLite.
    Layer 2 (fallback): unrealized P&L% for any positions not in stop_targets.
    Called every POLL_INTERVAL_S seconds.
    """
    if not _market_is_open():
        log.debug("Watchdog: market closed — skipping force-close check")
        return

    # Layer 1: price-based targets (NEXUS Step 3)
    _check_price_based_stops()

    # Layer 2: fallback unrealized P&L% check for anything not covered above
    positions = _alpaca_get("/v2/positions")
    if positions is None:
        log.warning("Watchdog: could not fetch positions (Alpaca unavailable)")
        return

    if not positions:
        return

    # Build set of OSI symbols already monitored by stop_targets
    try:
        from src.database import get_open_stop_targets
        monitored_osi = {r.get("osi_symbol", "").upper() for r in get_open_stop_targets()}
    except Exception:
        monitored_osi = set()

    for pos in positions:
        try:
            symbol      = pos.get("symbol", "UNKNOWN").upper()
            qty         = float(pos.get("qty", 0))
            side        = pos.get("side", "long")
            avg_cost    = float(pos.get("avg_entry_price", 0))
            cur_price   = float(pos.get("current_price", 0))
            unreal_pl   = float(pos.get("unrealized_pl", 0))
            unreal_plpc = float(pos.get("unrealized_plpc", 0)) * 100

            # Skip if already covered by price-based stop_targets
            if symbol in monitored_osi:
                continue

            # Only trigger on losses exceeding threshold
            if unreal_plpc > -STOP_LOSS_PCT:
                continue

            # FIX (2026-04-27): Cross-check DB before firing a sell.
            # If monitor_positions already claimed the position (status=CLOSING/CLOSED),
            # skip — prevents duplicate sell that created a naked short on GOOGL.
            try:
                from src.database import get_connection as _get_conn
                _conn = _get_conn()
                _row = _conn.execute(
                    "SELECT status FROM swing_positions "
                    "WHERE symbol=? AND status NOT IN ('CLOSED','CLOSING','VOIDED') "
                    "LIMIT 1",
                    (symbol,),
                ).fetchone()
                _conn.close()
                if _row is None:
                    log.info(
                        "WATCHDOG FALLBACK: %s already CLOSED/CLOSING in DB — skipping sell",
                        symbol,
                    )
                    continue
            except Exception as _db_err:
                log.warning("WATCHDOG DB cross-check failed for %s: %s", symbol, _db_err)

            log.warning(
                "WATCHDOG FALLBACK: %s unrealized loss %.2f%% exceeds %.1f%% — FORCE CLOSING",
                symbol, unreal_plpc, STOP_LOSS_PCT,
            )

            # Use smart close: handles options (sell_to_close / buy_to_close fallback)
            underlying_sym = symbol[:4] if len(symbol) > 6 else symbol
            close_ok = _close_option_position(symbol, underlying_sym, position_side=side)
            ts = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

            if close_ok:
                result = True  # for alert path below
            else:
                result = None

            if result is not None:
                alert_msg = (
                    f"🛑 *WATCHDOG_FALLBACK_CLOSE*\n"
                    f"Symbol:      `{symbol}` (not in stop_targets)\n"
                    f"Side:        {side.upper()}\n"
                    f"Qty:         {qty}\n"
                    f"Entry Price: ${avg_cost:.2f}\n"
                    f"Curr Price:  ${cur_price:.2f}\n"
                    f"Loss:        `{unreal_plpc:.2f}%` (${unreal_pl:+.2f})\n"
                    f"Threshold:   {STOP_LOSS_PCT:.0f}%\n"
                    f"Action:      Market order submitted\n"
                    f"Time:        {ts}"
                )
                log.warning("WATCHDOG_FALLBACK_CLOSE: %s | loss=%.2f%%", symbol, unreal_plpc)
            else:
                alert_msg = (
                    f"❌ *WATCHDOG_CLOSE_FAILED*\n"
                    f"Symbol: `{symbol}` | Loss: `{unreal_plpc:.2f}%`\n"
                    f"⚠️ Manual intervention required!\n"
                    f"Time: {ts}"
                )
                log.error("WATCHDOG_CLOSE_FAILED: %s | loss=%.2f%%", symbol, unreal_plpc)

            # Dedup: only alert once per ALERT_COOLDOWN_S per symbol
            now = time.time()
            if now - _last_alert_ts.get(symbol, 0) >= ALERT_COOLDOWN_S:
                _last_alert_ts[symbol] = now
                _telegram_send(alert_msg)
            else:
                log.debug("Alert dedup: suppressing repeat alert for %s", symbol)

        except Exception as e:
            log.error("Watchdog error processing position %s: %s", pos.get("symbol", "?"), e)


# ── Thread with self-healing ──────────────────────────────────────────────────

def _watchdog_loop():
    """
    Main loop for the watchdog daemon thread.
    Catches all exceptions and restarts after RESTART_DELAY_S.
    Designed to survive scanner crashes and all other failures.
    """
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
            log.error("Watchdog loop error #%d: %s", consecutive_errors, e, exc_info=True)

            # Exponential backoff up to 5 minutes
            delay = min(RESTART_DELAY_S * (2 ** min(consecutive_errors - 1, 5)), 300)
            log.info("Watchdog restarting in %ds...", delay)
            time.sleep(delay)


def start_position_watchdog() -> threading.Thread:
    """
    Start the position watchdog as a daemon thread.
    Returns the thread handle (for reference; thread is already running).

    Usage:
        from src.position_watchdog import start_position_watchdog
        watchdog_thread = start_position_watchdog()
    """
    t = threading.Thread(
        target=_watchdog_loop,
        name="position-watchdog",
        daemon=True,  # dies when main process exits
    )
    t.start()
    log.info("Position Watchdog thread launched (daemon=True, id=%s)", t.ident)
    return t


# ── Standalone test ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    )
    log.info("Running position watchdog in standalone test mode (single check)...")
    _check_and_enforce()
    log.info("Single check complete. Start full watchdog loop? Press Ctrl+C to exit.")
    t = start_position_watchdog()
    try:
        t.join()
    except KeyboardInterrupt:
        log.info("Watchdog stopped.")
