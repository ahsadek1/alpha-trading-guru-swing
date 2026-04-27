"""
ATG Trade Executor v3.0 — Dynamic exit management with Capital Router integration.

Exit priority (in order):
  1. Market deterioration (SPY breaks 50MA) → exit all longs
  2. Hard stop hit
  3. Time stop (no progress in TIME_STOP_DAYS)
  4. Staged exit — sell first 50% at target1
  5. Full exit of remaining shares at target2
  6. Max hold time (42 calendar days)

Position management:
  - Breakeven stop: move to entry after 1x ATR gain
  - Trailing stop: activate at 8% gain, trail 2x ATR below peak
  - Add-on: place additional 50% position after 8% gain × 10 days
  - Capital Router: request on open, release on close
"""
import logging
import requests
from datetime import datetime
from typing import Optional, List, Tuple
import numpy as np
import pytz

from config.settings import (
    ALPACA_API_KEY,
    ALPACA_SECRET_KEY,
    ALPACA_BASE_URL,
    PAPER_MODE,
    MAX_POSITIONS,
    MAX_SECTOR_PCT,
    INITIAL_CAPITAL,
    TIME_STOP_DAYS,
    BREAKEVEN_TRIGGER_ATR,
    TRAILING_ACTIVATION_PCT,
    TRAILING_ATR_MULTIPLE,
    STAGED_EXIT_ATR,
    STAGED_EXIT_SIZE,
    ADDON_MIN_GAIN_PCT,
    ADDON_SIZE_PCT,
    MAX_CORRELATED_POSITIONS,
    CORRELATED_GROUPS,
    CONVICTION_SIZING,
)
from src.database import (
    get_open_positions,
    mark_position_closing,
    record_position_open,
    record_position_close,
    record_bandit_outcome,
    update_position_stop,
    update_high_water_mark,
    mark_staged_exit_done,
    mark_addon_done,
)
from src.market_regime import is_spy_deteriorating
from src.capital_router import request_allocation, release_allocation

log = logging.getLogger(__name__)
ET  = pytz.timezone("America/New_York")

# ── Alpaca headers ────────────────────────────────────────────────────────────
_HEADERS = {
    "APCA-API-KEY-ID":     ALPACA_API_KEY,
    "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
    "Content-Type":        "application/json",
}


def _alpaca_get(path: str) -> dict:
    """
    HTTP GET against the Alpaca Paper API.

    Args:
        path: API path starting with /.

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.HTTPError on non-2xx response.
    """
    import time as _t
    last_exc = None
    for attempt in range(3):
        try:
            r = requests.get(f"{ALPACA_BASE_URL}{path}", headers=_HEADERS, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            if attempt < 2: _t.sleep(2**attempt)
    raise last_exc


def _alpaca_post(path: str, body: dict) -> dict:
    """
    HTTP POST to the Alpaca Paper API.

    Args:
        path : API path starting with /.
        body : JSON request body.

    Returns:
        Parsed JSON response dict.

    Raises:
        requests.HTTPError on non-2xx response.
    """
    import time as _t
    last_exc = None
    for attempt in range(3):
        try:
            r = requests.post(f"{ALPACA_BASE_URL}{path}", headers=_HEADERS, json=body, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            last_exc = e
            if attempt < 2: _t.sleep(2**attempt)
    raise last_exc


def get_current_price(symbol: str) -> Optional[float]:
    """
    Fetch the current market price for a symbol from Alpaca.

    Tries live position first, then latest trade.

    Args:
        symbol: ticker symbol.

    Returns:
        Current price float, or None if unavailable.
    """
    try:
        pos = _alpaca_get(f"/v2/positions/{symbol}")
        return float(pos["current_price"])
    except requests.RequestException:
        try:
            quote = _alpaca_get(f"/v2/latest/trades/{symbol}")
            return float(quote["trade"]["p"])
        except requests.RequestException as e:
            log.debug("Could not fetch price for %s: %s", symbol, e)
            return None


def _get_correlation_group(symbol: str) -> Optional[str]:
    """
    Return the correlation group key for a symbol, or None if not grouped.

    Args:
        symbol: ticker symbol.

    Returns:
        Group name string, or None.
    """
    for group, members in CORRELATED_GROUPS.items():
        if symbol in members:
            return group
    return None


def conviction_size_multiplier(score: int) -> float:
    """
    Return position size multiplier based on setup conviction score.

    Args:
        score: setup scanner score in [0, 100].

    Returns:
        Size multiplier float (0.80 – 1.50).
    """
    for threshold in sorted(CONVICTION_SIZING.keys(), reverse=True):
        if score >= threshold:
            return CONVICTION_SIZING[threshold]
    return 0.80


def _get_alpaca_live_position_count() -> int:
    """
    FIX [F27]: Query Alpaca for live position count across ALL systems on this account.
    Returns 0 on any error (fail-open on position count — conservative: use DB count as floor).
    """
    try:
        resp = _req.get(
            f"{ALPACA_BASE}/v2/positions",
            headers=_HEADERS,
            timeout=5,
        )
        if resp.status_code == 200:
            positions = resp.json()
            count = len(positions) if isinstance(positions, list) else 0
            log.debug("Alpaca live position count (cross-system): %d", count)
            return count
    except Exception as e:
        log.debug("Alpaca position count check failed (using DB): %s", e)
    return 0



def can_open_position(symbol: Optional[str] = None, sector: Optional[str] = None) -> Tuple[bool, str]:
    """
    Validate position-count, sector-concentration, and correlation limits.

    Args:
        symbol : ticker being considered (for correlation check).
        sector : sector of the ticker (for concentration check).

    Returns:
        (allowed: bool, reason: str)
    """
    open_pos = get_open_positions()

    # FIX [F27]: Cross-system position check — query Alpaca directly
    # Both ATG Swing and Nexus Alpha share the same Alpaca paper account.
    # DB-only check misses positions opened by other systems.
    alpaca_position_count = _get_alpaca_live_position_count()
    cross_system_count = max(len(open_pos), alpaca_position_count)

    if cross_system_count >= MAX_POSITIONS:
        return False, (
            f"Position cap reached: DB={len(open_pos)} Alpaca={alpaca_position_count} "
            f"max={MAX_POSITIONS} (cross-system check)"
        )

    if sector:
        sector_count = sum(1 for p in open_pos if p.get("sector") == sector)
        sector_pct   = sector_count / max(MAX_POSITIONS, 1)
        if sector_pct >= MAX_SECTOR_PCT:
            return False, f"Sector limit: {sector} at {sector_pct:.0%} >= {MAX_SECTOR_PCT:.0%}"

    if symbol:
        group = _get_correlation_group(symbol)
        if group:
            open_symbols = [p["symbol"] for p in open_pos]
            corr_count   = sum(
                1 for s in open_symbols if _get_correlation_group(s) == group
            )
            if corr_count >= MAX_CORRELATED_POSITIONS:
                return False, f"Correlation limit: {corr_count} positions in '{group}'"

    return True, "ok"


def open_swing_position(
    scan_result: dict,
    arm_index: int,
    stop_multiplier: float,
    context_vector,
    phase: int,
) -> dict:
    """
    Place a day-limit buy order for a qualifying swing setup.

    Applies conviction-based position sizing on top of the scanner's base shares.
    Requests capital allocation from the Capital Router.

    Args:
        scan_result     : scanner output dict for the symbol.
        arm_index       : bandit arm index selected.
        stop_multiplier : stop ATR multiplier from the bandit.
        context_vector  : context array (list or np.ndarray).
        phase           : current learning phase.

    Returns:
        Dict with "status" key: "OPENED" | "REJECTED" | "ERROR".
    """
    if not PAPER_MODE:
        return {"status": "REJECTED", "reason": "live_mode_disabled"}

    symbol  = scan_result["symbol"]
    price   = scan_result["price"]
    stop    = scan_result["stop_loss"]
    target  = scan_result["target_price"]
    target2 = scan_result.get("target_stage2", price + 4.0 * scan_result.get("atr_weekly", 5.0))
    atr     = scan_result["atr_weekly"]
    sector  = scan_result.get("sector", "Unknown")
    score   = scan_result.get("score", 80)

    ok, reason = can_open_position(symbol=symbol, sector=sector)
    if not ok:
        return {"status": "REJECTED", "reason": reason}

    # Conviction-based sizing
    size_mult = conviction_size_multiplier(score)
    shares    = max(1, int(scan_result["shares"] * size_mult))

    # Capital Router HARD GATE: must approve before placing order (P0.5 + P1.5 fix)
    alloc_dollars = shares * price
    approved, approved_amount, cr_trade_id = request_allocation(symbol, alloc_dollars)
    if not approved:
        log.info("[CR] Blocked %s — capital not approved", symbol)
        return {"status": "REJECTED", "reason": "capital_router_denied"}
    if approved_amount < alloc_dollars:
        shares = max(1, int(approved_amount / max(price, 0.01)))
        alloc_dollars = shares * price

    try:
        order = _alpaca_post("/v2/orders", {
            "symbol":        symbol,
            "qty":           str(shares),
            "side":          "buy",
            "type":          "limit",
            "limit_price":   str(round(price * 1.001, 2)),
            "time_in_force": "day",
        })

        pos_id = record_position_open({
            "symbol":              symbol,
            "setup_type":          scan_result["setup_type"],
            "stop_multiplier":     stop_multiplier,
            "arm_index":           arm_index,
            "entry_date":          datetime.now(ET).date().isoformat(),
            "entry_price":         price,
            "shares":              shares,
            "stop_loss":           stop,
            "target_price":        target,
            "target_stage2":       target2,
            "atr_at_entry":        atr,
            "sector":              sector,
            "phase":               phase,
            "context_vector":      list(context_vector),
            "capital_router_ref":  cr_trade_id,      # Step 17: store trade_id for release
            "allocated_amount":    approved_amount,   # Step 17: store approved amount for release
        })

        log.info(
            "✅ OPEN | %s %s | shares=%d entry=%.2f stop=%.2f t1=%.2f t2=%.2f",
            symbol, scan_result["setup_type"], shares, price, stop, target, target2,
        )

        return {
            "status":     "OPENED",
            "pos_id":     pos_id,
            "symbol":     symbol,
            "setup_type": scan_result["setup_type"],
            "shares":     shares,
            "entry":      price,
            "stop":       stop,
            "target":     target,
            "target2":    target2,
            "order_id":   order.get("id"),
        }

    except requests.HTTPError as e:
        log.error("Alpaca order rejected for %s: %s", symbol, e)
        release_allocation(alloc_dollars, reason=f"{symbol} order rejected")
        return {"status": "ERROR", "reason": str(e)}


def _place_sell(symbol: str, shares: int, order_type: str = "market",
                limit_price: Optional[float] = None) -> dict:
    """
    Submit a sell order to Alpaca.

    Args:
        symbol     : ticker to sell.
        shares     : number of shares to sell.
        order_type : "market" or "limit".
        limit_price: required when order_type is "limit".

    Returns:
        Alpaca order response dict.
    """
    body: dict = {
        "symbol":        symbol,
        "qty":           str(shares),
        "side":          "sell",
        "type":          order_type,
        "time_in_force": "day",
    }
    if order_type == "limit" and limit_price is not None:
        body["limit_price"] = str(round(limit_price, 2))
    return _alpaca_post("/v2/orders", body)


def _wait_for_fill_price(order_id: str, fallback_price: float, timeout_s: int = 30) -> float:
    """
    FIX [F5]: Poll Alpaca for actual fill price after order submission.
    
    A limit/market sell order returns filled_avg_price=null immediately.
    This function polls until the order is filled or timeout expires.
    
    Args:
        order_id     : Alpaca order ID to poll.
        fallback_price: price to use if order_id empty or timeout reached.
        timeout_s    : maximum seconds to wait for fill confirmation.
    
    Returns:
        Actual filled_avg_price, or fallback_price on timeout/error.
    """
    import time
    if not order_id:
        log.warning("_wait_for_fill_price: no order_id — using fallback price %.2f", fallback_price)
        return fallback_price
    
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            resp = _req.get(
                f"{ALPACA_BASE}/v2/orders/{order_id}",
                headers=_HEADERS,
                timeout=5,
            )
            if resp.status_code == 200:
                order_data = resp.json()
                status     = order_data.get("status", "")
                filled_avg = order_data.get("filled_avg_price")
                if status == "filled" and filled_avg:
                    fill_price = float(filled_avg)
                    log.info("Fill confirmed: order=%s price=%.2f", order_id[:8], fill_price)
                    return fill_price
                elif status in ("canceled", "rejected", "expired"):
                    log.warning("Order %s %s — using fallback price", order_id[:8], status)
                    return fallback_price
            time.sleep(2)
        except Exception as e:
            log.debug("Fill poll error (retrying): %s", e)
            time.sleep(2)
    
    log.warning(
        "Fill confirmation timeout after %ds for order %s — using fallback %.2f",
        timeout_s, order_id[:8], fallback_price,
    )
    return fallback_price


def _close_position(
    pos: dict,
    current_price: float,
    exit_reason: str,
    shares_to_close: Optional[int] = None,
    bandit=None,
) -> dict:
    """
    Execute a full or partial position close, record outcome, update bandit.

    Args:
        pos             : open position dict from DB.
        current_price   : current market price used for P&L calculation.
        exit_reason     : string exit tag (STOP_HIT, TARGET_HIT, etc.).
        shares_to_close : override share count (uses shares_remaining if None).
        bandit          : AutonomousSwingBandit instance for reward update.

    Returns:
        Dict summarising the closed trade.
    """
    symbol      = pos["symbol"]
    entry_price = float(pos["entry_price"])
    shares      = shares_to_close or pos.get("shares_remaining") or pos["shares"]
    entry_date  = datetime.fromisoformat(pos["entry_date"])
    hold_days   = (datetime.now(ET).date() - entry_date.date()).days

    # FIX (2026-04-27): Atomic claim — mark CLOSING before the sell order fires.
    # Prevents the position monitor and watchdog from both closing the same position
    # (the 30s fill-poll window was wide enough for a duplicate sell to fire).
    # The first caller wins; the second sees status != OPEN and raises.
    if not mark_position_closing(pos["id"], exit_reason):
        log.warning(
            "_close_position: %s pos_id=%d already CLOSING/CLOSED — duplicate close blocked",
            symbol, pos["id"],
        )
        return {
            "status":      "SKIPPED",
            "symbol":      symbol,
            "exit_reason": exit_reason,
            "detail":      "duplicate_close_blocked",
        }

    try:
        order      = _place_sell(symbol, shares)
        # FIX [F5]: Poll for actual fill price instead of using pre-sell current_price
        # filled_avg_price is null immediately after order submission for limit orders
        order_id   = order.get("id") or order.get("order_id", "")
        exit_price = _wait_for_fill_price(order_id, current_price, timeout_s=30)
    except requests.HTTPError as sell_err:
        log.warning("Sell order failed for %s: %s — using current price for P&L", symbol, sell_err)
        exit_price = current_price

    pnl_pct     = (exit_price - entry_price) / entry_price * 100.0
    pnl_dollars = (exit_price - entry_price) * shares

    record_position_close(pos["id"], {
        "exit_date":   datetime.now(ET).date().isoformat(),
        "exit_price":  exit_price,
        "exit_reason": exit_reason,
        "pnl_pct":     round(pnl_pct, 4),
        "pnl_dollars": round(pnl_dollars, 2),
        "hold_days":   hold_days,
    })

    # Capital Router: release allocation (Step 17: use stored trade_id + allocated_amount)
    release_allocation(
        symbol,
        float(pos.get("allocated_amount") or exit_price * shares),
        pos.get("capital_router_ref", ""),
        pnl=pnl_dollars,
    )

    # Step 16: report updated equity to CR after every trade close
    try:
        from src.capital_router_client import report_equity as _report_eq
        import requests as _req
        _acc = _req.get(
            f"{__import__('os').getenv('ALPACA_BASE_URL','https://paper-api.alpaca.markets')}/v2/account",
            headers={"APCA-API-KEY-ID": __import__('os').getenv("ALPACA_API_KEY",""),
                     "APCA-API-SECRET-KEY": __import__('os').getenv("ALPACA_SECRET_KEY","")},
            timeout=5
        )
        if _acc.status_code == 200:
            _report_eq(float(_acc.json().get("portfolio_value", 0)))
    except Exception:
        pass

    # Bandit reward update — magnitude-aware, clipped to [-1, +1]
    if bandit is not None:
        ctx = np.array(pos.get("context_vector") or [0.5] * 32, dtype=float)
        if len(ctx) != 32:
            ctx = np.full(32, 0.5)
        raw_reward = pnl_pct / 100.0
        reward     = float(np.clip(raw_reward * 3.0, -1.0, 1.0))
        # FIX [F25]: Exit reason weighting — INV-2 reward signal purity
        from src.quad_intelligence import get_exit_reason_weight
        exit_weight = get_exit_reason_weight(exit_reason or "UNKNOWN")
        bandit.update(pos["arm_index"], ctx, reward, exit_reason_weight=exit_weight)
        record_bandit_outcome({
            "position_id":    pos["id"],
            "arm_index":      pos["arm_index"],
            "setup_type":     pos["setup_type"],
            "stop_multiplier": pos["stop_multiplier"],
            "context_vector": list(ctx),
            "reward":         round(reward, 4),
            "phase":          pos.get("phase", 1),
        })

    log.info(
        "🔴 CLOSE %s | %s | pnl=%.2f%% ($%.2f) | hold=%dd | reason=%s",
        symbol, pos["setup_type"], pnl_pct, pnl_dollars, hold_days, exit_reason,
    )

    return {
        "status":      "CLOSED",
        "symbol":      symbol,
        "setup_type":  pos["setup_type"],
        "exit_reason": exit_reason,
        "pnl_pct":     round(pnl_pct, 2),
        "pnl_dollars": round(pnl_dollars, 2),
        "hold_days":   hold_days,
        "pos":         pos,
    }


def monitor_positions(bandit) -> List[dict]:
    """
    v3.0 position monitor — runs every 15 minutes during market hours.

    Exit checks (in priority order):
      1. SPY deterioration (exit all longs)
      2. Hard stop hit
      3. Time stop (≥ TIME_STOP_DAYS with ≤ 1% gain)
      4. Staged exit — sell first 50% at target1
      5. Full exit at target2 (after staged exit)
      6. Max hold (42 calendar days)

    Stop management (non-exit):
      - Breakeven stop: once up 1x ATR
      - Trailing stop: once 8% gain, trail 2x ATR below peak

    Args:
        bandit: AutonomousSwingBandit for reward updates.

    Returns:
        List of event dicts (closed or partial close).
    """
    open_pos  = get_open_positions()
    events    = []
    spy_bad   = is_spy_deteriorating()

    for pos in open_pos:
        symbol      = pos["symbol"]
        entry_price = float(pos["entry_price"])
        atr         = float(pos.get("atr_at_entry") or 5.0)
        stop_loss   = float(pos["stop_loss"])
        target1     = float(pos.get("target_price")  or entry_price + STAGED_EXIT_ATR * atr)
        target2     = float(pos.get("target_stage2") or entry_price + 4.0 * atr)
        entry_date  = datetime.fromisoformat(pos["entry_date"])
        hold_days   = (datetime.now(ET).date() - entry_date.date()).days
        hwm         = float(pos.get("high_water_mark") or entry_price)
        breakeven   = bool(pos.get("breakeven_set"))
        staged_done = bool(pos.get("staged_exit_done"))
        shares_rem  = int(pos.get("shares_remaining") or pos["shares"])

        current = get_current_price(symbol)
        if current is None:
            continue

        gain_pct = (current - entry_price) / entry_price

        # Update high-water mark
        if current > hwm:
            hwm = current
            update_high_water_mark(pos["id"], hwm)

        exit_reason: Optional[str] = None
        partial = False

        # 1. Market deterioration
        if spy_bad:
            exit_reason = "SPY_DETERIORATION"

        # 2. Hard stop hit
        elif current <= stop_loss:
            exit_reason = "STOP_HIT"

        # 3. Time stop
        elif hold_days >= TIME_STOP_DAYS and gain_pct <= 0.01 and not breakeven:
            exit_reason = "TIME_STOP"

        # 4. Staged exit (first 50% at target1)
        elif not staged_done and current >= target1:
            half_shares = max(1, int(shares_rem * STAGED_EXIT_SIZE))
            try:
                _place_sell(symbol, half_shares)
                mark_staged_exit_done(pos["id"], shares_rem - half_shares)
                update_position_stop(pos["id"], entry_price, breakeven_set=True)
                log.info(
                    "✅ STAGED EXIT 50%% | %s | price=%.2f | shares=%d",
                    symbol, current, half_shares,
                )
                events.append({
                    "status":      "PARTIAL_CLOSE",
                    "symbol":      symbol,
                    "setup_type":  pos["setup_type"],
                    "exit_reason": "STAGED_EXIT_50PCT",
                    "shares_sold": half_shares,
                    "pnl_pct":     round(gain_pct * 100, 2),
                    "pos":         pos,
                })
                partial = True
            except requests.HTTPError as sell_err:
                log.error("Staged exit failed for %s: %s", symbol, sell_err)

        # 5. Full exit of remaining shares at target2
        elif staged_done and current >= target2:
            exit_reason = "TARGET_HIT"

        # 6. Max hold
        elif hold_days >= 42:
            exit_reason = "TIME_EXIT"

        # ── Stop management (no exit, just stop updates) ───────────────────────
        else:
            gain_abs = (current - entry_price)

            # Breakeven: move stop to entry once up 1x ATR
            if not breakeven and gain_abs >= BREAKEVEN_TRIGGER_ATR * atr:
                update_position_stop(pos["id"], entry_price, breakeven_set=True)
                log.info("📌 Breakeven stop set for %s at %.2f", symbol, entry_price)

            # Trailing stop: trail 2x ATR below peak once 8% gain activated
            elif breakeven and gain_pct >= TRAILING_ACTIVATION_PCT:
                trail_stop = round(hwm - TRAILING_ATR_MULTIPLE * atr, 2)
                if trail_stop > stop_loss:
                    update_position_stop(pos["id"], trail_stop)
                    log.info("📈 Trailing stop updated for %s: %.2f", symbol, trail_stop)

        if exit_reason and not partial:
            result = _close_position(
                pos, current, exit_reason,
                shares_to_close=shares_rem,
                bandit=bandit,
            )
            # Only append real closes; skip duplicate-blocked events
            if result.get("status") != "SKIPPED":
                events.append(result)

    return events


def check_addon_opportunities(bandit) -> List[dict]:
    """
    Add-on position logic: if a trade is up ≥ 8% after 10 days, place 50% more.

    Args:
        bandit: bandit instance (not used for update here — only on close).

    Returns:
        List of add-on summary dicts.
    """
    open_pos = get_open_positions()
    addons   = []

    for pos in open_pos:
        if pos.get("addon_done"):
            continue

        symbol      = pos["symbol"]
        entry_price = float(pos["entry_price"])
        entry_date  = datetime.fromisoformat(pos["entry_date"])
        hold_days   = (datetime.now(ET).date() - entry_date.date()).days
        shares      = int(pos.get("shares_remaining") or pos["shares"])

        current = get_current_price(symbol)
        if current is None:
            continue

        gain_pct = (current - entry_price) / entry_price

        if hold_days >= 10 and gain_pct >= ADDON_MIN_GAIN_PCT:
            addon_shares = max(1, int(shares * ADDON_SIZE_PCT))
            ok, reason   = can_open_position(sector=pos.get("sector"))
            if not ok:
                log.debug("Add-on blocked for %s: %s", symbol, reason)
                continue

            try:
                # CR gate BEFORE Alpaca order (P1.5 fix: was reversed)
                addon_alloc = addon_shares * current
                addon_ok, addon_approved, addon_tid = request_allocation(symbol, addon_alloc)
                if not addon_ok:
                    log.info("[CR] Add-on blocked for %s — capital denied", symbol)
                    continue
                addon_alloc = addon_approved  # use approved amount

                _alpaca_post("/v2/orders", {
                    "symbol":        symbol,
                    "qty":           str(addon_shares),
                    "side":          "buy",
                    "type":          "market",
                    "time_in_force": "day",
                })
                mark_addon_done(pos["id"])

                log.info(
                    "➕ ADD-ON | %s +%d shares @ %.2f (up %.1f%%)",
                    symbol, addon_shares, current, gain_pct * 100,
                )
                addons.append({
                    "symbol":        symbol,
                    "addon_shares":  addon_shares,
                    "current_price": current,
                    "gain_pct":      round(gain_pct * 100, 2),
                })

            except requests.HTTPError as add_err:
                log.error("Add-on order failed for %s: %s", symbol, add_err)

    return addons

