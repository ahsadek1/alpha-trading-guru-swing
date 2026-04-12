"""
ATG Capital Router Client — v5.0 (Schema-verified 2026-04-09)

VERIFIED from live CR OpenAPI spec:
  POST /allocate → {"system": str, "symbol": str, "amount": float, "trade_id": str}
  Response       → {"approved": bool, "allocated": float, "trade_id": str}
  POST /release  → {"system": str, "symbol": str, "amount": float, "trade_id": str, "pnl": float}

HARD GATE: CR denial or error blocks the trade (not advisory).
"""
import logging
import uuid
import requests
from typing import Tuple

from config.settings import CAPITAL_ROUTER_URL

log     = logging.getLogger(__name__)
SYSTEM  = "ATG_SWING"
_BASE   = CAPITAL_ROUTER_URL.rstrip("/") if CAPITAL_ROUTER_URL else ""
_TIMEOUT = 10


def request_allocation(symbol: str, amount: float,
                        vix: float = 20.0, spread_pct: float = 0.0,
                        open_symbols: list = None) -> Tuple[bool, float, str]:
    """
    Request capital allocation. HARD GATE — returns (False, 0.0, "") on denial/error.
    Caller MUST check bool before placing any order.
    Returns (approved, approved_amount, trade_id).
    Optional risk context: vix, spread_pct, open_symbols — enables CR RiskEngine scoring.
    """
    if not _BASE:
        log.warning("[CR] No URL configured — trade blocked")
        return False, 0.0, ""

    trade_id = str(uuid.uuid4())
    try:
        r = requests.post(f"{_BASE}/allocate", json={
            "system":      SYSTEM,
            "symbol":      symbol.upper(),
            "amount":      amount,
            "trade_id":    trade_id,
            "vix":         vix,
            "spread_pct":  spread_pct,
            "open_symbols": open_symbols or [],
        }, timeout=_TIMEOUT)

        if r.status_code == 200:
            data = r.json()
            approved = bool(data.get("approved", False))
            amt      = float(data.get("allocated", amount))
            tid      = data.get("trade_id", trade_id)
            if approved:
                log.info("[CR] APPROVED: %s $%.0f → $%.0f", symbol, amount, amt)
                return True, amt, tid
            log.info("[CR] DENIED: %s — %s", symbol, data.get("reason", "denied"))
            return False, 0.0, ""

        log.warning("[CR] HTTP %d for %s — trade blocked", r.status_code, symbol)
        return False, 0.0, ""
    except Exception as e:
        log.error("[CR] unreachable for %s: %s — trade blocked", symbol, e)
        return False, 0.0, ""


def release_allocation(symbol: str, amount: float, trade_id: str = "",
                       pnl: float = 0.0) -> bool:
    """Release allocated capital after position close."""
    if not _BASE or not trade_id:
        return True
    try:
        r = requests.post(f"{_BASE}/release", json={
            "system":   SYSTEM,
            "symbol":   symbol.upper(),
            "amount":   amount,
            "trade_id": trade_id,
            "pnl":      round(pnl, 4),
        }, timeout=_TIMEOUT)
        if r.status_code == 200:
            log.info("[CR] RELEASED: %s $%.0f pnl=$%.2f", symbol, amount, pnl)
            return True
        log.warning("[CR] /release HTTP %d for %s", r.status_code, symbol)
        return False
    except Exception as e:
        log.warning("[CR] /release error: %s", e)
        return False
