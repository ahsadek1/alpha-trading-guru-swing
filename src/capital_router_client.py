"""
Capital Router Client — communicates with the central capital allocation service.
"""
import os
import logging
import requests

logger = logging.getLogger(__name__)

CAPITAL_ROUTER_URL = os.getenv("CAPITAL_ROUTER_URL", "https://capital-router-production.up.railway.app")
SYSTEM_ID = os.getenv("CAPITAL_ROUTER_SYSTEM_ID", "ATG_SWING")


def allocate_capital(symbol: str, amount: float, trade_id: str) -> dict:
    """Request capital allocation from Capital Router."""
    try:
        resp = requests.post(
            f"{CAPITAL_ROUTER_URL}/allocate",
            json={"system": SYSTEM_ID, "symbol": symbol, "amount": amount, "trade_id": trade_id},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"Capital Router allocate failed: {resp.status_code} — {resp.text}")
        return {"approved": False, "reason": f"http_{resp.status_code}"}
    except Exception as e:
        logger.error(f"Capital Router allocate exception: {e}")
        return {"approved": False, "reason": "connection_error"}


def release_capital(symbol: str, amount: float, trade_id: str, pnl: float = 0.0) -> dict:
    """Release capital allocation back to Capital Router."""
    try:
        resp = requests.post(
            f"{CAPITAL_ROUTER_URL}/release",
            json={"system": SYSTEM_ID, "symbol": symbol, "amount": amount, "trade_id": trade_id, "pnl": pnl},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        logger.error(f"Capital Router release failed: {resp.status_code} — {resp.text}")
        return {"released": False, "reason": f"http_{resp.status_code}"}
    except Exception as e:
        logger.error(f"Capital Router release exception: {e}")
        return {"released": False, "reason": "connection_error"}

def report_equity(equity: float) -> dict:
    """Step 15: Report current Alpaca equity to CR for global circuit breaker math."""
    try:
        resp = requests.post(
            f"{CAPITAL_ROUTER_URL}/equity",
            json={"system": SYSTEM_ID, "equity": equity, "pnl": 0.0},
            timeout=10
        )
        if resp.status_code == 200:
            return resp.json()
        logger.warning(f"Capital Router equity report failed: {resp.status_code}")
        return {"ok": False}
    except Exception as e:
        logger.warning(f"Capital Router equity report exception: {e}")
        return {"ok": False}
