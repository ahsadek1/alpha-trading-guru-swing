"""
startup_reconciler.py — 3-Way Startup Reconciliation
=====================================================
Runs once on every service boot. Aligns DB ↔ Capital Router ↔ Alpaca.

Problem this solves:
  After a restart or redeploy, the three systems can diverge:
  - DB may have OPEN/PENDING trades with no real Alpaca position (ghost)
  - CR may have allocations for trades that no longer exist (stale)
  - Alpaca may have positions the DB doesn't know about (orphan)

  Without reconciliation, ghost DB trades block new entries, stale CR
  allocations reduce available capital, and orphaned positions are never
  monitored for stop-loss or exit conditions.

Reconciliation rules:
  ┌─────────────┬──────────────┬────────────────────────────────────────────┐
  │ DB (OPEN)   │ Alpaca pos   │ Action                                     │
  ├─────────────┼──────────────┼────────────────────────────────────────────┤
  │ YES         │ YES          │ OK — alive, no action needed               │
  │ YES         │ NO           │ GHOST → mark DB VOID, release CR alloc     │
  │ NO          │ YES          │ ORPHAN → alert Ahmed, log for manual review │
  └─────────────┴──────────────┴────────────────────────────────────────────┘

Also handles PENDING trades (order submitted but not confirmed filled):
  - Check Alpaca order status
  - Filled   → confirm OPEN
  - Not filled → cancel Alpaca order + mark VOID + release CR

Usage:
    from startup_reconciler import StartupReconciler

    reconciler = StartupReconciler(
        alpaca_base_url=settings.ALPACA_BASE_URL,
        alpaca_key=settings.ALPACA_API_KEY,
        alpaca_secret=settings.ALPACA_SECRET_KEY,
        cr_client=cr_client,
        db_get_open_fn=get_open_trades,       # returns list of trade dicts
        db_get_pending_fn=get_pending_trades,  # returns list of pending trade dicts
        db_void_fn=void_trade,                 # fn(db_id, reason) → marks VOID
        db_confirm_fn=confirm_trade_open,      # fn(db_id, fill_price) → marks OPEN
        system_id="ATG_SWING",
        notify_fn=send_telegram_alert,         # optional: fn(str) for alerts
    )
    report = reconciler.run()
    # report: {"ghosts": [...], "orphans": [...], "pending_confirmed": [...], "pending_voided": [...]}
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Dict, List, Optional

import requests

logger = logging.getLogger("startup_reconciler")


class StartupReconciler:
    """
    Runs once at startup to align DB, Capital Router, and Alpaca.
    Call reconciler.run() before starting the trading loop.
    """

    def __init__(
        self,
        alpaca_base_url: str,
        alpaca_key: str,
        alpaca_secret: str,
        cr_client,                                           # CapitalRouterClient
        db_get_open_fn: Callable[[], List[Dict[str, Any]]],
        db_get_pending_fn: Callable[[], List[Dict[str, Any]]],
        db_void_fn: Callable[[int, str], None],              # fn(db_id, reason)
        db_confirm_fn: Callable[[int, float], None],         # fn(db_id, fill_price)
        system_id: str,
        notify_fn: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._base          = alpaca_base_url.rstrip("/")
        self._key           = alpaca_key
        self._secret        = alpaca_secret
        self._cr            = cr_client
        self._get_open      = db_get_open_fn
        self._get_pending   = db_get_pending_fn
        self._void          = db_void_fn
        self._confirm       = db_confirm_fn
        self._system_id     = system_id
        self._notify        = notify_fn

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self) -> Dict[str, List]:
        """
        Execute full 3-way reconciliation.

        Returns
        -------
        dict with keys:
            ghosts            — DB OPEN trades with no Alpaca position (voided)
            orphans           — Alpaca positions with no DB trade (alerted)
            pending_confirmed — PENDING orders that filled (confirmed OPEN)
            pending_voided    — PENDING orders that didn't fill (voided)
        """
        logger.info("[RECONCILE] Starting startup reconciliation for %s", self._system_id)

        report: Dict[str, List] = {
            "ghosts": [],
            "orphans": [],
            "pending_confirmed": [],
            "pending_voided": [],
        }

        # Step 1: Get real Alpaca positions
        alpaca_positions = self._get_alpaca_positions()
        alpaca_symbols   = {p["symbol"].upper() for p in alpaca_positions}
        logger.info("[RECONCILE] Alpaca real positions: %s", alpaca_symbols or "(none)")

        # Step 2: Reconcile PENDING trades (order submitted, not yet confirmed filled)
        pending_trades = self._get_pending()
        logger.info("[RECONCILE] DB PENDING trades: %d", len(pending_trades))
        for trade in pending_trades:
            result = self._reconcile_pending(trade)
            report[f"pending_{result}"].append(trade)

        # Step 3: Reconcile OPEN trades (should have Alpaca position)
        open_trades = self._get_open()
        logger.info("[RECONCILE] DB OPEN trades: %d", len(open_trades))
        for trade in open_trades:
            symbol = (trade.get("ticker") or trade.get("symbol") or "").upper()
            if not symbol:
                continue

            if symbol in alpaca_symbols:
                logger.debug("[RECONCILE] OK: %s in DB and in Alpaca", symbol)
            else:
                logger.warning(
                    "[RECONCILE] GHOST: DB has OPEN trade for %s but no Alpaca position — voiding",
                    symbol,
                )
                self._void_ghost(trade)
                report["ghosts"].append(trade)

        # Step 4: Find orphaned Alpaca positions (position exists but no DB trade)
        open_symbols = {
            (t.get("ticker") or t.get("symbol") or "").upper()
            for t in open_trades
        }
        for pos in alpaca_positions:
            sym = pos["symbol"].upper()
            if sym not in open_symbols:
                logger.warning(
                    "[RECONCILE] ORPHAN: Alpaca has position %s with no DB trade — alerting",
                    sym,
                )
                report["orphans"].append(pos)
                self._alert_orphan(pos)

        # Summary
        self._log_summary(report)
        return report

    # ── Internal ──────────────────────────────────────────────────────────────

    def _get_alpaca_positions(self) -> List[Dict[str, Any]]:
        """Fetch all open positions from Alpaca."""
        try:
            resp = requests.get(
                f"{self._base}/v2/positions",
                headers={
                    "APCA-API-KEY-ID":     self._key,
                    "APCA-API-SECRET-KEY": self._secret,
                },
                timeout=15,
            )
            if resp.status_code == 200:
                return resp.json() or []
            logger.warning("[RECONCILE] Alpaca /positions HTTP %d", resp.status_code)
            return []
        except Exception as exc:
            logger.error("[RECONCILE] Failed to fetch Alpaca positions: %s", exc)
            return []

    def _get_alpaca_order_status(self, order_id: str):
        """Return (status, fill_price) for an Alpaca order."""
        try:
            resp = requests.get(
                f"{self._base}/v2/orders/{order_id}",
                headers={
                    "APCA-API-KEY-ID":     self._key,
                    "APCA-API-SECRET-KEY": self._secret,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data       = resp.json()
                status     = data.get("status", "")
                fill_raw   = data.get("filled_avg_price")
                fill_price = float(fill_raw) if fill_raw else None
                return status, fill_price
            elif resp.status_code == 404:
                return "not_found", None
            else:
                return None, None
        except Exception as exc:
            logger.warning("[RECONCILE] Order status fetch error: %s", exc)
            return None, None

    def _cancel_alpaca_order(self, order_id: str) -> None:
        """Cancel a pending Alpaca order (best-effort)."""
        try:
            resp = requests.delete(
                f"{self._base}/v2/orders/{order_id}",
                headers={
                    "APCA-API-KEY-ID":     self._key,
                    "APCA-API-SECRET-KEY": self._secret,
                },
                timeout=10,
            )
            if resp.status_code in (200, 204):
                logger.info("[RECONCILE] Canceled pending order %s", order_id[:8])
            else:
                logger.warning(
                    "[RECONCILE] Cancel order %s → HTTP %d",
                    order_id[:8], resp.status_code,
                )
        except Exception as exc:
            logger.warning("[RECONCILE] Cancel order error: %s", exc)

    def _reconcile_pending(self, trade: Dict[str, Any]) -> str:
        """
        Reconcile a single PENDING trade.
        Returns "confirmed" or "voided".
        """
        db_id    = trade["id"]
        symbol   = (trade.get("ticker") or trade.get("symbol") or "").upper()
        order_id = trade.get("alpaca_order_id", "")

        if not order_id:
            logger.warning("[RECONCILE] PENDING trade db_id=%d has no order_id — voiding", db_id)
            self._void_ghost(trade, reason="no_order_id")
            return "voided"

        status, fill_price = self._get_alpaca_order_status(order_id)

        if status is None:
            logger.warning(
                "[RECONCILE] Cannot determine status for order %s (db_id=%d) — skipping",
                order_id[:8], db_id,
            )
            return "voided"  # safe default: treat as voided if unknown

        filled_states  = {"filled", "partially_filled"}
        canceled_states = {"canceled", "cancelled", "expired", "rejected", "replaced", "not_found"}

        if status in filled_states:
            fp = fill_price or float(trade.get("entry_price", 0))
            logger.info(
                "[RECONCILE] PENDING→OPEN: %s order %s filled @ %.4f (db_id=%d)",
                symbol, order_id[:8], fp, db_id,
            )
            try:
                self._confirm(db_id, fp)
            except Exception as exc:
                logger.error("[RECONCILE] db_confirm_fn failed: %s", exc)
            return "confirmed"

        else:
            # Cancel order if still open, then void
            if status not in canceled_states:
                self._cancel_alpaca_order(order_id)

            logger.info(
                "[RECONCILE] PENDING→VOID: %s order %s status=%s (db_id=%d)",
                symbol, order_id[:8], status, db_id,
            )
            self._void_ghost(trade, reason=f"order_{status}")
            return "voided"

    def _void_ghost(self, trade: Dict[str, Any], reason: str = "no_alpaca_position") -> None:
        """Mark a DB trade as VOID and release its CR allocation."""
        db_id    = trade["id"]
        symbol   = (trade.get("ticker") or trade.get("symbol") or "").upper()
        alloc_id = trade.get("capital_alloc_id") or trade.get("capital_router_ref") or ""
        amount   = float(trade.get("allocated_amount") or 0.0)

        try:
            self._void(db_id, reason)
            logger.info("[RECONCILE] Voided DB trade db_id=%d (%s) reason=%s", db_id, symbol, reason)
        except Exception as exc:
            logger.error("[RECONCILE] db_void_fn failed for db_id=%d: %s", db_id, exc)

        # Release CR allocation
        if alloc_id and amount > 0 and self._cr:
            released = self._cr.release(symbol, amount, alloc_id, pnl=0.0)
            if released:
                logger.info("[RECONCILE] CR released $%.0f for ghost %s", amount, symbol)
            else:
                logger.warning("[RECONCILE] CR release failed for ghost %s (alloc_id=%s)",
                               symbol, alloc_id[:8] if alloc_id else "none")

    def _alert_orphan(self, pos: Dict[str, Any]) -> None:
        """Alert Ahmed about an Alpaca position with no DB trade."""
        symbol = pos.get("symbol", "?")
        qty    = pos.get("qty", "?")
        side   = pos.get("side", "?")
        pnl    = pos.get("unrealized_pl", "?")

        msg = (
            f"⚠️ ORPHAN POSITION DETECTED\n"
            f"Symbol: {symbol}\n"
            f"Qty: {qty} ({side})\n"
            f"Unrealized P&L: ${pnl}\n"
            f"System: {self._system_id}\n"
            f"Action needed: position exists in Alpaca with no DB record. "
            f"Manual review required."
        )
        logger.warning("[RECONCILE] ORPHAN: %s", msg)
        if self._notify:
            try:
                self._notify(msg)
            except Exception as exc:
                logger.warning("[RECONCILE] notify failed: %s", exc)

    def _log_summary(self, report: Dict[str, List]) -> None:
        ghosts    = len(report["ghosts"])
        orphans   = len(report["orphans"])
        confirmed = len(report["pending_confirmed"])
        voided    = len(report["pending_voided"])
        total     = ghosts + orphans + confirmed + voided

        if total == 0:
            logger.info("[RECONCILE] ✅ Clean — DB, CR, and Alpaca all aligned")
        else:
            logger.warning(
                "[RECONCILE] Summary: ghosts=%d orphans=%d pending_confirmed=%d pending_voided=%d",
                ghosts, orphans, confirmed, voided,
            )

        if self._notify and (ghosts > 0 or orphans > 0):
            msg = (
                f"🔄 STARTUP RECONCILIATION — {self._system_id}\n"
                f"Ghosts voided:       {ghosts}\n"
                f"Orphans detected:    {orphans}\n"
                f"Pending confirmed:   {confirmed}\n"
                f"Pending voided:      {voided}\n"
                f"{'✅ All clear' if total == 0 else '⚠️ Review required'}"
            )
            try:
                self._notify(msg)
            except Exception:
                pass


# ── Module-level convenience wrapper (used by main.py boot sequence) ──────────

def run_startup_reconciliation() -> None:
    """
    Module-level wrapper used by main.py lifespan boot.
    Instantiates StartupReconciler with ATG_SWING settings and runs it.
    Logs summary; non-fatal (caller wraps in try/except).
    """
    import logging
    _log = logging.getLogger("atg.startup_reconciler")

    try:
        from config.settings import ALPACA_BASE_URL, ALPACA_API_KEY, ALPACA_SECRET_KEY, CAPITAL_ROUTER_URL
        from src.database import get_open_positions, void_position, confirm_position_open
        _get_pending = lambda: []   # ATG Swing has no pending-state tracking yet
    except ImportError as e:
        _log.warning("startup_reconciler: import error — skipping (%s)", e)
        return

    reconciler = StartupReconciler(
        alpaca_base_url=ALPACA_BASE_URL,
        alpaca_key=ALPACA_API_KEY,
        alpaca_secret=ALPACA_SECRET_KEY,
        cr_client=None,
        db_get_open_fn=get_open_positions,
        db_get_pending_fn=_get_pending,
        db_void_fn=lambda db_id, reason: (
            void_position(db_id, reason),
            _log.info("RECONCILE void db_id=%d reason=%s", db_id, reason),
        ),
        db_confirm_fn=lambda db_id, price: confirm_position_open(db_id, price),
        system_id="ATG_SWING",
        notify_fn=None,
    )
    rpt = reconciler.run()
    _log.info(
        "Startup reconciliation complete — ghosts=%d orphans=%d confirmed=%d voided=%d",
        len(rpt["ghosts"]), len(rpt["orphans"]),
        len(rpt["pending_confirmed"]), len(rpt["pending_voided"]),
    )
