"""
order_lifecycle.py — Order State Machine + Async Watcher Task
=============================================================
Tracks every submitted Alpaca order: PENDING → FILLED | CANCELED | EXPIRED.

FIX F6: Converted from threading.Thread + threading.Lock to asyncio coroutine.
        - Removes GIL contention and event-loop stall risk
        - register() called from async context; no lock needed (single event loop)
        - _loop() is an async coroutine started via asyncio.create_task()
        - Blocking HTTP calls run in executor to avoid stalling the event loop
        - asyncio.CancelledError handled gracefully for clean shutdown
        (INV-14, INV-18)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Callable, Dict, Optional

import requests

logger = logging.getLogger("order_lifecycle")

_TERMINAL_FILLED    = {"filled"}
_TERMINAL_VOID      = {"canceled", "cancelled", "expired", "rejected", "replaced"}
_PENDING_STATES     = {"new", "partially_filled", "pending_new",
                       "pending_cancel", "pending_replace", "held", "accepted",
                       "accepted_for_bidding"}


class _PendingOrder:
    """Internal record for a tracked order."""
    __slots__ = (
        "alpaca_order_id", "db_trade_id", "symbol",
        "amount_usd", "cr_trade_id", "registered_at",
    )

    def __init__(
        self,
        alpaca_order_id: str,
        db_trade_id: int,
        symbol: str,
        amount_usd: float,
        cr_trade_id: str,
    ) -> None:
        self.alpaca_order_id = alpaca_order_id
        self.db_trade_id     = db_trade_id
        self.symbol          = symbol
        self.amount_usd      = amount_usd
        self.cr_trade_id     = cr_trade_id
        self.registered_at   = time.monotonic()


class OrderLifecycleWatcher:
    """
    Async order state machine. Tracks pending Alpaca orders to FILLED | VOID.

    FIX F6: All state is accessed from the asyncio event loop thread only.
             No threading.Lock needed. Blocking HTTP runs in executor.
    """

    def __init__(
        self,
        alpaca_base_url: str,
        alpaca_key: str,
        alpaca_secret: str,
        cr_client,
        db_void_fn: Callable[[int], None],
        db_confirm_fn: Callable[[int, float], None],
        poll_interval_s: int = 30,
        order_timeout_s: int = 3600,
    ) -> None:
        self._base     = alpaca_base_url.rstrip("/")
        self._key      = alpaca_key
        self._secret   = alpaca_secret
        self._cr       = cr_client
        self._void     = db_void_fn
        self._confirm  = db_confirm_fn
        self._interval = poll_interval_s
        self._timeout  = order_timeout_s

        # FIX F6: plain dict — safe because all accesses are from event loop thread
        self._pending: Dict[str, _PendingOrder] = {}
        self._task: Optional[asyncio.Task] = None

    # ── Public ────────────────────────────────────────────────────────────────

    def register(
        self,
        alpaca_order_id: str,
        db_trade_id: int,
        symbol: str,
        amount_usd: float,
        cr_trade_id: str,
    ) -> None:
        """
        Register a newly submitted order for lifecycle tracking.
        Must be called from the asyncio event loop thread (e.g. inside execute_trade).
        FIX F6: No lock needed — single-threaded event loop access.
        """
        rec = _PendingOrder(
            alpaca_order_id=alpaca_order_id,
            db_trade_id=db_trade_id,
            symbol=symbol,
            amount_usd=amount_usd,
            cr_trade_id=cr_trade_id,
        )
        self._pending[alpaca_order_id] = rec
        logger.info(
            "[OLW] Registered order %s — symbol=%s db_id=%d",
            alpaca_order_id[:8], symbol, db_trade_id,
        )

    def start(self) -> None:
        """
        Schedule the async polling coroutine via asyncio.create_task (idempotent).
        FIX F6: Replaces threading.Thread.start(). Must be called inside a running event loop.
        """
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.error("[OLW] No event loop — watcher not started")
            return
        if self._task and not self._task.done():
            return
        self._task = loop.create_task(self._loop())
        logger.info("[OLW] Async watcher task started (poll=%ds, timeout=%ds)",
                    self._interval, self._timeout)

    def pending_count(self) -> int:
        return len(self._pending)

    # ── Internal ──────────────────────────────────────────────────────────────

    async def _loop(self) -> None:
        """
        Main async polling loop. FIX F6: asyncio.sleep replaces time.sleep.
        Blocking HTTP runs in executor to avoid stalling the event loop.
        """
        consecutive_errors = 0
        while True:
            try:
                order_ids = list(self._pending.keys())
                for oid in order_ids:
                    await self._check_order(oid)
                consecutive_errors = 0
                await asyncio.sleep(self._interval)  # FIX F6: non-blocking sleep

            except asyncio.CancelledError:
                logger.info("[OLW] Watcher task cancelled — shutting down")
                break
            except Exception as exc:
                consecutive_errors += 1
                backoff = min(self._interval * (2 ** min(consecutive_errors - 1, 4)), 300)
                logger.error("[OLW] Loop error #%d: %s — retry in %ds",
                             consecutive_errors, exc, backoff)
                await asyncio.sleep(backoff)  # FIX F6: non-blocking sleep

    async def _check_order(self, alpaca_order_id: str) -> None:
        """Poll one order and handle state transitions."""
        rec = self._pending.get(alpaca_order_id)
        if rec is None:
            return

        age_s = time.monotonic() - rec.registered_at
        if age_s > self._timeout:
            logger.warning("[OLW] Order %s timed out after %.0fs — voiding",
                           alpaca_order_id[:8], age_s)
            self._handle_void(rec, "timeout")
            return

        # FIX F6: run blocking HTTP call in executor, do not stall event loop
        loop   = asyncio.get_event_loop()
        status, fill_price = await loop.run_in_executor(
            None, self._fetch_order_status, alpaca_order_id
        )
        if status is None:
            return

        if status in _TERMINAL_FILLED:
            self._handle_filled(rec, fill_price or 0.0)
        elif status in _TERMINAL_VOID:
            self._handle_void(rec, status)
        elif status in _PENDING_STATES:
            logger.debug("[OLW] Order %s still pending (status=%s)", alpaca_order_id[:8], status)
        else:
            logger.warning("[OLW] Unknown order status \"%s\" for %s",
                           status, alpaca_order_id[:8])

    def _fetch_order_status(self, alpaca_order_id: str):
        """Blocking HTTP fetch — called via run_in_executor."""
        try:
            resp = requests.get(
                f"{self._base}/v2/orders/{alpaca_order_id}",
                headers={
                    "APCA-API-KEY-ID":     self._key,
                    "APCA-API-SECRET-KEY": self._secret,
                },
                timeout=10,
            )
            if resp.status_code == 200:
                data       = resp.json()
                status     = data.get("status", "")
                fill_price = None
                fp_raw     = data.get("filled_avg_price")
                if fp_raw:
                    try:
                        fill_price = float(fp_raw)
                    except (TypeError, ValueError):
                        pass
                return status, fill_price
            elif resp.status_code == 404:
                return "canceled", None
            else:
                logger.warning("[OLW] Alpaca GET order %s → HTTP %d",
                               alpaca_order_id[:8], resp.status_code)
                return None, None
        except Exception as exc:
            logger.warning("[OLW] Fetch order %s error: %s", alpaca_order_id[:8], exc)
            return None, None

    def _handle_filled(self, rec: _PendingOrder, fill_price: float) -> None:
        try:
            self._confirm(rec.db_trade_id, fill_price)
            logger.info("[OLW] FILLED: %s fill_price=%.4f db_id=%d → OPEN",
                        rec.symbol, fill_price, rec.db_trade_id)
        except Exception as exc:
            logger.error("[OLW] db_confirm_fn failed for db_id=%d: %s", rec.db_trade_id, exc)
        finally:
            self._pending.pop(rec.alpaca_order_id, None)

    def _handle_void(self, rec: _PendingOrder, reason: str) -> None:
        try:
            self._void(rec.db_trade_id)
            logger.info("[OLW] VOID: %s reason=%s db_id=%d → releasing CR $%.0f",
                        rec.symbol, reason, rec.db_trade_id, rec.amount_usd)
        except Exception as exc:
            logger.error("[OLW] db_void_fn failed for db_id=%d: %s", rec.db_trade_id, exc)

        if rec.cr_trade_id and self._cr:
            try:
                self._cr.release(rec.symbol, rec.amount_usd, rec.cr_trade_id, pnl=0.0)
            except Exception as exc:
                logger.warning("[OLW] CR release failed for void %s: %s", rec.symbol, exc)

        self._pending.pop(rec.alpaca_order_id, None)
