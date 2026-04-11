"""
order_lifecycle.py — Order State Machine + Watcher Thread
==========================================================
Tracks every submitted Alpaca order from PENDING → FILLED | CANCELED | EXPIRED.

Problem this solves:
  Previously, trades were written to DB as OPEN the moment an order was
  submitted. If the service restarted before the order filled (or the order
  was canceled), the DB showed a ghost OPEN trade with no real Alpaca position.

New flow:
  1. Order submitted to Alpaca                → DB: status=PENDING
  2. OrderLifecycleWatcher polls every 30s    → checks Alpaca order status
  3a. Order filled                            → DB: status=OPEN (real position now exists)
  3b. Order canceled/expired/rejected         → DB: status=VOID, CR allocation released
  4. Position closed normally                 → DB: status=WIN | LOSS (handled by trade_executor)

Usage:
    from order_lifecycle import OrderLifecycleWatcher

    watcher = OrderLifecycleWatcher(
        alpaca_base_url=settings.ALPACA_BASE_URL,
        alpaca_key=settings.ALPACA_API_KEY,
        alpaca_secret=settings.ALPACA_SECRET_KEY,
        cr_client=cr_client,           # CapitalRouterClient instance
        db_void_fn=void_trade,         # fn(db_trade_id) → marks trade VOID in DB
        db_confirm_fn=confirm_trade,   # fn(db_trade_id, fill_price) → marks OPEN
        poll_interval_s=30,
    )
    watcher.register(alpaca_order_id, db_trade_id, symbol, amount_usd, cr_trade_id)
    watcher.start()   # starts daemon thread
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Dict, Optional

import requests

logger = logging.getLogger("order_lifecycle")

# Order states Alpaca can return
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
    Daemon thread that polls Alpaca for order status changes.

    Thread-safe: register() can be called from any thread at any time.
    """

    def __init__(
        self,
        alpaca_base_url: str,
        alpaca_key: str,
        alpaca_secret: str,
        cr_client,                              # CapitalRouterClient
        db_void_fn: Callable[[int], None],      # fn(db_trade_id) → void the trade
        db_confirm_fn: Callable[[int, float], None],  # fn(db_trade_id, fill_price) → confirm open
        poll_interval_s: int = 30,
        order_timeout_s: int = 3600,            # 1 hour — cancel tracking after this
    ) -> None:
        self._base     = alpaca_base_url.rstrip("/")
        self._key      = alpaca_key
        self._secret   = alpaca_secret
        self._cr       = cr_client
        self._void     = db_void_fn
        self._confirm  = db_confirm_fn
        self._interval = poll_interval_s
        self._timeout  = order_timeout_s

        self._pending: Dict[str, _PendingOrder] = {}   # alpaca_order_id → record
        self._lock = threading.Lock()

        self._thread: Optional[threading.Thread] = None

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

        Call this immediately after a successful Alpaca order submission,
        BEFORE writing anything to the DB as OPEN.

        Parameters
        ----------
        alpaca_order_id : str
            The order ID returned by Alpaca.
        db_trade_id : int
            The DB primary key of the trade record (written as PENDING).
        symbol : str
            Ticker symbol.
        amount_usd : float
            Capital allocated for this trade (needed to release CR on void).
        cr_trade_id : str
            The trade_id returned by CapitalRouterClient.allocate().
        """
        rec = _PendingOrder(
            alpaca_order_id=alpaca_order_id,
            db_trade_id=db_trade_id,
            symbol=symbol,
            amount_usd=amount_usd,
            cr_trade_id=cr_trade_id,
        )
        with self._lock:
            self._pending[alpaca_order_id] = rec

        logger.info(
            "[OLW] Registered order %s — symbol=%s db_id=%d",
            alpaca_order_id[:8], symbol, db_trade_id,
        )

    def start(self) -> None:
        """Start the watcher daemon thread (idempotent)."""
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(
            target=self._loop,
            name="order-lifecycle-watcher",
            daemon=True,
        )
        self._thread.start()
        logger.info("[OLW] Watcher thread started (poll=%ds, timeout=%ds)",
                    self._interval, self._timeout)

    def pending_count(self) -> int:
        """Return number of currently tracked pending orders."""
        with self._lock:
            return len(self._pending)

    # ── Internal ──────────────────────────────────────────────────────────────

    def _loop(self) -> None:
        """Main watcher loop — runs every poll_interval_s seconds."""
        consecutive_errors = 0
        while True:
            try:
                with self._lock:
                    order_ids = list(self._pending.keys())

                for oid in order_ids:
                    self._check_order(oid)

                consecutive_errors = 0
                time.sleep(self._interval)

            except Exception as exc:
                consecutive_errors += 1
                backoff = min(self._interval * (2 ** min(consecutive_errors - 1, 4)), 300)
                logger.error("[OLW] Loop error #%d: %s — retry in %ds", consecutive_errors, exc, backoff)
                time.sleep(backoff)

    def _check_order(self, alpaca_order_id: str) -> None:
        """Poll one order and handle state transitions."""
        with self._lock:
            rec = self._pending.get(alpaca_order_id)
        if rec is None:
            return

        # Timeout guard — stop tracking very old orders
        age_s = time.monotonic() - rec.registered_at
        if age_s > self._timeout:
            logger.warning(
                "[OLW] Order %s timed out after %.0fs — voiding and releasing CR",
                alpaca_order_id[:8], age_s,
            )
            self._handle_void(rec, "timeout")
            return

        # Fetch order status from Alpaca
        status, fill_price = self._fetch_order_status(alpaca_order_id)
        if status is None:
            return  # transient error, retry next cycle

        if status in _TERMINAL_FILLED:
            self._handle_filled(rec, fill_price or 0.0)
        elif status in _TERMINAL_VOID:
            self._handle_void(rec, status)
        elif status in _PENDING_STATES:
            logger.debug("[OLW] Order %s still pending (status=%s)", alpaca_order_id[:8], status)
        else:
            logger.warning("[OLW] Unknown order status '%s' for %s", status, alpaca_order_id[:8])

    def _fetch_order_status(self, alpaca_order_id: str):
        """
        Fetch order from Alpaca.
        Returns (status_str, fill_price_float) or (None, None) on error.
        """
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
                logger.warning("[OLW] Order %s not found in Alpaca — voiding", alpaca_order_id[:8])
                return "canceled", None
            else:
                logger.warning("[OLW] Alpaca GET order %s → HTTP %d", alpaca_order_id[:8], resp.status_code)
                return None, None
        except Exception as exc:
            logger.warning("[OLW] Fetch order %s error: %s", alpaca_order_id[:8], exc)
            return None, None

    def _handle_filled(self, rec: _PendingOrder, fill_price: float) -> None:
        """Order filled — confirm the DB trade as OPEN."""
        try:
            self._confirm(rec.db_trade_id, fill_price)
            logger.info(
                "[OLW] FILLED: %s fill_price=%.4f db_id=%d → status=OPEN",
                rec.symbol, fill_price, rec.db_trade_id,
            )
        except Exception as exc:
            logger.error("[OLW] db_confirm_fn failed for db_id=%d: %s", rec.db_trade_id, exc)
        finally:
            with self._lock:
                self._pending.pop(rec.alpaca_order_id, None)

    def _handle_void(self, rec: _PendingOrder, reason: str) -> None:
        """Order canceled/expired — void the DB trade and release CR allocation."""
        try:
            self._void(rec.db_trade_id)
            logger.info(
                "[OLW] VOID: %s reason=%s db_id=%d → releasing CR $%.0f",
                rec.symbol, reason, rec.db_trade_id, rec.amount_usd,
            )
        except Exception as exc:
            logger.error("[OLW] db_void_fn failed for db_id=%d: %s", rec.db_trade_id, exc)

        # Release Capital Router allocation regardless of DB success
        if rec.cr_trade_id and self._cr:
            try:
                self._cr.release(rec.symbol, rec.amount_usd, rec.cr_trade_id, pnl=0.0)
            except Exception as exc:
                logger.warning("[OLW] CR release failed for void %s: %s", rec.symbol, exc)

        with self._lock:
            self._pending.pop(rec.alpaca_order_id, None)
