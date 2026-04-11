"""
src/performance_tracker.py — Circuit Breaker + Daily P&L Tracking (ATG Swing)

Persists state to SQLite (system_state table) on every trade close.
Loads today's state from DB on __init__ — survives Railway restarts.

Step 35 — fix(step35): persist circuit breaker state to SQLite — ATG Swing

Circuit breakers:
  - 3% daily loss (CIRCUIT_DAILY_LOSS_PCT from settings, default 0.03)
  - 8% weekly loss (CIRCUIT_WEEKLY_LOSS_PCT from settings, default 0.08)
"""
import logging
import datetime

log = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks daily/weekly P&L and enforces circuit breakers for ATG Swing.

    State is persisted to SQLite (system_state table) on every trade.
    Loads today's state on __init__ — survives Railway restarts.
    """

    def __init__(self) -> None:
        self._daily_pnl              = 0.0
        self._weekly_pnl             = 0.0
        self._circuit_breaker_active  = False
        self._daily_loss_limit        = 0.0
        self._closed_today            = 0
        self._opened_today            = 0
        # Load persisted state from DB before accepting any trades
        self._load_from_db()

    # ── DB I/O ────────────────────────────────────────────────────────────────

    def _load_from_db(self) -> None:
        """
        Load today's circuit breaker state from SQLite.
        Fail-safe: starts fresh with zero values on any error.
        """
        try:
            from src.database import load_circuit_breaker_state
            state = load_circuit_breaker_state()
            self._daily_pnl              = state["daily_pnl"]
            self._circuit_breaker_active  = state["circuit_breaker_active"]
            self._daily_loss_limit        = state["daily_loss_limit"]
            if self._circuit_breaker_active:
                log.warning(
                    "⚠️  CB loaded ACTIVE from DB — trading halted. "
                    "daily_pnl=%.2f loss_limit=%.2f",
                    self._daily_pnl, self._daily_loss_limit,
                )
            else:
                log.info(
                    "CB state loaded: daily_pnl=%.2f active=%s",
                    self._daily_pnl, self._circuit_breaker_active,
                )
        except Exception as e:
            log.warning("CB state load failed (starting fresh with zeros): %s", e)

    def _persist_to_db(self) -> None:
        """
        Write current circuit breaker state to SQLite.
        Fail-safe: logs warning on error, never raises.
        """
        try:
            from src.database import save_circuit_breaker_state
            save_circuit_breaker_state(
                daily_pnl=self._daily_pnl,
                circuit_breaker_active=self._circuit_breaker_active,
                daily_loss_limit=self._daily_loss_limit,
            )
        except Exception as e:
            log.warning("CB state persist failed (non-fatal): %s", e)

    # ── Trade tracking ────────────────────────────────────────────────────────

    def track_trade(self, pnl: float) -> None:
        """
        Record a completed trade. Updates daily/weekly P&L counters.
        Persists state to SQLite on every call.
        """
        self._daily_pnl   += pnl
        self._weekly_pnl  += pnl
        self._closed_today += 1
        # Evaluate CB after each trade
        self._evaluate_daily_loss()
        # Persist on every trade close (Step 35 requirement)
        self._persist_to_db()

    def record_open(self) -> None:
        """Record that a new position was opened."""
        self._opened_today += 1

    def _evaluate_daily_loss(self) -> None:
        """Trip circuit breaker if daily P&L exceeds the daily loss limit."""
        if self._circuit_breaker_active:
            return
        limit = self._daily_loss_limit
        if limit > 0 and self._daily_pnl <= -limit:
            self._circuit_breaker_active = True
            log.warning(
                "🚨 CIRCUIT BREAKER TRIPPED: daily_pnl=%.2f exceeds limit=%.2f",
                self._daily_pnl, limit,
            )

    # ── Circuit breaker checks ────────────────────────────────────────────────

    def check_circuit_breakers(self) -> bool:
        """
        Returns True if circuit is OPEN (trading should halt).
        Evaluates: daily loss limit, weekly loss limit.
        Refreshes daily_loss_limit from settings on each call.
        """
        if self._circuit_breaker_active:
            return True

        try:
            from config.settings import settings
            equity = getattr(settings, "INITIAL_CAPITAL", 100_000.0)

            daily_pct  = getattr(settings, "CIRCUIT_DAILY_LOSS_PCT",  0.03)
            weekly_pct = getattr(settings, "CIRCUIT_WEEKLY_LOSS_PCT", 0.08)

            daily_limit  = equity * daily_pct
            weekly_limit = equity * weekly_pct

            # Keep daily_loss_limit in sync for persistence
            self._daily_loss_limit = daily_limit

            if self._daily_pnl <= -daily_limit:
                self._circuit_breaker_active = True
                log.warning(
                    "🚨 CIRCUIT BREAKER: daily loss %.2f >= limit %.2f (%.0f%%)",
                    -self._daily_pnl, daily_limit, daily_pct * 100,
                )
                self._persist_to_db()
                return True

            if self._weekly_pnl <= -weekly_limit:
                self._circuit_breaker_active = True
                log.warning(
                    "🚨 CIRCUIT BREAKER: weekly loss %.2f >= limit %.2f (%.0f%%)",
                    -self._weekly_pnl, weekly_limit, weekly_pct * 100,
                )
                self._persist_to_db()
                return True

        except Exception as e:
            log.warning("Circuit breaker check error (allowing trade): %s", e)

        return False

    def is_circuit_open(self) -> bool:
        """Alias for check_circuit_breakers()."""
        return self.check_circuit_breakers()

    # ── Resets ────────────────────────────────────────────────────────────────

    def reset_daily(self) -> None:
        """Call at market open (9:30 AM ET) each day. Resets daily counters."""
        self._daily_pnl              = 0.0
        self._circuit_breaker_active  = False
        self._closed_today            = 0
        self._opened_today            = 0
        self._persist_to_db()
        log.info("Daily circuit breaker reset")

    def reset_weekly(self) -> None:
        """Call on Monday open. Resets weekly P&L."""
        self._weekly_pnl = 0.0
        log.info("Weekly circuit breaker reset")

    # ── Status ────────────────────────────────────────────────────────────────

    def get_circuit_breaker_status(self) -> dict:
        """Accessor for /health endpoint — returns CB state + thresholds."""
        from config.settings import (
            CIRCUIT_DAILY_LOSS_PCT, CIRCUIT_WEEKLY_LOSS_PCT, CIRCUIT_DRAWDOWN_PCT,
        )
        return {
            "tripped":      self._circuit_breaker_active,
            "daily_pnl":   round(self._daily_pnl, 2),
            "weekly_pnl":  round(self._weekly_pnl, 2),
            "thresholds": {
                "daily":    CIRCUIT_DAILY_LOSS_PCT,
                "weekly":   CIRCUIT_WEEKLY_LOSS_PCT,
                "drawdown": CIRCUIT_DRAWDOWN_PCT,
            },
        }

    def get_stats(self) -> dict:
        return {
            "circuit_breaker_active": self._circuit_breaker_active,
            "daily_pnl":              round(self._daily_pnl, 2),
            "weekly_pnl":             round(self._weekly_pnl, 2),
            "daily_loss_limit":       round(self._daily_loss_limit, 2),
            "closed_today":           self._closed_today,
            "opened_today":           self._opened_today,
        }
