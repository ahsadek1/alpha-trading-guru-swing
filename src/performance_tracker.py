"""
src/performance_tracker.py — Circuit Breaker + Daily P&L Tracking (ATG Swing)

FIX [F1]: Replaced broken `from config.settings import settings` (ImportError)
           with direct constant imports. CB is now functional.
FIX [F6]: Added drawdown circuit breaker with equity_peak tracking.
FIX [F2b]: Added equity_peak persistence for drawdown calculation.

Persists state to SQLite (system_state table) on every trade close.
Loads today's state from DB on __init__ — survives Railway restarts.
"""
import logging
import datetime

log = logging.getLogger(__name__)


class PerformanceTracker:
    """
    Tracks daily/weekly P&L and enforces circuit breakers for ATG Swing.

    State is persisted to SQLite (system_state table) on every trade.
    Loads today's state on __init__ — survives Railway restarts.

    Circuit Breakers (all checked before any new trade):
      - Daily loss limit (CIRCUIT_DAILY_LOSS_PCT × INITIAL_CAPITAL)
      - Weekly loss limit (CIRCUIT_WEEKLY_LOSS_PCT × INITIAL_CAPITAL)
      - Drawdown from equity peak (CIRCUIT_DRAWDOWN_PCT × equity_peak)  [F6]
    """

    def __init__(self) -> None:
        self._daily_pnl              = 0.0
        self._weekly_pnl             = 0.0
        self._circuit_breaker_active  = False
        self._daily_loss_limit        = 0.0
        self._closed_today            = 0
        self._opened_today            = 0
        self._equity_peak             = 0.0   # [F6] track peak for drawdown CB
        self._current_equity          = 0.0   # [F6] updated via report_equity()
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
            self._equity_peak             = state.get("equity_peak", 0.0)
            self._current_equity          = state.get("current_equity", 0.0)
            if self._circuit_breaker_active:
                log.warning(
                    "⚠️  CB loaded ACTIVE from DB — trading halted. "
                    "daily_pnl=%.2f loss_limit=%.2f",
                    self._daily_pnl, self._daily_loss_limit,
                )
            else:
                log.info(
                    "CB state loaded: daily_pnl=%.2f active=%s equity_peak=%.2f",
                    self._daily_pnl, self._circuit_breaker_active, self._equity_peak,
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
                equity_peak=self._equity_peak,
                current_equity=self._current_equity,
            )
        except Exception as e:
            log.warning("CB state persist failed (non-fatal): %s", e)

    # ── Equity tracking (for drawdown CB) ─────────────────────────────────────

    def report_equity(self, current_equity: float) -> None:
        """
        Update current equity and equity peak.
        Call whenever Capital Router reports current account value.
        [F6] Enables drawdown circuit breaker evaluation.
        """
        self._current_equity = current_equity
        if current_equity > self._equity_peak:
            self._equity_peak = current_equity
            log.debug("New equity peak: %.2f", self._equity_peak)
        self._persist_to_db()

    # ── Trade tracking ────────────────────────────────────────────────────────

    def track_trade(self, pnl: float) -> None:
        """
        Record a completed trade. Updates daily/weekly P&L counters.
        Persists state to SQLite on every call.
        """
        self._daily_pnl   += pnl
        self._weekly_pnl  += pnl
        self._closed_today += 1
        self._evaluate_daily_loss()
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

        FIX [F1]: Replaced broken `from config.settings import settings`
                  with direct constant imports. CB is now functional.
        FIX [F6]: Added drawdown CB from equity peak.

        Evaluates:
          1. Daily loss limit (CIRCUIT_DAILY_LOSS_PCT × INITIAL_CAPITAL)
          2. Weekly loss limit (CIRCUIT_WEEKLY_LOSS_PCT × INITIAL_CAPITAL)
          3. Drawdown from peak (CIRCUIT_DRAWDOWN_PCT × equity_peak)
        """
        if self._circuit_breaker_active:
            return True

        # FIX [F1]: Direct imports — no more `from config.settings import settings`
        try:
            from config.settings import (
                INITIAL_CAPITAL,
                CIRCUIT_DAILY_LOSS_PCT,
                CIRCUIT_WEEKLY_LOSS_PCT,
                CIRCUIT_DRAWDOWN_PCT,
            )
        except ImportError as e:
            log.error("CRITICAL: Cannot import CB thresholds from settings: %s", e)
            return True  # fail-closed: halt if we can't read limits

        equity       = INITIAL_CAPITAL
        daily_limit  = equity * CIRCUIT_DAILY_LOSS_PCT
        weekly_limit = equity * CIRCUIT_WEEKLY_LOSS_PCT

        # Keep daily_loss_limit in sync for persistence
        self._daily_loss_limit = daily_limit

        # [1] Daily loss check
        if self._daily_pnl <= -daily_limit:
            self._circuit_breaker_active = True
            log.warning(
                "🚨 CIRCUIT BREAKER: daily loss $%.2f >= limit $%.2f (%.0f%%)",
                -self._daily_pnl, daily_limit, CIRCUIT_DAILY_LOSS_PCT * 100,
            )
            self._persist_to_db()
            self._alert_ahmed(f"🚨 ATG DAILY CB TRIPPED: -${-self._daily_pnl:.0f} (limit ${daily_limit:.0f})")
            return True

        # [2] Weekly loss check
        if self._weekly_pnl <= -weekly_limit:
            self._circuit_breaker_active = True
            log.warning(
                "🚨 CIRCUIT BREAKER: weekly loss $%.2f >= limit $%.2f (%.0f%%)",
                -self._weekly_pnl, weekly_limit, CIRCUIT_WEEKLY_LOSS_PCT * 100,
            )
            self._persist_to_db()
            self._alert_ahmed(f"🚨 ATG WEEKLY CB TRIPPED: -${-self._weekly_pnl:.0f} (limit ${weekly_limit:.0f})")
            return True

        # [3] FIX [F6]: Drawdown from equity peak
        if self._equity_peak > 0 and self._current_equity > 0:
            drawdown = (self._equity_peak - self._current_equity) / self._equity_peak
            if drawdown >= CIRCUIT_DRAWDOWN_PCT:
                self._circuit_breaker_active = True
                log.warning(
                    "🚨 CIRCUIT BREAKER: drawdown %.1f%% from peak $%.0f (limit %.0f%%)",
                    drawdown * 100, self._equity_peak, CIRCUIT_DRAWDOWN_PCT * 100,
                )
                self._persist_to_db()
                self._alert_ahmed(
                    f"🚨 ATG DRAWDOWN CB TRIPPED: {drawdown*100:.1f}% from peak ${self._equity_peak:.0f}"
                )
                return True

        return False

    def is_circuit_open(self) -> bool:
        """Alias for check_circuit_breakers()."""
        return self.check_circuit_breakers()

    def _alert_ahmed(self, message: str) -> None:
        """Send Telegram alert to Ahmed on CB trip. Non-blocking, never raises."""
        import os, json, urllib.request
        try:
            token    = os.getenv("TELEGRAM_BOT_TOKEN", "")
            chat_id  = os.getenv("TELEGRAM_AHMED_ID", "8573754783")
            if not token:
                return
            payload = json.dumps({"chat_id": chat_id, "text": message}).encode()
            req = urllib.request.Request(
                f"https://api.telegram.org/bot{token}/sendMessage",
                data=payload, headers={"Content-Type": "application/json"}
            )
            urllib.request.urlopen(req, timeout=5)
        except Exception:
            pass  # Never let alert failure affect CB logic

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
        try:
            from config.settings import (
                CIRCUIT_DAILY_LOSS_PCT, CIRCUIT_WEEKLY_LOSS_PCT, CIRCUIT_DRAWDOWN_PCT,
            )
        except ImportError:
            CIRCUIT_DAILY_LOSS_PCT = CIRCUIT_WEEKLY_LOSS_PCT = CIRCUIT_DRAWDOWN_PCT = 0.0

        drawdown_pct = 0.0
        if self._equity_peak > 0 and self._current_equity > 0:
            drawdown_pct = (self._equity_peak - self._current_equity) / self._equity_peak

        return {
            "tripped":       self._circuit_breaker_active,
            "daily_pnl":    round(self._daily_pnl, 2),
            "weekly_pnl":   round(self._weekly_pnl, 2),
            "equity_peak":  round(self._equity_peak, 2),
            "drawdown_pct": round(drawdown_pct, 4),
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
            "equity_peak":            round(self._equity_peak, 2),
            "current_equity":         round(self._current_equity, 2),
            "closed_today":           self._closed_today,
            "opened_today":           self._opened_today,
        }
