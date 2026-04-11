"""
ATG Self-Evolving Orchestrator v3.0

The central engine of Alpha Trading Guru.  Coordinates:
  1. Database init + bandit restore (with backtest warm-start on cold start)
  2. Market regime gate check
  3. EOD swing scan (weekdays 15:30 ET)
  4. Quad-Intelligence trade validation
  5. Position opening with conviction-based sizing
  6. Position monitoring (every 15 min during market hours)
  7. Add-on opportunity checks
  8. Phase management & promotion
  9. Daily + weekly Telegram summaries
  10. Circuit breaker enforcement

Scheduling: run_cycle() is called by the FastAPI background scheduler
in main.py on a cron-like schedule.
"""
import logging
from datetime import datetime, time
from typing import Optional
import pytz
import numpy as np

from config.settings import (
    CIRCUIT_DAILY_LOSS_PCT,
    CIRCUIT_WEEKLY_LOSS_PCT,
    CIRCUIT_DRAWDOWN_PCT,
    INITIAL_CAPITAL,
)
from src.database import (
    initialize_database,
    load_bandit_from_db,
    save_bandit_to_db,
    get_trade_stats,
    get_open_positions,
    save_snapshot,
)
from src.bandit import AutonomousSwingBandit
from src.backtest_warmstart import warmstart_bandit
from src.market_regime import is_market_open_for_trading
from src.swing_scanner import run_swing_scan
from src.context_builder import build_context
from src.trade_executor import (
    open_swing_position,
    monitor_positions,
    check_addon_opportunities,
)
from src.quad_intelligence import quad_validate
from src.phase_manager import (
    get_current_phase,
    maybe_advance_phase,
    get_recent_win_rate,
)
from src import telegram_bot as tg
from src.performance_tracker import PerformanceTracker

log = logging.getLogger(__name__)
ET  = pytz.timezone("America/New_York")


class ATGOrchestrator:
    """
    Central orchestrator for the ATG Swing trading system.

    Responsibilities:
    - Initialise and persist the bandit across restarts
    - Enforce circuit breakers
    - Drive the EOD scan → validate → execute workflow
    - Manage position monitoring and exit events
    - Trigger phase transitions when criteria are met
    """

    def __init__(self) -> None:
        """Initialise orchestrator state (does not start DB or bandit yet)."""
        self.bandit: Optional[AutonomousSwingBandit] = None
        self.phase: int = 1
        self._perf_tracker:  PerformanceTracker = PerformanceTracker()
        # Legacy aliases kept for backward compat with existing callers
        self._circuit_tripped: bool = False
        self._daily_pnl: float = 0.0
        self._peak_equity: float = INITIAL_CAPITAL

    # ── Startup ───────────────────────────────────────────────────────────────

    def startup(self) -> None:
        """
        Full initialisation sequence called once at process start.

        1. Init database (WAL mode)
        2. Load or create bandit (with warm-start on fresh DB)
        3. Restore current phase
        4. Send Telegram startup card
        """
        initialize_database()

        # Restore or create bandit
        self.bandit = AutonomousSwingBandit(alpha=1.0)
        saved_state = load_bandit_from_db()
        if saved_state:
            self.bandit.load_state(saved_state)
            log.info("Bandit restored from DB | total_pulls=%d", self.bandit.total_pulls)
        else:
            log.info("Cold start — running backtest warm-start …")
            n = warmstart_bandit(self.bandit)
            save_bandit_to_db(self.bandit.get_state())
            log.info("Warm-start complete: %d trades ingested", n)

        self.phase = get_current_phase()

        stats = get_trade_stats()
        tg.send_startup_card(
            phase=self.phase,
            total_trades=stats["total_closed"],
            win_rate=stats["win_rate"],
        )
        log.info(
            "ATGOrchestrator v3 started | phase=%d total_trades=%d win_rate=%.1f%%",
            self.phase, stats["total_closed"], stats["win_rate"] * 100,
        )

    # ── Circuit breakers ──────────────────────────────────────────────────────

    def _check_circuit_breakers(self) -> bool:
        """
        Delegate to PerformanceTracker circuit breaker checks.
        Returns True if a circuit breaker is tripped (halt trading).
        """
        tripped = self._perf_tracker.check_circuit_breakers()
        self._circuit_tripped = tripped  # keep legacy alias in sync
        return tripped

    def reset_daily_state(self) -> None:
        """Reset per-day counters. Call at market open each day."""
        self._perf_tracker.reset_daily()
        self._circuit_tripped = False
        self._daily_pnl       = 0.0
        log.info("Daily circuit breaker reset (ATG_SWING)")

    def reset_weekly_state(self) -> None:
        """Reset weekly P&L. Call on Monday open."""
        self._perf_tracker.reset_weekly()
        log.info("Weekly circuit breaker reset (ATG_SWING)")

    # ── EOD scan cycle ────────────────────────────────────────────────────────

    def run_eod_scan(self) -> None:
        """
        End-of-day scan and trade-entry workflow.

        Steps:
          1. Regime gate
          2. Run scanner
          3. For each setup: build context → bandit selects arm → QI validates
          4. If QI proceeds: open position and send alert
          5. Save bandit state
        """
        if self._check_circuit_breakers():
            log.warning("EOD scan skipped: circuit breaker active")
            return

        scan_result = run_swing_scan(top_n=5)

        if not scan_result["ok"]:
            tg.send_scan_result([], scan_type="EOD", gate_reason=scan_result["reason"])
            return

        setups = scan_result["setups"]
        tg.send_scan_result(setups, scan_type="EOD")

        for setup in setups:
            try:
                self._process_setup(setup)
            except Exception as e:
                log.error("Error processing setup %s: %s", setup.get("symbol"), e)

        save_bandit_to_db(self.bandit.get_state())

    def _process_setup(self, setup: dict) -> None:
        """
        Process a single scanner setup: validate + open position.

        Args:
            setup: scanner result dict.
        """
        symbol = setup["symbol"]
        sector = setup.get("sector", "Technology")

        # Build context vector
        from src.market_regime import get_regime
        regime = get_regime()
        ctx    = build_context(symbol, sector=sector, regime=regime)

        # Bandit arm selection
        arm_idx      = self.bandit.select_arm(ctx)
        setup_str, stop_mult = self.bandit.decode_arm(arm_idx)
        selection    = {"setup_type": setup_str, "stop_multiplier": stop_mult}

        # Quad-Intelligence validation
        qi = quad_validate(setup, selection)
        if not qi["proceed"]:
            log.info(
                "QI BLOCKED %s | consensus=%s | risks=%s",
                symbol, qi["consensus"], qi.get("key_risks"),
            )
            return

        # Open position
        result = open_swing_position(
            scan_result   = setup,
            arm_index     = arm_idx,
            stop_multiplier = stop_mult,
            context_vector  = ctx,
            phase           = self.phase,
        )

        if result["status"] == "OPENED":
            self._perf_tracker.record_open()
            tg.send_trade_opened(result, qi=qi)
            log.info("Trade opened: %s | arm=%d | QI=%s", symbol, arm_idx, qi["consensus"])
        else:
            log.info("Trade not opened for %s: %s", symbol, result.get("reason"))

    # ── Position monitoring ───────────────────────────────────────────────────

    def run_position_monitor(self) -> None:
        """
        Run position monitoring cycle (called every 15 min during market hours).

        Checks stops, targets, time stops, and trailing stops.
        Sends Telegram alert for every close event.
        Updates daily P&L tracker.
        """
        if not get_open_positions():
            return

        events = monitor_positions(self.bandit)

        for ev in events:
            if ev.get("status") in ("CLOSED",):
                pnl = ev.get("pnl_dollars", 0)
                self._daily_pnl += pnl
                self._perf_tracker.track_trade(pnl)  # evaluate CB + persist state
                tg.send_trade_closed(ev)

        # Check add-on opportunities
        addons = check_addon_opportunities(self.bandit)
        for addon in addons:
            tg.send_alert(
                f"➕ Add-on placed: *{addon['symbol']}* +{addon['addon_shares']} shares "
                f"@ ${addon['current_price']:.2f} (up {addon['gain_pct']:.1f}%)"
            )

        if events:
            save_bandit_to_db(self.bandit.get_state())

    # ── Daily summary ─────────────────────────────────────────────────────────

    def run_daily_summary(self) -> None:
        """
        Send daily performance summary and save equity snapshot.
        Called at 16:30 ET each trading day.
        """
        stats      = get_trade_stats()
        open_count = len(get_open_positions())
        best       = self.bandit.best_setup()

        tg.send_daily_summary(
            phase      = self.phase,
            open_count = open_count,
            stats      = stats,
            best_setup = best,
        )

        # Save equity snapshot
        equity = INITIAL_CAPITAL + stats["total_pnl"]
        save_snapshot({
            "snap_date":       datetime.now(ET).date().isoformat(),
            "equity":          round(equity, 2),
            "daily_pnl":       round(self._daily_pnl, 2),
            "total_trades":    stats["total_closed"],
            "win_rate":        round(stats["win_rate"], 4),
            "open_positions":  open_count,
        })

        # Check phase promotion
        if maybe_advance_phase(self):
            pass  # on_phase_transition handles alerting

    # ── Weekly report ─────────────────────────────────────────────────────────

    def run_weekly_report(self) -> None:
        """
        Send weekly performance report (call on Fridays at 16:00 ET).
        Computes win/loss breakdown for the past 5 trading days.
        """
        from src.database import get_connection
        conn = get_connection()
        rows = conn.execute("""
            SELECT pnl_pct, pnl_dollars FROM swing_positions
            WHERE status='CLOSED'
            AND date(exit_date) >= date('now', '-7 days')
        """).fetchall()
        conn.close()

        total_trades = len(rows)
        wins         = [r["pnl_pct"] for r in rows if r["pnl_pct"] > 0]
        losses       = [r["pnl_pct"] for r in rows if r["pnl_pct"] <= 0]
        total_pnl    = sum(r["pnl_dollars"] for r in rows)

        tg.send_weekly_report({
            "total_trades": total_trades,
            "wins":         len(wins),
            "losses":       len(losses),
            "win_rate":     len(wins) / total_trades if total_trades > 0 else 0.0,
            "avg_win_pct":  float(np.mean(wins))   if wins   else 0.0,
            "avg_loss_pct": float(np.mean(losses))  if losses else 0.0,
            "total_pnl":    round(total_pnl, 2),
            "phase":        self.phase,
        })

    # ── Phase transition callback ─────────────────────────────────────────────

    def on_phase_transition(self, from_phase: int, to_phase: int) -> None:
        """
        Called by maybe_advance_phase() when a transition occurs.

        Updates instance phase and sends Telegram alert.

        Args:
            from_phase : phase being exited.
            to_phase   : new phase.
        """
        self.phase = to_phase
        stats      = get_trade_stats()
        tg.send_phase_transition(from_phase, to_phase, stats)
        log.info("Phase transition: %d → %d", from_phase, to_phase)

    # ── Health check data ─────────────────────────────────────────────────────

    def health_data(self) -> dict:
        """
        Return health check payload for the FastAPI /health endpoint.

        Returns:
            Dict suitable for JSON serialisation.
        """
        stats      = get_trade_stats()
        open_pos   = get_open_positions()
        best       = self.bandit.best_setup()
        equity     = INITIAL_CAPITAL + stats["total_pnl"]
        win_rate   = get_recent_win_rate()

        return {
            "status":           "ok",
            "system":           "ATG_SWING",
            "version":          "3.0.0",
            "phase":            self.phase,
            "paper_mode":       True,
            "total_trades":     stats["total_closed"],
            "win_rate":         round(win_rate, 4),
            "open_positions":   len(open_pos),
            "total_pnl":        stats["total_pnl"],
            "equity":           round(equity, 2),
            "bandit_pulls":     self.bandit.total_pulls,
            "best_setup":       best,
            "circuit_tripped":  self._circuit_tripped,
            "daily_pnl":        round(self._daily_pnl, 2),
            "circuit_breaker":  self._perf_tracker.get_circuit_breaker_status(),
            "timestamp":        datetime.now(ET).isoformat(),
        }
