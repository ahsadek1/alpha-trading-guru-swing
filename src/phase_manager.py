"""
ATG Phase Manager v3.0 — Phase transition logic for the self-evolving bandit.

Phase 1 → Linear Bandit
Phase 2 → Neural Bandit
Phase 3 → Distributional RL
Phase 4 → Causal Discovery

Transition criteria checked after each EOD scan cycle.
"""
import logging
from datetime import datetime
from typing import Optional

from src.database import get_trade_stats, get_connection
from config.settings import (
    PHASE_1_MIN_TRADES, PHASE_1_MIN_WIN_RATE, PHASE_1_MIN_DAYS,
    PHASE_2_MIN_TRADES, PHASE_2_MIN_WIN_RATE, PHASE_2_MIN_DAYS,
    PHASE_3_MIN_TRADES, PHASE_3_MIN_WIN_RATE, PHASE_3_MIN_DAYS,
    PERFORMANCE_WINDOW,
)

log = logging.getLogger(__name__)


def get_current_phase() -> int:
    """
    Read the current phase from system_state table.

    Returns:
        Phase integer (1–4), defaulting to 1 if not yet set.
    """
    conn = get_connection()
    row  = conn.execute(
        "SELECT value FROM system_state WHERE key='current_phase'"
    ).fetchone()
    conn.close()
    return int(row["value"]) if row else 1


def _set_phase(phase: int, trigger: str = "") -> None:
    """
    Persist the new phase to system_state and record a phase_log entry.

    Args:
        phase   : target phase number.
        trigger : human-readable reason for the transition.
    """
    stats        = get_trade_stats()
    current      = get_current_phase()
    conn         = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO system_state (key, value) VALUES ('current_phase', ?)",
        (str(phase),),
    )
    conn.execute("""
        INSERT INTO phase_log (from_phase, to_phase, trigger, total_trades, win_rate)
        VALUES (?, ?, ?, ?, ?)
    """, (current, phase, trigger, stats["total_closed"], stats["win_rate"]))
    conn.commit()
    conn.close()
    log.info("🚀 Phase transition → Phase %d | trigger: %s", phase, trigger)


def get_phase_start_date() -> datetime:
    """
    Return the UTC datetime when the current phase began.

    Creates and stores the start date on first call.

    Returns:
        datetime (UTC) of phase start.
    """
    conn = get_connection()
    row  = conn.execute(
        "SELECT value FROM system_state WHERE key='phase_start_date'"
    ).fetchone()
    conn.close()
    if row:
        return datetime.fromisoformat(row["value"])
    start = datetime.utcnow()
    _set_phase_start_date(start)
    return start


def _set_phase_start_date(dt: datetime) -> None:
    """
    Persist the phase start date.

    Args:
        dt: start datetime (UTC).
    """
    conn = get_connection()
    conn.execute(
        "INSERT OR REPLACE INTO system_state (key, value) VALUES ('phase_start_date', ?)",
        (dt.isoformat(),),
    )
    conn.commit()
    conn.close()


def get_recent_win_rate(window: int = PERFORMANCE_WINDOW) -> float:
    """
    Calculate win rate over the most recent `window` closed trades.

    Args:
        window: number of recent trades to consider.

    Returns:
        Win rate in [0, 1].
    """
    conn = get_connection()
    rows = conn.execute("""
        SELECT pnl_pct FROM swing_positions
        WHERE status='CLOSED'
        ORDER BY id DESC
        LIMIT ?
    """, (window,)).fetchall()
    conn.close()
    if not rows:
        return 0.0
    wins = sum(1 for r in rows if r["pnl_pct"] > 0)
    return wins / len(rows)


def check_phase_transition(current_phase: int) -> Optional[int]:
    """
    Check whether transition criteria are met for the current phase.

    Args:
        current_phase: phase currently active (1–4).

    Returns:
        New phase integer if transition should happen, else None.
    """
    stats      = get_trade_stats()
    total      = stats["total_closed"]
    win_rate   = get_recent_win_rate()
    days_alive = (datetime.utcnow() - get_phase_start_date()).days

    if current_phase == 1:
        if (total >= PHASE_1_MIN_TRADES
                and win_rate >= PHASE_1_MIN_WIN_RATE
                and days_alive >= PHASE_1_MIN_DAYS):
            return 2
    elif current_phase == 2:
        if (total >= PHASE_2_MIN_TRADES
                and win_rate >= PHASE_2_MIN_WIN_RATE
                and days_alive >= PHASE_2_MIN_DAYS):
            return 3
    elif current_phase == 3:
        if (total >= PHASE_3_MIN_TRADES
                and win_rate >= PHASE_3_MIN_WIN_RATE
                and days_alive >= PHASE_3_MIN_DAYS):
            return 4

    return None


def maybe_advance_phase(orchestrator) -> bool:
    """
    Check and execute a phase transition if criteria are met.

    Calls orchestrator.on_phase_transition() when advancing.

    Args:
        orchestrator: ATGOrchestrator instance.

    Returns:
        True if a transition occurred, False otherwise.
    """
    current   = get_current_phase()
    if current >= 4:
        return False

    # FIX [F16]: Block advancement beyond implemented phases
    # Phases 2-4 are design stubs — advancing causes theater (DB says Phase 2, code runs Phase 1)
    MAX_IMPLEMENTED_PHASE = 1
    if current >= MAX_IMPLEMENTED_PHASE:
        log.debug("Phase %d criteria may be met but phase not yet implemented — holding", current + 1)
        return False

    new_phase = check_phase_transition(current)
    if new_phase:
        # FIX [F17]: Regime diversity is not yet tracked (swing_positions has no regime_label)
        # Phase advancement additionally requires: distinct regimes >= 3, stress trades >= 10
        # Until regime_label column is added, this check is skipped (gated by F16 anyway)
        _set_phase(new_phase, trigger="auto_criteria_met")
        _set_phase_start_date(datetime.utcnow())
        orchestrator.on_phase_transition(current, new_phase)
        return True
    return False
