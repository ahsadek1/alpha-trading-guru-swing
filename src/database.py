"""
ATG Database v3.0 — SQLite schema + WAL mode + bandit/normalizer persistence.

All tables created in WAL mode for concurrent access safety.
Every connection enables WAL and row_factory for dict-like row access.
"""
import sqlite3
import os
import json
import logging
import datetime
from typing import Optional

from config.settings import DB_PATH

log = logging.getLogger(__name__)


def get_connection() -> sqlite3.Connection:
    """Open a WAL-mode SQLite connection with Row factory."""
    os.makedirs(os.path.dirname(os.path.abspath(DB_PATH)), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def initialize_database() -> None:
    """Create all tables if they don't exist. Safe to call on every startup."""
    conn = get_connection()
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS swing_positions (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol           TEXT    NOT NULL,
            setup_type       TEXT    NOT NULL,
            stop_multiplier  REAL    NOT NULL,
            arm_index        INTEGER NOT NULL,
            entry_date       TEXT    NOT NULL,
            entry_price      REAL    NOT NULL,
            shares           INTEGER NOT NULL,
            shares_remaining INTEGER,
            stop_loss        REAL    NOT NULL,
            target_price     REAL,
            target_stage2    REAL,
            atr_at_entry     REAL,
            sector           TEXT,
            phase            INTEGER DEFAULT 1,
            context_vector   TEXT,
            high_water_mark  REAL,
            breakeven_set    INTEGER DEFAULT 0,
            staged_exit_done INTEGER DEFAULT 0,
            addon_done       INTEGER DEFAULT 0,
            status           TEXT    DEFAULT 'OPEN',
            exit_date        TEXT,
            exit_price       REAL,
            exit_reason      TEXT,
            pnl_pct          REAL,
            pnl_dollars      REAL,
            hold_days        INTEGER,
            created_at       TEXT    DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bandit_outcomes (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            position_id      INTEGER REFERENCES swing_positions(id),
            arm_index        INTEGER NOT NULL,
            setup_type       TEXT    NOT NULL,
            stop_multiplier  REAL    NOT NULL,
            context_vector   TEXT    NOT NULL,
            reward           REAL    NOT NULL,
            phase            INTEGER NOT NULL,
            recorded_at      TEXT    DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS phase_log (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            from_phase      INTEGER,
            to_phase        INTEGER,
            trigger         TEXT,
            total_trades    INTEGER,
            win_rate        REAL,
            transitioned_at TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS system_state (
            key        TEXT PRIMARY KEY,
            value      TEXT,
            updated_at TEXT DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS bandit_state (
            id         INTEGER PRIMARY KEY CHECK (id = 1),
            state_json TEXT    NOT NULL,
            saved_at   TEXT    DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS normalizer_state (
            id         INTEGER PRIMARY KEY CHECK (id = 1),
            state_json TEXT    NOT NULL,
            saved_at   TEXT    DEFAULT (datetime('now'))
        )
    """)

    c.execute("""
        CREATE TABLE IF NOT EXISTS performance_snapshots (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            snap_date      TEXT    NOT NULL UNIQUE,
            equity         REAL,
            daily_pnl      REAL,
            total_trades   INTEGER,
            win_rate       REAL,
            open_positions INTEGER,
            created_at     TEXT DEFAULT (datetime('now'))
        )
    """)

    conn.commit()
    conn.close()
    log.info("ATG database v3.0 initialized at %s (WAL mode)", DB_PATH)


# ── Position CRUD ─────────────────────────────────────────────────────────────

def record_position_open(pos: dict) -> int:
    """Insert a new OPEN position. Returns the new row id."""
    conn = get_connection()
    c = conn.cursor()
    c.execute("""
        INSERT INTO swing_positions
        (symbol, setup_type, stop_multiplier, arm_index, entry_date, entry_price,
         shares, shares_remaining, stop_loss, target_price, target_stage2,
         atr_at_entry, sector, phase, context_vector, high_water_mark)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        pos["symbol"],
        pos["setup_type"],
        pos["stop_multiplier"],
        pos["arm_index"],
        pos["entry_date"],
        pos["entry_price"],
        pos["shares"],
        pos["shares"],                              # shares_remaining starts = shares
        pos["stop_loss"],
        pos.get("target_price"),
        pos.get("target_stage2"),
        pos.get("atr_at_entry"),
        pos.get("sector"),
        pos.get("phase", 1),
        json.dumps(pos.get("context_vector", [])),
        pos["entry_price"],                         # high_water_mark starts at entry
    ))
    pos_id = c.lastrowid
    conn.commit()
    conn.close()
    return pos_id


def record_position_close(pos_id: int, exit_data: dict) -> None:
    """Mark a position as CLOSED with exit details."""
    conn = get_connection()
    conn.execute("""
        UPDATE swing_positions SET
            status=?, exit_date=?, exit_price=?, exit_reason=?,
            pnl_pct=?, pnl_dollars=?, hold_days=?
        WHERE id=?
    """, (
        "CLOSED",
        exit_data["exit_date"],
        exit_data["exit_price"],
        exit_data["exit_reason"],
        exit_data["pnl_pct"],
        exit_data["pnl_dollars"],
        exit_data["hold_days"],
        pos_id,
    ))
    conn.commit()
    conn.close()


def update_position_stop(pos_id: int, new_stop: float, breakeven_set: bool = False) -> None:
    """Update stop loss level; optionally flag breakeven as set."""
    conn = get_connection()
    conn.execute(
        "UPDATE swing_positions SET stop_loss=?, breakeven_set=? WHERE id=?",
        (new_stop, 1 if breakeven_set else 0, pos_id),
    )
    conn.commit()
    conn.close()


def update_high_water_mark(pos_id: int, hwm: float) -> None:
    """Update peak price seen for a position (trailing stop reference)."""
    conn = get_connection()
    conn.execute(
        "UPDATE swing_positions SET high_water_mark=? WHERE id=?",
        (hwm, pos_id),
    )
    conn.commit()
    conn.close()


def mark_staged_exit_done(pos_id: int, shares_remaining: int) -> None:
    """Flag that first 50% partial exit has been executed."""
    conn = get_connection()
    conn.execute(
        "UPDATE swing_positions SET staged_exit_done=1, shares_remaining=? WHERE id=?",
        (shares_remaining, pos_id),
    )
    conn.commit()
    conn.close()


def mark_addon_done(pos_id: int) -> None:
    """Flag that an add-on position has been placed for this trade."""
    conn = get_connection()
    conn.execute(
        "UPDATE swing_positions SET addon_done=1 WHERE id=?",
        (pos_id,),
    )
    conn.commit()
    conn.close()


def record_bandit_outcome(outcome: dict) -> None:
    """Persist bandit arm + reward for post-hoc analysis."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO bandit_outcomes
        (position_id, arm_index, setup_type, stop_multiplier, context_vector, reward, phase)
        VALUES (?,?,?,?,?,?,?)
    """, (
        outcome.get("position_id"),
        outcome["arm_index"],
        outcome["setup_type"],
        outcome["stop_multiplier"],
        json.dumps(outcome.get("context_vector", [])),
        outcome["reward"],
        outcome["phase"],
    ))
    conn.commit()
    conn.close()


def get_open_positions() -> list:
    """Return all currently OPEN positions as list of dicts."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM swing_positions WHERE status='OPEN' ORDER BY entry_date"
    ).fetchall()
    conn.close()
    result = []
    for r in rows:
        d = dict(r)
        try:
            d["context_vector"] = json.loads(d.get("context_vector") or "[]")
        except json.JSONDecodeError:
            d["context_vector"] = []
        result.append(d)
    return result


def get_trade_stats() -> dict:
    """Return aggregate closed-trade statistics."""
    conn = get_connection()
    c = conn.cursor()
    total     = c.execute("SELECT COUNT(*) FROM swing_positions WHERE status='CLOSED'").fetchone()[0]
    wins      = c.execute("SELECT COUNT(*) FROM swing_positions WHERE status='CLOSED' AND pnl_pct > 0").fetchone()[0]
    avg_pnl   = c.execute("SELECT AVG(pnl_pct) FROM swing_positions WHERE status='CLOSED'").fetchone()[0] or 0.0
    total_pnl = c.execute("SELECT SUM(pnl_dollars) FROM swing_positions WHERE status='CLOSED'").fetchone()[0] or 0.0
    conn.close()
    return {
        "total_closed": total,
        "wins":         wins,
        "losses":       total - wins,
        "win_rate":     wins / total if total > 0 else 0.0,
        "avg_pnl_pct":  round(float(avg_pnl), 4),
        "total_pnl":    round(float(total_pnl), 2),
    }


# ── Bandit Persistence ────────────────────────────────────────────────────────

def save_bandit_to_db(state: dict) -> None:
    """Upsert bandit state JSON into bandit_state table (id=1 singleton)."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO bandit_state (id, state_json, saved_at)
        VALUES (1, ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            state_json=excluded.state_json,
            saved_at=excluded.saved_at
    """, (json.dumps(state),))
    conn.commit()
    conn.close()


def load_bandit_from_db() -> Optional[dict]:
    """Load bandit state JSON from DB. Returns None if not found."""
    conn = get_connection()
    row = conn.execute("SELECT state_json FROM bandit_state WHERE id=1").fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError as e:
            log.warning("Bandit state JSON malformed: %s", e)
            return None
    return None


# ── Normalizer Persistence ────────────────────────────────────────────────────

def save_normalizer_to_db(state: dict) -> None:
    """Upsert Welford normalizer state JSON."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO normalizer_state (id, state_json, saved_at)
        VALUES (1, ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
            state_json=excluded.state_json,
            saved_at=excluded.saved_at
    """, (json.dumps(state),))
    conn.commit()
    conn.close()


def load_normalizer_from_db() -> Optional[dict]:
    """Load normalizer state JSON from DB. Returns None if not found."""
    conn = get_connection()
    row = conn.execute("SELECT state_json FROM normalizer_state WHERE id=1").fetchone()
    conn.close()
    if row:
        try:
            return json.loads(row[0])
        except json.JSONDecodeError as e:
            log.warning("Normalizer state JSON malformed: %s", e)
            return None
    return None


# ── Performance Snapshots ─────────────────────────────────────────────────────

def save_snapshot(snap: dict) -> None:
    """Insert or update a daily equity snapshot."""
    conn = get_connection()
    conn.execute("""
        INSERT INTO performance_snapshots
        (snap_date, equity, daily_pnl, total_trades, win_rate, open_positions)
        VALUES (?,?,?,?,?,?)
        ON CONFLICT(snap_date) DO UPDATE SET
            equity=excluded.equity,
            daily_pnl=excluded.daily_pnl,
            total_trades=excluded.total_trades,
            win_rate=excluded.win_rate,
            open_positions=excluded.open_positions
    """, (
        snap["snap_date"],
        snap.get("equity"),
        snap.get("daily_pnl"),
        snap.get("total_trades", 0),
        snap.get("win_rate", 0.0),
        snap.get("open_positions", 0),
    ))
    conn.commit()
    conn.close()


def get_snapshots(limit: int = 30) -> list:
    """Return recent daily snapshots ordered newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM performance_snapshots ORDER BY snap_date DESC LIMIT ?",
        (limit,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ── Circuit Breaker Persistence (Step 35) ─────────────────────────────────────

def save_circuit_breaker_state(daily_pnl: float, circuit_breaker_active: bool,
                                daily_loss_limit: float, date_str: str = None) -> None:
    """
    Persist circuit breaker state for today to system_state table.
    Fail-safe: logs warning on error, never raises.
    """
    try:
        today = date_str or datetime.date.today().isoformat()
        conn = get_connection()
        for key, value in [
            (f"cb_daily_pnl_{today}",        str(round(daily_pnl, 4))),
            (f"cb_active_{today}",            "1" if circuit_breaker_active else "0"),
            (f"cb_daily_loss_limit_{today}",  str(round(daily_loss_limit, 4))),
        ]:
            conn.execute("""
                INSERT INTO system_state (key, value, updated_at)
                VALUES (?, ?, datetime('now'))
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
            """, (key, value))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning("Circuit breaker state save failed (non-fatal): %s", e)


def load_circuit_breaker_state(date_str: str = None) -> dict:
    """
    Load today's circuit breaker state from system_state table.
    Returns default (zeroed) state if no record found. Never raises.
    """
    defaults = {
        "daily_pnl":             0.0,
        "circuit_breaker_active": False,
        "daily_loss_limit":      0.0,
    }
    try:
        today = date_str or datetime.date.today().isoformat()
        conn  = get_connection()
        rows  = conn.execute(
            "SELECT key, value FROM system_state WHERE key LIKE ?",
            (f"cb_%_{today}",)
        ).fetchall()
        conn.close()
        if not rows:
            return defaults
        row_map = {r["key"]: r["value"] for r in rows}
        return {
            "daily_pnl":             float(row_map.get(f"cb_daily_pnl_{today}", 0.0)),
            "circuit_breaker_active": row_map.get(f"cb_active_{today}", "0") == "1",
            "daily_loss_limit":      float(row_map.get(f"cb_daily_loss_limit_{today}", 0.0)),
        }
    except Exception as e:
        log.warning("Circuit breaker state load failed (starting fresh): %s", e)
        return defaults


# ── OLW + reconciler helpers (Steps 30–31) ────────────────────────────────────

def get_pending_positions() -> list:
    """Return all positions with status='pending_open'."""
    try:
        with _db() as conn:
            rows = conn.execute(
                "SELECT * FROM positions WHERE status='pending_open'"
            ).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        log.warning("get_pending_positions error: %s", e)
        return []


def void_position(db_id: int, reason: str) -> None:
    """Mark a position as voided (order rejected / canceled / aged out)."""
    try:
        with _db() as conn:
            conn.execute(
                "UPDATE positions SET status='voided', exit_reason=? WHERE id=?",
                (reason, db_id),
            )
            conn.commit()
        log.info("Position %d voided: %s", db_id, reason)
    except Exception as e:
        log.warning("void_position(%d) error: %s", db_id, e)


def confirm_position_open(db_id: int, filled_price: float) -> None:
    """Mark a pending position as open once fill confirmed."""
    try:
        with _db() as conn:
            conn.execute(
                "UPDATE positions SET status='open', entry_price=? WHERE id=?",
                (filled_price, db_id),
            )
            conn.commit()
        log.info("Position %d confirmed open @ %.2f", db_id, filled_price)
    except Exception as e:
        log.warning("confirm_position_open(%d) error: %s", db_id, e)
