"""
ATG Backtest Warm-Start v3.0

Loads historical trades from backtest/results/trades.csv to pre-warm the
LinUCB bandit before live trading begins.

This prevents the cold-start problem: instead of starting with all arms at
uniform prior, the bandit inherits signal from real backtested outcomes.

CSV column requirements (any superset is fine):
  symbol, setup_type, stop_multiplier, pnl_pct, [context_*] (optional)
"""
import csv
import logging
import os
import numpy as np
from pathlib import Path
from typing import Optional

from config.settings import SETUP_TYPES, STOP_MULTIPLIERS, NUM_STOP_MULTIPLIERS, CONTEXT_DIM

log = logging.getLogger(__name__)

_DEFAULT_CSV = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "backtest", "results", "trades.csv",
)


def _find_arm(setup_type: str, stop_multiplier: float) -> Optional[int]:
    """
    Map (setup_type, stop_multiplier) to a bandit arm index.

    Finds the closest stop multiplier in STOP_MULTIPLIERS by absolute difference.

    Args:
        setup_type      : one of SETUP_TYPES.
        stop_multiplier : ATR multiplier float.

    Returns:
        Arm index int, or None if setup_type is not recognised.
    """
    try:
        setup_idx = SETUP_TYPES.index(setup_type)
    except ValueError:
        return None

    dists     = np.abs(STOP_MULTIPLIERS - stop_multiplier)
    stop_idx  = int(np.argmin(dists))
    return setup_idx * NUM_STOP_MULTIPLIERS + stop_idx


def _build_neutral_context() -> np.ndarray:
    """
    Return a neutral (all-0.5) context vector for historical trades that lack
    feature data.

    Returns:
        Float32 array of shape (CONTEXT_DIM,).
    """
    return np.full(CONTEXT_DIM, 0.5, dtype=np.float32)


def warmstart_bandit(bandit, csv_path: Optional[str] = None) -> int:
    """
    Feed historical trade outcomes into the bandit to pre-warm its priors.

    Each CSV row becomes a synthetic bandit update with:
      - arm determined by (setup_type, stop_multiplier)
      - context = neutral 0.5 vector (unless context columns present)
      - reward  = clip(pnl_pct / 100 * 3, -1, +1)

    Args:
        bandit  : AutonomousSwingBandit instance.
        csv_path: path to trades CSV. Defaults to backtest/results/trades.csv.

    Returns:
        Number of rows processed.
    """
    path = csv_path or _DEFAULT_CSV

    if not Path(path).exists():
        log.warning("Backtest CSV not found at %s — skipping warm-start", path)
        return 0

    processed = 0
    skipped   = 0

    try:
        with open(path, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    setup_type      = row.get("setup_type", "").strip().upper()
                    # stop_multiplier may not be present in backtest CSV — default to 2.0
                    stop_mult_raw   = row.get("stop_multiplier") or row.get("stop_mult") or "2.0"
                    pnl_raw         = row.get("pnl_pct", "0")

                    stop_multiplier = float(stop_mult_raw)
                    pnl_pct         = float(pnl_raw)
                except (ValueError, TypeError) as parse_err:
                    log.debug("Warm-start row parse error: %s — %s", row, parse_err)
                    skipped += 1
                    continue

                arm = _find_arm(setup_type, stop_multiplier)
                if arm is None:
                    log.debug("Unknown setup_type '%s' in warm-start row — skipping", setup_type)
                    skipped += 1
                    continue

                reward  = float(np.clip(pnl_pct / 100.0 * 3.0, -1.0, 1.0))
                context = _build_neutral_context()
                bandit.update(arm, context, reward)
                processed += 1

    except OSError as e:
        log.error("Could not open backtest CSV at %s: %s", path, e)
        return 0

    log.info(
        "Bandit warm-start complete: %d trades loaded, %d skipped | total_pulls=%d",
        processed, skipped, bandit.total_pulls,
    )
    return processed
