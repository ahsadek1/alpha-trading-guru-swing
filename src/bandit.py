"""
ATG Bandit v3.0 — AutonomousSwingBandit (LinUCB contextual bandit).

Arms encode (setup_type × stop_multiplier) pairs.
Mirrors ATM's AutonomousDeltaBandit architecture.
"""
import numpy as np
import logging
from typing import Tuple

from config.settings import (
    NUM_BANDIT_ARMS,
    NUM_SETUP_TYPES,
    NUM_STOP_MULTIPLIERS,
    SETUP_TYPES,
    STOP_MULTIPLIERS,
    CONTEXT_DIM,
)

log = logging.getLogger(__name__)


class AutonomousSwingBandit:
    """
    Linear contextual bandit for swing setup + stop multiplier selection.

    Arms  : NUM_SETUP_TYPES × NUM_STOP_MULTIPLIERS  (6 × 11 = 66)
    Context : CONTEXT_DIM-dimensional market feature vector
    Reward  : trade P&L % scaled to [-1, +1]

    Algorithm: LinUCB — A matrix + b vector per arm.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        """
        Initialise bandit with one A/b pair per arm.

        Args:
            alpha: LinUCB exploration parameter.
        """
        self.n_arms      = NUM_BANDIT_ARMS
        self.alpha       = alpha
        self.context_dim = CONTEXT_DIM

        # LinUCB: A_i ∈ R^{d×d}, b_i ∈ R^d
        self.A = [np.eye(CONTEXT_DIM) * alpha for _ in range(self.n_arms)]
        self.b = [np.zeros(CONTEXT_DIM)       for _ in range(self.n_arms)]

        self.total_pulls  = 0
        self.arm_pulls    = np.zeros(self.n_arms, dtype=int)
        self.arm_rewards  = np.zeros(self.n_arms, dtype=float)
        self.win_rates    = np.zeros(self.n_arms, dtype=float)

        # Warm-start: bias arms toward historically optimal setups and stop multipliers
        self._load_warmstart_priors()

        log.info(
            "AutonomousSwingBandit v3 initialized | arms=%d (%d setup types × %d stop mults)",
            self.n_arms, NUM_SETUP_TYPES, NUM_STOP_MULTIPLIERS,
        )

    def _load_warmstart_priors(self) -> None:
        """
        Load warm-start biases from data/bandit_priors.json and inject into
        LinUCB A and b matrices as virtual trades.

        Formula (per Omni spec):
            A[arm] = alpha * eye(dim) + PRIOR_TRADES * eye(dim)
            b[arm] = bias * PRIOR_TRADES * ones(dim)

        Called once at __init__. If priors file missing/corrupt → silent no-op,
        A/b remain at default (current behavior preserved).
        Subsequent load_state() calls with real DB data OVERWRITE these matrices.
        """
        import json, os
        priors_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data", "bandit_priors.json"
        )
        try:
            with open(priors_path) as f:
                priors = json.load(f)

            prior_trades = int(priors.get("prior_trades", 50))
            arm_cfg      = priors.get("arms", {})
            eye          = np.eye(self.context_dim)
            ones         = np.ones(self.context_dim)
            applied      = 0

            for arm_str, cfg in arm_cfg.items():
                arm_idx = int(arm_str)
                if arm_idx >= self.n_arms:
                    continue
                bias = float(cfg.get("bias", 0.0))
                # Omni spec: A[arm] = alpha*eye + PRIOR_TRADES*eye; b[arm] = bias*PRIOR_TRADES*ones
                self.A[arm_idx] = self.alpha * eye + prior_trades * eye
                self.b[arm_idx] = bias * prior_trades * ones
                applied += 1

            log.info(
                "Bandit warm-start loaded: %d/%d arms primed (prior_trades=%d)",
                applied, self.n_arms, prior_trades,
            )
        except FileNotFoundError:
            log.debug("No bandit_priors.json at %s — using uniform prior", priors_path)
        except Exception as e:
            log.warning("Bandit warm-start failed (non-fatal, uniform prior used): %s", e)


    def select_arm(self, context: np.ndarray) -> int:
        """
        LinUCB arm selection.

        Args:
            context: 1-D array of shape (CONTEXT_DIM,).

        Returns:
            Index of the selected arm.
        """
        assert context.shape == (self.context_dim,), (
            f"Context must be {self.context_dim}-dim, got {context.shape}"
        )
        ucb_scores = np.zeros(self.n_arms)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for i in range(self.n_arms):
                A_inv    = np.linalg.inv(self.A[i])
                theta    = A_inv @ self.b[i]
                variance = float(context @ A_inv @ context)
                # Clamp to [0, ∞) to guard against numerical noise
                variance = max(variance, 0.0)
                ucb_scores[i] = (
                    float(np.dot(theta, context))
                    + self.alpha * np.sqrt(variance)
                )
        # Replace any NaN/Inf with 0 before argmax
        ucb_scores = np.nan_to_num(ucb_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, context: np.ndarray, reward: float) -> None:
        """
        Update LinUCB parameters after observing a reward.

        Args:
            arm    : arm index that was pulled.
            context: context vector used at selection time.
            reward : scaled reward in [-1, +1].
        """
        self.A[arm]           += np.outer(context, context)
        self.b[arm]           += reward * context
        self.arm_pulls[arm]   += 1
        self.arm_rewards[arm] += reward
        self.total_pulls      += 1

        # Rolling win rate
        n                    = self.arm_pulls[arm]
        prev                 = self.win_rates[arm]
        self.win_rates[arm]  = prev + (1.0 if reward > 0 else 0.0 - prev) / n

        log.debug(
            "Bandit update | arm=%d setup=%s stop=%.2fx reward=%.4f",
            arm, *self.decode_arm(arm), reward,
        )

    def decode_arm(self, arm: int) -> Tuple[str, float]:
        """
        Convert arm index to (setup_type, stop_multiplier).

        Args:
            arm: Arm index in [0, n_arms).

        Returns:
            Tuple of (setup_type string, stop_multiplier float).
        """
        setup_idx = arm // NUM_STOP_MULTIPLIERS
        stop_idx  = arm %  NUM_STOP_MULTIPLIERS
        return SETUP_TYPES[setup_idx], round(float(STOP_MULTIPLIERS[stop_idx]), 2)

    def best_setup(self) -> dict:
        """
        Return the arm with highest expected reward under a unit context.

        Returns:
            Dict with setup_type, stop_multiplier, arm, expected_reward, confidence.
        """
        if self.total_pulls == 0:
            setup, stop = self.decode_arm(0)
            return {
                "setup_type":       setup,
                "stop_multiplier":  stop,
                "arm":              0,
                "expected_reward":  0.0,
                "confidence":       "LOW",
            }

        theta_scores = np.array([
            np.linalg.inv(self.A[i]) @ self.b[i] @ np.ones(self.context_dim)
            for i in range(self.n_arms)
        ])
        best_arm    = int(np.argmax(theta_scores))
        setup, stop = self.decode_arm(best_arm)
        pulls       = self.arm_pulls[best_arm]
        confidence  = "HIGH" if pulls >= 20 else "MEDIUM" if pulls >= 5 else "LOW"
        return {
            "setup_type":      setup,
            "stop_multiplier": stop,
            "arm":             best_arm,
            "expected_reward": float(theta_scores[best_arm]),
            "confidence":      confidence,
        }

    def get_state(self) -> dict:
        """Serialise bandit state for persistence (A/b matrices excluded for size)."""
        return {
            "total_pulls":  self.total_pulls,
            "arm_pulls":    self.arm_pulls.tolist(),
            "arm_rewards":  self.arm_rewards.tolist(),
            "win_rates":    self.win_rates.tolist(),
        }

    def load_state(self, state: dict) -> None:
        """Restore bandit counters from a persisted state dict."""
        self.total_pulls  = state.get("total_pulls", 0)
        self.arm_pulls    = np.array(state.get("arm_pulls",   [0] * self.n_arms), dtype=int)
        self.arm_rewards  = np.array(state.get("arm_rewards", [0.0] * self.n_arms), dtype=float)
        self.win_rates    = np.array(state.get("win_rates",   [0.0] * self.n_arms), dtype=float)
