"""
ATG Bandit v3.1 — AutonomousSwingBandit (LinUCB contextual bandit).

FIX [F2]:  A_i and b_i matrices now fully persisted via get_state()/load_state().
           Intelligence survives Railway restarts.
FIX [F9]:  Added forgetting factor (λ=0.99) in update() to prevent over-confidence.
           Bandit can re-explore after regime shifts.
FIX [F18]: Added dimension + version validation in load_state().
           Incompatible state halts instead of silently corrupting posterior.

Arms encode (setup_type × stop_multiplier) pairs.
"""
import io
import base64
import numpy as np
import logging
from typing import Tuple, Optional

from config.settings import (
    NUM_BANDIT_ARMS,
    NUM_SETUP_TYPES,
    NUM_STOP_MULTIPLIERS,
    SETUP_TYPES,
    STOP_MULTIPLIERS,
    CONTEXT_DIM,
)

log = logging.getLogger(__name__)

# FIX [F9]: Forgetting factor — decays old information to allow regime adaptation.
# λ=1.0 = no forgetting (original behavior, leads to over-confidence)
# λ=0.99 = gentle forgetting (allows exploration after ~100 trades on same arm)
FORGETTING_FACTOR: float = 0.99

# Version string — bump this when CONTEXT_DIM or feature order changes.
# FIX [F2]: persisted in state for compatibility validation.
BANDIT_STATE_VERSION: str = "v3.1"


class AutonomousSwingBandit:
    """
    Linear contextual bandit for swing setup + stop multiplier selection.

    Arms    : NUM_SETUP_TYPES × NUM_STOP_MULTIPLIERS  (6 × 11 = 66)
    Context : CONTEXT_DIM-dimensional market feature vector
    Reward  : trade P&L % scaled to [-1, +1]

    Algorithm : LinUCB with Thompson Sampling exploration.
    FIX [F2]  : A/b matrices persisted to SQLite and restored on restart.
    FIX [F9]  : Forgetting factor prevents posterior over-confidence.
    """

    def __init__(self, alpha: float = 1.0) -> None:
        self.n_arms      = NUM_BANDIT_ARMS
        self.alpha       = alpha
        self.context_dim = CONTEXT_DIM

        # LinUCB: A_i ∈ R^{d×d}, b_i ∈ R^d
        self.A = [np.eye(CONTEXT_DIM, dtype=np.float64) * alpha for _ in range(self.n_arms)]
        self.b = [np.zeros(CONTEXT_DIM, dtype=np.float64)       for _ in range(self.n_arms)]

        self.total_pulls  = 0
        self.arm_pulls    = np.zeros(self.n_arms, dtype=int)
        self.arm_rewards  = np.zeros(self.n_arms, dtype=float)
        self.win_rates    = np.zeros(self.n_arms, dtype=float)

        self._load_warmstart_priors()

        log.info(
            "AutonomousSwingBandit v3.1 initialized | arms=%d | λ=%.2f | version=%s",
            self.n_arms, FORGETTING_FACTOR, BANDIT_STATE_VERSION,
        )

    def _load_warmstart_priors(self) -> None:
        """Load warm-start biases from data/bandit_priors.json."""
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
            eye          = np.eye(self.context_dim, dtype=np.float64)
            ones         = np.ones(self.context_dim, dtype=np.float64)
            applied      = 0
            for arm_str, cfg in arm_cfg.items():
                arm_idx = int(arm_str)
                if arm_idx >= self.n_arms:
                    continue
                bias = float(cfg.get("bias", 0.0))
                self.A[arm_idx] = self.alpha * eye + prior_trades * eye
                self.b[arm_idx] = bias * prior_trades * ones
                applied += 1
            log.info("Bandit warm-start: %d/%d arms primed (prior_trades=%d)", applied, self.n_arms, prior_trades)
        except FileNotFoundError:
            log.debug("No bandit_priors.json found — using uniform prior")
        except Exception as e:
            log.warning("Bandit warm-start failed (non-fatal): %s", e)

    def select_arm(self, context: np.ndarray) -> int:
        """LinUCB arm selection via Thompson Sampling."""
        x = np.asarray(context, dtype=np.float64)
        assert x.shape == (self.context_dim,), (
            f"Context must be {self.context_dim}-dim, got {x.shape}"
        )
        ucb_scores = np.zeros(self.n_arms)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            for i in range(self.n_arms):
                # FIX [F39]: np.linalg.solve() more numerically stable than inv() for large A
                try:
                    theta    = np.linalg.solve(self.A[i], self.b[i])
                    A_inv_x  = np.linalg.solve(self.A[i], x)
                    variance = max(float(x @ A_inv_x), 0.0)
                except np.linalg.LinAlgError:
                    # A_i is singular — use warm priors (identity)
                    theta    = self.b[i]
                    variance = 1.0
                ucb_scores[i] = float(np.dot(theta, x)) + self.alpha * np.sqrt(variance)
        ucb_scores = np.nan_to_num(ucb_scores, nan=0.0, posinf=0.0, neginf=0.0)
        return int(np.argmax(ucb_scores))

    def update(self, arm: int, context: np.ndarray, reward: float,
               exit_reason_weight: float = 1.0) -> None:
        """
        Update LinUCB parameters after observing a reward.

        FIX [F9]: Forgetting factor (FORGETTING_FACTOR) applied to A and b
                  before outer product update. Prevents posterior over-confidence
                  and allows re-exploration after market regime shifts.

        Args:
            arm                 : arm index that was pulled.
            context             : context vector used at selection time.
            reward              : scaled reward in [-1, +1].
            exit_reason_weight  : INV-2 weight (0.0=skip update, 0.5=partial, 1.0=full).
        """
        if exit_reason_weight == 0.0:
            log.debug("Bandit update skipped (exit_reason_weight=0.0, arm=%d)", arm)
            return

        x = np.asarray(context, dtype=np.float64)
        weighted_reward = reward * exit_reason_weight

        # FIX [F9]: Apply forgetting factor before update
        self.A[arm] = FORGETTING_FACTOR * self.A[arm] + np.outer(x, x)
        self.b[arm] = FORGETTING_FACTOR * self.b[arm] + weighted_reward * x

        self.arm_pulls[arm]   += 1
        self.arm_rewards[arm] += weighted_reward
        self.total_pulls      += 1

        n                    = self.arm_pulls[arm]
        prev                 = self.win_rates[arm]
        self.win_rates[arm]  = prev + (1.0 if reward > 0 else 0.0 - prev) / n

        log.debug(
            "Bandit update | arm=%d setup=%s stop=%.2fx reward=%.4f weight=%.1f pulls=%d",
            arm, *self.decode_arm(arm), reward, exit_reason_weight, n,
        )

    def decode_arm(self, arm: int) -> Tuple[str, float]:
        """Convert arm index → (setup_type, stop_multiplier)."""
        setup_idx = arm // NUM_STOP_MULTIPLIERS
        stop_idx  = arm %  NUM_STOP_MULTIPLIERS
        return SETUP_TYPES[setup_idx], round(float(STOP_MULTIPLIERS[stop_idx]), 2)

    def best_setup(self, context: Optional[np.ndarray] = None) -> dict:
        """
        Return the arm with highest expected reward.
        FIX: Uses provided context vector if given (not unit vector).
        Falls back to uniform context if none provided.
        """
        if self.total_pulls == 0:
            setup, stop = self.decode_arm(0)
            return {"setup_type": setup, "stop_multiplier": stop, "arm": 0,
                    "expected_reward": 0.0, "confidence": "LOW"}

        ctx = np.asarray(context, dtype=np.float64) if context is not None \
              else np.ones(self.context_dim, dtype=np.float64)

        theta_scores = np.array([
            float(np.linalg.solve(self.A[i], self.b[i]) @ ctx)  # FIX [F39]
            for i in range(self.n_arms)
        ])
        best_arm    = int(np.argmax(theta_scores))
        setup, stop = self.decode_arm(best_arm)
        pulls       = self.arm_pulls[best_arm]
        return {
            "setup_type":      setup,
            "stop_multiplier": stop,
            "arm":             best_arm,
            "expected_reward": float(theta_scores[best_arm]),
            "confidence":      "HIGH" if pulls >= 20 else "MEDIUM" if pulls >= 5 else "LOW",
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def _matrix_to_b64(self, arr: np.ndarray) -> str:
        """Serialize numpy array → base64 string for SQLite storage."""
        buf = io.BytesIO()
        np.save(buf, arr)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def _b64_to_matrix(self, b64str: str) -> np.ndarray:
        """Deserialize base64 string → numpy array."""
        buf = io.BytesIO(base64.b64decode(b64str))
        return np.load(buf, allow_pickle=False)

    def get_state(self) -> dict:
        """
        Serialise full bandit state for persistence.

        FIX [F2]: Now includes A and b matrices as base64-encoded numpy binary.
        FIX [F18]: Includes context_version and dimension for compatibility checks.

        Prior behavior only saved scalar counters — intelligence was lost on restart.
        """
        return {
            # Metadata for compatibility validation [F18]
            "state_version":  BANDIT_STATE_VERSION,
            "context_dim":    self.context_dim,
            "n_arms":         self.n_arms,
            "alpha":          self.alpha,
            # Scalar counters (always included)
            "total_pulls":    self.total_pulls,
            "arm_pulls":      self.arm_pulls.tolist(),
            "arm_rewards":    self.arm_rewards.tolist(),
            "win_rates":      self.win_rates.tolist(),
            # FIX [F2]: Full A and b matrices (LinUCB posterior)
            "A_matrices":     [self._matrix_to_b64(a) for a in self.A],
            "b_vectors":      [self._matrix_to_b64(b) for b in self.b],
        }

    def load_state(self, state: dict) -> None:
        """
        Restore bandit state from a persisted state dict.

        FIX [F2]:  Restores A and b matrices — full posterior survives restart.
        FIX [F18]: Validates context_dim and n_arms before loading.
                   Raises ValueError on mismatch to prevent silent corruption.
        """
        # [F18] Dimension + version validation
        saved_dim   = state.get("context_dim", self.context_dim)
        saved_arms  = state.get("n_arms",      self.n_arms)
        saved_ver   = state.get("state_version", "unknown")

        if saved_dim != self.context_dim:
            raise ValueError(
                f"Bandit state dimension mismatch: saved={saved_dim}, current={self.context_dim}. "
                f"Reset bandit state or bump CONTEXT_VERSION."
            )
        if saved_arms != self.n_arms:
            raise ValueError(
                f"Bandit arm count mismatch: saved={saved_arms}, current={self.n_arms}. "
                f"Reset bandit state."
            )
        if saved_ver != BANDIT_STATE_VERSION:
            log.warning(
                "Bandit state version mismatch: saved=%s, current=%s — loading anyway (minor version)",
                saved_ver, BANDIT_STATE_VERSION,
            )

        # Restore counters
        self.total_pulls  = state.get("total_pulls", 0)
        self.arm_pulls    = np.array(state.get("arm_pulls",   [0]   * self.n_arms), dtype=int)
        self.arm_rewards  = np.array(state.get("arm_rewards", [0.0] * self.n_arms), dtype=float)
        self.win_rates    = np.array(state.get("win_rates",   [0.0] * self.n_arms), dtype=float)

        # FIX [F2]: Restore A and b matrices if present
        if "A_matrices" in state and "b_vectors" in state:
            try:
                loaded_A = [self._b64_to_matrix(s) for s in state["A_matrices"]]
                loaded_b = [self._b64_to_matrix(s) for s in state["b_vectors"]]
                # Validate restored matrix shapes
                for i, (a, b) in enumerate(zip(loaded_A, loaded_b)):
                    if a.shape != (self.context_dim, self.context_dim):
                        raise ValueError(f"A[{i}] shape mismatch: {a.shape}")
                    if b.shape != (self.context_dim,):
                        raise ValueError(f"b[{i}] shape mismatch: {b.shape}")
                self.A = loaded_A
                self.b = loaded_b
                log.info(
                    "Bandit v3.1 restored: %d pulls, %d arms, A/b matrices loaded (intelligence preserved)",
                    self.total_pulls, self.n_arms,
                )
            except Exception as e:
                log.warning(
                    "A/b matrix restore failed (%s) — using warm-start priors (counters preserved)",
                    e,
                )
        else:
            log.warning(
                "Bandit state has no A/b matrices (old format) — "
                "intelligence not restored, warm-start priors active"
            )
