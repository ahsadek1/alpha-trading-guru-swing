"""
ATG Phase 4 — Causal Discovery (NOTEARS) for Swing Trading
Mirrors ATM's causal_discovery exactly.
Learns causal structure between context features and swing setup returns.
"""
import numpy as np
import logging
from src.enhancements.distributional_rl import DistributionalSwingRL
from src.database import get_connection
from config.settings import CONTEXT_DIM, NUM_BANDIT_ARMS

log = logging.getLogger(__name__)


def _notears_linear(X: np.ndarray, lambda1: float = 0.1, max_iter: int = 100) -> np.ndarray:
    """
    Simplified NOTEARS linear DAG learning.
    Returns adjacency matrix W where W[i,j] = causal effect of feature i on j.
    """
    n, d = X.shape
    W     = np.zeros((d, d))
    X_std = (X - X.mean(0)) / (X.std(0) + 1e-8)

    for _ in range(max_iter):
        grad = -2/n * X_std.T @ (X_std @ W - X_std) + lambda1 * np.sign(W)
        W   -= 0.01 * grad
        np.fill_diagonal(W, 0)  # no self-loops

    return W


class CausalSwingDiscovery(DistributionalSwingRL):
    """
    Phase 4: Learn causal graph between market features and swing returns.
    Uses causal structure to prune irrelevant context dims per setup type.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)
        self.causal_weights = np.ones((NUM_BANDIT_ARMS, CONTEXT_DIM))  # uniform start
        self._outcomes_buffer: list = []
        self._causal_update_freq = 50  # recompute every 50 trades
        log.info("CausalSwingDiscovery initialized")

    @classmethod
    def from_distributional(cls, dist_bandit: DistributionalSwingRL) -> "CausalSwingDiscovery":
        causal = cls()
        causal.arm_pulls   = dist_bandit.arm_pulls.copy()
        causal.arm_rewards = dist_bandit.arm_rewards.copy()
        causal.win_rates   = dist_bandit.win_rates.copy()
        causal.total_pulls = dist_bandit.total_pulls
        log.info("CausalSwingDiscovery promoted from distributional | pulls=%d", causal.total_pulls)
        # Bootstrap causal weights from historical DB
        causal._rebuild_causal_graph()
        return causal

    def select_arm(self, context: np.ndarray) -> int:
        # Weight context by causal relevance before selection
        if self.total_pulls > 0:
            # Use mean causal weights across arms
            mean_weights = self.causal_weights.mean(axis=0)
            mean_weights = mean_weights / (mean_weights.sum() + 1e-9) * CONTEXT_DIM
            context = context * mean_weights
        return super().select_arm(context)

    def update(self, arm: int, context: np.ndarray, reward: float):
        super().update(arm, context, reward)
        self._outcomes_buffer.append((arm, context.copy(), reward))
        if len(self._outcomes_buffer) >= self._causal_update_freq:
            self._rebuild_causal_graph()
            self._outcomes_buffer = []

    def _rebuild_causal_graph(self):
        """Rebuild causal weights from trade history."""
        try:
            conn = get_connection()
            rows = conn.execute("""
                SELECT context_vector, reward, arm_index
                FROM bandit_outcomes
                WHERE reward IS NOT NULL
                ORDER BY id DESC LIMIT 500
            """).fetchall()
            conn.close()

            if len(rows) < 30:
                return

            import ast
            contexts = []
            rewards  = []
            for r in rows:
                try:
                    ctx = np.array(ast.literal_eval(r["context_vector"]), dtype=float)
                    if len(ctx) == CONTEXT_DIM:
                        contexts.append(ctx)
                        rewards.append(float(r["reward"]))
                except Exception:
                    continue

            if len(contexts) < 20:
                return

            X = np.column_stack([np.array(contexts), np.array(rewards)])
            W = _notears_linear(X, lambda1=0.05)

            # Feature importance = sum of absolute causal effects on reward (last col)
            feat_importance = np.abs(W[:CONTEXT_DIM, -1])
            feat_importance = feat_importance / (feat_importance.sum() + 1e-9) * CONTEXT_DIM

            # Apply per-arm (uniform for now — can be refined per setup type)
            for arm in range(NUM_BANDIT_ARMS):
                self.causal_weights[arm] = feat_importance

            log.info("Causal graph rebuilt | top features: %s",
                     np.argsort(feat_importance)[-5:][::-1].tolist())

        except Exception as e:
            log.warning("Causal graph rebuild failed: %s", e)
