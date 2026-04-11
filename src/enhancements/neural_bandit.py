"""
ATG Phase 2 — Neural Swing Bandit
Mirrors ATM's NeuralBandit, adapted for swing setup selection.
"""
import numpy as np
import logging
from src.bandit import AutonomousSwingBandit
from config.settings import NUM_BANDIT_ARMS, CONTEXT_DIM

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class SwingNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, context_dim: int, n_arms: int):
        if TORCH_AVAILABLE:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(context_dim, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64),          nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, n_arms),
            )

    def forward(self, x):
        return self.net(x)


class NeuralSwingBandit(AutonomousSwingBandit):
    """
    Phase 2: Neural network replaces LinUCB for arm selection.
    Falls back to linear bandit if torch unavailable.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)
        if TORCH_AVAILABLE:
            self.net       = SwingNet(CONTEXT_DIM, NUM_BANDIT_ARMS)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=1e-3)
            self.loss_fn   = nn.MSELoss()
            self._use_net  = True
        else:
            log.warning("PyTorch not available — NeuralSwingBandit falls back to linear")
            self._use_net = False
        log.info("NeuralSwingBandit initialized | torch=%s", TORCH_AVAILABLE)

    @classmethod
    def from_linear(cls, linear_bandit: AutonomousSwingBandit) -> "NeuralSwingBandit":
        """Upgrade from linear bandit, preserving pull history."""
        neural = cls()
        neural.arm_pulls   = linear_bandit.arm_pulls.copy()
        neural.arm_rewards = linear_bandit.arm_rewards.copy()
        neural.win_rates   = linear_bandit.win_rates.copy()
        neural.total_pulls = linear_bandit.total_pulls
        log.info("NeuralSwingBandit promoted from linear | pulls=%d", neural.total_pulls)
        return neural

    def select_arm(self, context: np.ndarray) -> int:
        if not self._use_net:
            return super().select_arm(context)
        self.net.eval()
        with torch.no_grad():
            x      = torch.FloatTensor(context).unsqueeze(0)
            scores = self.net(x).squeeze().numpy()
        # UCB exploration bonus
        ucb_bonus = self.alpha * np.sqrt(1.0 / (self.arm_pulls + 1))
        return int(np.argmax(scores + ucb_bonus))

    def update(self, arm: int, context: np.ndarray, reward: float):
        super().update(arm, context, reward)
        if not self._use_net:
            return
        self.net.train()
        x      = torch.FloatTensor(context).unsqueeze(0)
        target = torch.zeros(NUM_BANDIT_ARMS)
        with torch.no_grad():
            target = self.net(x).squeeze().clone()
        target[arm] = float(reward)
        pred   = self.net(x).squeeze()
        loss   = self.loss_fn(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
