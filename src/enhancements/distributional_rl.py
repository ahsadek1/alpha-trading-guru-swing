"""
ATG Phase 3 — Distributional RL (C51) for Swing Trading
Mirrors ATM's distributional_rl exactly.
Models full return distribution (not just expected value).
"""
import numpy as np
import logging
from src.enhancements.neural_bandit import NeuralSwingBandit
from config.settings import NUM_BANDIT_ARMS, CONTEXT_DIM

log = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

N_ATOMS  = 51
V_MIN, V_MAX = -0.20, 0.30   # swing return range: -20% to +30%


class C51SwingNet(nn.Module if TORCH_AVAILABLE else object):
    def __init__(self, context_dim: int, n_arms: int, n_atoms: int):
        if TORCH_AVAILABLE:
            super().__init__()
            self.n_arms  = n_arms
            self.n_atoms = n_atoms
            self.net = nn.Sequential(
                nn.Linear(context_dim, 256), nn.ReLU(),
                nn.Linear(256, 128),         nn.ReLU(),
                nn.Linear(128, n_arms * n_atoms),
            )

    def forward(self, x):
        logits = self.net(x).view(-1, self.n_arms, self.n_atoms)
        return torch.softmax(logits, dim=-1)


class DistributionalSwingRL(NeuralSwingBandit):
    """
    Phase 3: C51 distributional RL.
    Models the full P&L distribution for each swing setup type.
    """

    def __init__(self, alpha: float = 1.0):
        super().__init__(alpha)
        if TORCH_AVAILABLE:
            self.c51_net   = C51SwingNet(CONTEXT_DIM, NUM_BANDIT_ARMS, N_ATOMS)
            self.optimizer = torch.optim.Adam(self.c51_net.parameters(), lr=5e-4)
            self.atoms     = torch.linspace(V_MIN, V_MAX, N_ATOMS)
            self._use_c51  = True
        else:
            self._use_c51 = False
        log.info("DistributionalSwingRL initialized")

    @classmethod
    def from_neural(cls, neural_bandit: NeuralSwingBandit) -> "DistributionalSwingRL":
        dist = cls()
        dist.arm_pulls   = neural_bandit.arm_pulls.copy()
        dist.arm_rewards = neural_bandit.arm_rewards.copy()
        dist.win_rates   = neural_bandit.win_rates.copy()
        dist.total_pulls = neural_bandit.total_pulls
        log.info("DistributionalSwingRL promoted from neural | pulls=%d", dist.total_pulls)
        return dist

    def select_arm(self, context: np.ndarray) -> int:
        if not self._use_c51:
            return super().select_arm(context)
        self.c51_net.eval()
        with torch.no_grad():
            x    = torch.FloatTensor(context).unsqueeze(0)
            dist = self.c51_net(x).squeeze(0)  # (n_arms, n_atoms)
            # Expected value of each arm's return distribution
            ev   = (dist * self.atoms).sum(dim=-1).numpy()
        ucb = self.alpha * np.sqrt(1.0 / (self.arm_pulls + 1))
        return int(np.argmax(ev + ucb))

    def update(self, arm: int, context: np.ndarray, reward: float):
        super().update(arm, context, reward)
        if not self._use_c51:
            return
        # Project reward onto atom space
        clipped = np.clip(reward, V_MIN, V_MAX)
        atom_idx = int((clipped - V_MIN) / (V_MAX - V_MIN) * (N_ATOMS - 1))
        target = torch.zeros(NUM_BANDIT_ARMS, N_ATOMS)
        target[arm, atom_idx] = 1.0
        x    = torch.FloatTensor(context).unsqueeze(0)
        pred = self.c51_net(x).squeeze(0)
        loss = -(target * torch.log(pred + 1e-8)).sum()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
