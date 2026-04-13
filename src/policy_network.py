"""
policy_network.py
=================
Paper-style 4-block policy network:
  - Main / Score / Gate / Memory
with variant toggles matching Table 1.

Important:
  - Output is stock-only post-rebalance target weights (N,)
  - Sum is <= 1 (cash is implicit and handled by simulator)
"""

import torch.nn as nn
import torch.nn.functional as F
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 변형별 플래그 매핑 (논문 Table 1)
# ─────────────────────────────────────────────────────────────────────────────

VARIANT_FLAGS: dict[str, dict] = {
    "NN-ST": dict(use_main=False, use_score=True, use_gate=False, use_memory=True),
    "NN-IR": dict(use_main=True, use_score=False, use_gate=False, use_memory=True),
    "NN-ISR": dict(use_main=True, use_score=False, use_gate=True, use_memory=True),
    "NN-All": dict(use_main=True, use_score=True, use_gate=True, use_memory=True),
}


# ─────────────────────────────────────────────────────────────────────────────
# (A) Main Block
# ─────────────────────────────────────────────────────────────────────────────

class MainBlock(nn.Module):
    """Eq.(7)-style Main block."""

    def __init__(self, n_stocks: int, hidden_dim: int = 16):
        super().__init__()
        self.n_stocks = n_stocks
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.theta_bull = nn.Parameter(torch.zeros(n_stocks + 1))
        self.theta_bear = nn.Parameter(torch.zeros(n_stocks + 1))

    def forward(self, index_regime_prob: torch.Tensor) -> torch.Tensor:
        omega_bull = torch.sigmoid(self.mlp(index_regime_prob))  # (B,1)
        w_bull_full = F.softmax(self.theta_bull, dim=0)          # (N+1,)
        w_bear_full = F.softmax(self.theta_bear, dim=0)          # (N+1,)
        w_bull = w_bull_full[1:]
        w_bear = w_bear_full[1:]
        return omega_bull * w_bull + (1.0 - omega_bull) * w_bear


# ─────────────────────────────────────────────────────────────────────────────
# (B) Score Block
# ─────────────────────────────────────────────────────────────────────────────

class ScoreBlock(nn.Module):
    """Eq.(8)-style shared per-asset scorer."""

    def __init__(self, in_features: int = 5, hidden_dim: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, features_with_cash: torch.Tensor) -> torch.Tensor:
        # features_with_cash: (B, N+1, 5)
        scores = self.net(features_with_cash).squeeze(-1)   # (B, N+1)
        w_full = F.softmax(scores, dim=-1)                  # (B, N+1)
        return w_full[:, 1:]                                # drop cash


# ─────────────────────────────────────────────────────────────────────────────
# (C) Gate Block
# ─────────────────────────────────────────────────────────────────────────────

class GateBlock(nn.Module):
    """Eq.(9)-style shared gate network."""

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, stock_regime_probs: torch.Tensor) -> torch.Tensor:
        x = stock_regime_probs.unsqueeze(-1)  # (B,N,1)
        return torch.sigmoid(self.net(x).squeeze(-1))


# ─────────────────────────────────────────────────────────────────────────────
# (D) Memory Block
# ─────────────────────────────────────────────────────────────────────────────

class MemoryBlock(nn.Module):
    """Memory gate from turnover scalar."""

    def __init__(self, hidden_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, turnover: torch.Tensor) -> torch.Tensor:
        # turnover: (B,1)
        return torch.sigmoid(self.net(turnover))


# ─────────────────────────────────────────────────────────────────────────────
# Full Policy Network (variant-aware)
# ─────────────────────────────────────────────────────────────────────────────

class PolicyNetwork(nn.Module):
    """Variant-aware 4-block paper policy."""

    def __init__(
        self,
        n_stocks: int,
        main_hidden: int = 16,
        score_hidden: int = 32,
        use_main: bool = True,
        use_score: bool = True,
        use_gate: bool = True,
        use_memory: bool = True,
    ):
        super().__init__()
        self.n_stocks         = n_stocks
        self.use_main         = use_main
        self.use_score        = use_score
        self.use_gate         = use_gate
        self.use_memory       = use_memory
        self.memory_enabled_override: bool | None = None

        if use_main:
            self.main_block  = MainBlock(n_stocks, hidden_dim=main_hidden)
            self.theta_omega1 = nn.Parameter(torch.tensor(0.0))

        if use_score:
            self.score_block = ScoreBlock(in_features=5, hidden_dim=score_hidden)
            if not use_main:
                self.theta_omega1 = nn.Parameter(torch.tensor(0.0))

        if use_gate:
            self.gate_block  = GateBlock()

        if use_memory:
            self.memory_block = MemoryBlock()

    @classmethod
    def from_variant(
        cls,
        variant: str,
        n_stocks: int,
        main_hidden: int = 16,
        score_hidden: int = 32,
    ) -> "PolicyNetwork":
        """
        논문 Table 1 변형명으로 PolicyNetwork를 생성합니다.

        Parameters
        ----------
        variant   : NN-ST | NN-IR | NN-ISR | NN-All
        n_stocks  : 종목 수
        """
        if variant not in VARIANT_FLAGS:
            raise ValueError(
                f"알 수 없는 변형: '{variant}'. 선택 가능: {list(VARIANT_FLAGS.keys())}"
            )
        flags = VARIANT_FLAGS[variant]
        return cls(
            n_stocks=n_stocks,
            main_hidden=main_hidden,
            score_hidden=score_hidden,
            **flags,
        )

    def forward(
        self,
        index_regime_prob: torch.Tensor,
        stock_features: torch.Tensor,
        stock_regime_probs: torch.Tensor,
        prev_weights: torch.Tensor,
        use_memory_override: bool | None = None,
    ) -> torch.Tensor:
        # index_regime_prob: (B,1), stock_features: (B,N,5)
        if index_regime_prob.dim() == 1:
            index_regime_prob = index_regime_prob.unsqueeze(-1)
        B, N = prev_weights.shape
        use_memory = self.use_memory if use_memory_override is None else use_memory_override
        if self.memory_enabled_override is not None:
            use_memory = self.memory_enabled_override

        if self.use_main:
            w1 = self.main_block(index_regime_prob)
        else:
            w1 = None

        if self.use_score:
            # cash pseudo-asset features: [0,0,0,mu_I,sigma_I]
            idx_mu = stock_features[:, :1, 3:4]
            idx_vol = stock_features[:, :1, 4:5]
            zeros = torch.zeros_like(idx_mu)
            cash_feat = torch.cat([zeros, zeros, zeros, idx_mu, idx_vol], dim=-1)  # (B,1,5)
            feat_with_cash = torch.cat([cash_feat, stock_features], dim=1)          # (B,N+1,5)
            w_sc = self.score_block(feat_with_cash)
        else:
            w_sc = None

        if w1 is not None and w_sc is not None:
            omega1 = torch.sigmoid(self.theta_omega1)
            w2 = (1.0 - omega1) * w_sc + omega1 * w1
        elif w1 is not None:
            w2 = w1
        elif w_sc is not None:
            w2 = w_sc
        else:
            w2 = torch.zeros(B, N, dtype=prev_weights.dtype, device=prev_weights.device)

        if self.use_gate:
            gate = self.gate_block(stock_regime_probs)
            w3 = gate * w2
        else:
            w3 = w2

        if use_memory and self.use_memory:
            turnover = torch.sum(torch.abs(w3 - prev_weights), dim=-1, keepdim=True)
            omega_p = self.memory_block(turnover)  # (B,1)
            w_final = (1.0 - omega_p) * w3 + omega_p * prev_weights
        else:
            w_final = w3

        return torch.clamp(w_final, min=0.0)

    def predict(
        self,
        index_regime_prob: float,
        stock_features_np,
        stock_regime_probs_np,
        prev_weights_np,
        device: str = "cpu",
    ):
        """
        Single-step inference wrapper.
        """
        import numpy as np
        self.eval()
        with torch.no_grad():
            idx_t  = torch.tensor([[index_regime_prob]], dtype=torch.float32).to(device)
            stk_t  = torch.tensor(stock_features_np[None], dtype=torch.float32).to(device)
            reg_t  = torch.tensor(stock_regime_probs_np[None], dtype=torch.float32).to(device)
            prev_t = torch.tensor(prev_weights_np[None], dtype=torch.float32).to(device)

            w = self.forward(idx_t, stk_t, reg_t, prev_t)
        return w.squeeze(0).cpu().numpy()
