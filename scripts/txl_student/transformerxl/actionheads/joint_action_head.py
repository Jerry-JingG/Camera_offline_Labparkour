from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
from torch.distributions import Normal


class JointPoseActionHead(nn.Module):
    """Gaussian policy head that outputs bounded joint pose deltas."""

    def __init__(
        self,
        d_model: int,
        action_dim: int = 12,
        hidden_dims: Sequence[int] = (256, 256),
        tanh_output: bool = True,
        action_scale: float = 0.5,
    ) -> None:
        super().__init__()
        layers = []
        in_features = d_model
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, action_dim))
        self.mu_head = nn.Sequential(*layers)
        self.log_std = nn.Parameter(torch.zeros(action_dim))
        self.tanh_output = tanh_output
        self.action_scale = action_scale

    def forward_step(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Gaussian policy parameters for single time-step features.

        Args:
            h: Tensor[B, d_model]
        """
        raw_mu = self.mu_head(h)
        mean = torch.tanh(raw_mu) * self.action_scale if self.tanh_output else raw_mu
        log_std = self.log_std.expand_as(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        return {"mean": mean, "log_std": log_std, "std": std, "dist": dist}

    def forward_sequence(self, h_seq: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Gaussian policy parameters for every step of a sequence.

        Args:
            h_seq: Tensor[B, S, d_model]
        """
        bsz, seq_len, feat_dim = h_seq.shape
        raw_mu = self.mu_head(h_seq.reshape(bsz * seq_len, feat_dim))
        mean_flat = torch.tanh(raw_mu) * self.action_scale if self.tanh_output else raw_mu
        mean = mean_flat.reshape(bsz, seq_len, -1)
        log_std = self.log_std.expand_as(mean)
        std = log_std.exp()
        dist = Normal(mean, std)
        return {"mean": mean, "log_std": log_std, "std": std, "dist": dist}

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Dispatch to step-wise or sequence inference depending on input rank."""
        if h.dim() == 2:
            return self.forward_step(h)
        if h.dim() == 3:
            return self.forward_sequence(h)
        raise ValueError("Input tensor must have rank 2 or 3.")
