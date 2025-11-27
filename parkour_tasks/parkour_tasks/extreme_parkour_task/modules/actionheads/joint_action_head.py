from __future__ import annotations

from typing import Dict, Sequence

import torch
from torch import nn
from torch.distributions import Normal


class JointPoseActionHead(nn.Module):
    """Gaussian policy head that outputs joint pose deltas.

    By default the head is purely linear (no tanh), producing unbounded actions that match
    the teacher policy outputs stored in ``action_teacher``.
    """

    def __init__(
        self,
        d_model: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (256, 256),
        tanh_output: bool = False,
        action_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
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

    def _compute_mean(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the final linear head and optional tanh scaling."""
        raw_mu = self.mu_head(x)
        if self.tanh_output:
            return torch.tanh(raw_mu) * self.action_scale
        return raw_mu

    def forward_step(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute Gaussian policy parameters for single time-step features.

        Args:
            h: Tensor[B, d_model]
        """
        mean = self._compute_mean(h)
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
        mean_flat = self._compute_mean(h_seq.reshape(bsz * seq_len, feat_dim))
        mean = mean_flat.reshape(bsz, seq_len, self.action_dim)
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
