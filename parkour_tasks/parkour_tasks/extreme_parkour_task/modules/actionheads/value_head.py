"""ValueHead module for PPO value function estimation.

This module estimates state values from fused temporal features.
Architecture: Linear(d_model, 256) + ReLU -> Linear(256, 256) + ReLU -> Linear(256, 1)
"""
from __future__ import annotations

from typing import Sequence

import torch
from torch import nn, Tensor


class ValueHead(nn.Module):
    """Value function head for PPO.

    Estimates state values from temporal features using a simple MLP.

    Args:
        d_model: Input feature dimension from temporal encoder.
        hidden_dims: Tuple of hidden layer dimensions. Default: (256, 256).
    """

    def __init__(
        self,
        d_model: int,
        hidden_dims: Sequence[int] = (256, 256),
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if not hidden_dims:
            raise ValueError("hidden_dims must not be empty")

        layers = []
        in_features = d_model
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim
        layers.append(nn.Linear(in_features, 1))

        self.mlp = nn.Sequential(*layers)

    def forward_step(self, h: Tensor) -> Tensor:
        """Single timestep value estimation.

        Args:
            h: Tensor of shape [B, d_model]

        Returns:
            values: Tensor of shape [B, 1]
        """
        return self.mlp(h)

    def forward_sequence(self, h_seq: Tensor) -> Tensor:
        """Sequence value estimation.

        Args:
            h_seq: Tensor of shape [B, S, d_model]

        Returns:
            values: Tensor of shape [B, S, 1]
        """
        bsz, seq_len, feat_dim = h_seq.shape
        # Flatten batch and sequence dimensions for MLP
        h_flat = h_seq.reshape(bsz * seq_len, feat_dim)
        values_flat = self.mlp(h_flat)
        # Reshape back to [B, S, 1]
        return values_flat.reshape(bsz, seq_len, 1)

    def forward(self, h: Tensor) -> Tensor:
        """Dispatch to step-wise or sequence inference depending on input rank.

        Args:
            h: Tensor of shape [B, d_model] or [B, S, d_model]

        Returns:
            values: Tensor of shape [B, 1] or [B, S, 1]

        Raises:
            ValueError: If input tensor does not have rank 2 or 3.
        """
        if h.dim() == 2:
            return self.forward_step(h)
        if h.dim() == 3:
            return self.forward_sequence(h)
        raise ValueError("Input tensor must have rank 2 or 3.")
