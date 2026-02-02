"""StudentActorCritic wrapper for PPO training with MultiModalStudentPolicy.

This module wraps a pre-trained MultiModalStudentPolicy with a value head
for PPO-based reinforcement learning fine-tuning.

Architecture:
    - Composition over inheritance: wraps MultiModalStudentPolicy
    - Shared feature extraction: value head uses temporal features
    - Gaussian policy: Normal distribution for continuous actions
    - Learnable std: log-parameterized standard deviation
"""
from __future__ import annotations

from typing import List, Optional, Tuple, Sequence

import torch
from torch import nn, Tensor
from torch.distributions import Normal


class StudentActorCritic(nn.Module):
    """Actor-Critic wrapper for MultiModalStudentPolicy.

    Wraps a pre-trained student policy with a value head for PPO training.
    Supports encoder freezing for fine-tuning strategies.

    Args:
        student_policy: Pre-trained MultiModalStudentPolicy to wrap.
        value_hidden_dims: Hidden layer dimensions for value head.
        init_noise_std: Initial standard deviation for action noise.
        freeze_encoders: If True, freeze proprio and depth encoders.
        freeze_fusion: If True, freeze fusion transformer.
        freeze_temporal: If True, freeze temporal model.
    """

    def __init__(
        self,
        student_policy: nn.Module,
        value_hidden_dims: Tuple[int, ...] = (256, 256),
        init_noise_std: float = 1.0,
        freeze_encoders: bool = False,
        freeze_fusion: bool = False,
        freeze_temporal: bool = False,
    ) -> None:
        super().__init__()

        # Input validation
        if init_noise_std <= 0:
            raise ValueError(f"init_noise_std must be positive, got {init_noise_std}")
        if not value_hidden_dims:
            raise ValueError("value_hidden_dims must not be empty")

        self.student_policy = student_policy
        self._freeze_encoders = freeze_encoders
        self._freeze_fusion = freeze_fusion
        self._freeze_temporal = freeze_temporal

        # Get dimensions from policy
        self._token_dim = getattr(student_policy, "token_dim", 128)
        self._action_dim = getattr(student_policy, "action_dim", 12)
        self._n_layers = getattr(student_policy, "n_layers", 3)
        if hasattr(student_policy, "temporal_model"):
            self._n_layers = getattr(student_policy.temporal_model, "n_layer", 3)

        # Build value head
        self.value_head = self._build_value_head(value_hidden_dims)

        # Learnable log standard deviation for Gaussian policy
        self.log_std = nn.Parameter(
            torch.full((self._action_dim,), fill_value=torch.log(torch.tensor(init_noise_std)).item())
        )

        # Internal state
        self._mems: Optional[List[Tensor]] = None
        self._distribution: Optional[Normal] = None
        self._action_mean: Optional[Tensor] = None

        # Apply freezing
        self._apply_freezing()

    def _build_value_head(self, hidden_dims: Tuple[int, ...]) -> nn.Module:
        """Build the value head MLP.

        Args:
            hidden_dims: Tuple of hidden layer dimensions.

        Returns:
            Value head module.
        """
        layers: List[nn.Module] = []
        in_features = self._token_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_features, hidden_dim))
            layers.append(nn.ReLU())
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, 1))
        return nn.Sequential(*layers)

    def _apply_freezing(self) -> None:
        """Apply freezing to specified components."""
        if self._freeze_encoders:
            if hasattr(self.student_policy, "proprio_encoder"):
                for param in self.student_policy.proprio_encoder.parameters():
                    param.requires_grad = False
            if hasattr(self.student_policy, "depth_encoder"):
                for param in self.student_policy.depth_encoder.parameters():
                    param.requires_grad = False

        if self._freeze_fusion:
            if hasattr(self.student_policy, "fusion_transformer"):
                for param in self.student_policy.fusion_transformer.parameters():
                    param.requires_grad = False

        if self._freeze_temporal:
            if hasattr(self.student_policy, "temporal_model"):
                for param in self.student_policy.temporal_model.parameters():
                    param.requires_grad = False

    def _get_temporal_features(
        self,
        proprio: Tensor,
        depth: Tensor,
        mems: Optional[List[Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Tensor, List[Tensor]]:
        """Extract temporal features and action mean from policy.

        Args:
            proprio: Proprioceptive input [B, prop_hist_len * proprio_dim].
            depth: Depth input [B, depth_hist_len, H, W].
            mems: Optional memory tensors from previous step.

        Returns:
            temporal_features: Features for value head [B, token_dim].
            action_mean: Action mean from policy [B, action_dim].
            new_mems: Updated memory tensors.
        """
        # Use policy's forward_step which returns (action_mean, yaw_pred, new_mems)
        action_mean, _, new_mems = self.student_policy.forward_step(
            proprio, depth, mems=mems
        )

        # Get temporal features for value head
        # We need to re-run the encoding to get features
        if hasattr(self.student_policy, "get_temporal_output"):
            temporal_features, _ = self.student_policy.get_temporal_output(
                proprio, depth, mems=mems
            )
        else:
            # Fallback: run encoding manually
            batch_size = proprio.shape[0]
            prop_enc = self.student_policy.proprio_encoder(proprio)
            if prop_enc.dim() == 3:
                prop_enc = prop_enc.squeeze(1)
            depth_enc = self.student_policy.depth_encoder(depth)
            if depth_enc.dim() == 3:
                depth_enc = depth_enc.mean(dim=1)

            # Fuse
            if hasattr(self.student_policy, "fusion_transformer"):
                if hasattr(self.student_policy.fusion_transformer, "forward"):
                    # Check if it's a simple linear or transformer
                    if isinstance(self.student_policy.fusion_transformer, nn.Linear):
                        fused = self.student_policy.fusion_transformer(
                            torch.cat([prop_enc, depth_enc], dim=-1)
                        )
                    else:
                        # Assume transformer-style fusion
                        fused_out = self.student_policy.fusion_transformer(
                            prop_enc.unsqueeze(1), depth_enc.unsqueeze(1)
                        )
                        if isinstance(fused_out, dict):
                            fused = fused_out.get("all_pooled", fused_out.get("output"))
                            if fused.dim() == 3:
                                fused = fused.squeeze(1)
                        else:
                            fused = fused_out
                            if fused.dim() == 3:
                                fused = fused.squeeze(1)
                else:
                    fused = torch.cat([prop_enc, depth_enc], dim=-1)
            else:
                fused = torch.cat([prop_enc, depth_enc], dim=-1)

            # Temporal
            if hasattr(self.student_policy, "temporal_model"):
                if isinstance(self.student_policy.temporal_model, nn.Linear):
                    temporal_features = self.student_policy.temporal_model(fused)
                else:
                    temporal_out, _ = self.student_policy.temporal_model(
                        fused.unsqueeze(1), mems=mems, return_mems=False
                    )
                    temporal_features = temporal_out.squeeze(1)
            else:
                temporal_features = fused

        return temporal_features, action_mean, new_mems

    def act(
        self,
        proprio: Tensor,
        depth: Tensor,
        mems: Optional[List[Tensor]] = None,
    ) -> Tuple[Tensor, Tensor, Tensor, List[Tensor]]:
        """Sample actions and compute values.

        Args:
            proprio: Proprioceptive input [B, prop_hist_len * proprio_dim].
            depth: Depth input [B, depth_hist_len, H, W].
            mems: Optional memory tensors from previous step.

        Returns:
            actions: Sampled actions [B, action_dim].
            log_probs: Log probabilities of sampled actions [B].
            values: State values [B, 1].
            new_mems: Updated memory tensors.
        """
        # Get features and action mean
        temporal_features, action_mean, new_mems = self._get_temporal_features(
            proprio, depth, mems
        )

        # Store action mean for properties
        self._action_mean = action_mean

        # Build Gaussian distribution
        std = self.log_std.exp().expand_as(action_mean)
        self._distribution = Normal(action_mean, std)

        # Sample actions
        actions = self._distribution.rsample()

        # Compute log probabilities (sum over action dimensions)
        log_probs = self._distribution.log_prob(actions).sum(dim=-1)

        # Compute values
        values = self.value_head(temporal_features)

        # Update internal memory
        self._mems = new_mems

        return actions, log_probs, values, new_mems

    def evaluate(
        self,
        proprio: Tensor,
        depth: Tensor,
        mems: Optional[List[Tensor]] = None,
    ) -> Tensor:
        """Compute state values only.

        Args:
            proprio: Proprioceptive input [B, prop_hist_len * proprio_dim].
            depth: Depth input [B, depth_hist_len, H, W].
            mems: Optional memory tensors.

        Returns:
            values: State values [B, 1].
        """
        temporal_features, _, _ = self._get_temporal_features(proprio, depth, mems)
        return self.value_head(temporal_features)

    def get_actions_log_prob(self, actions: Tensor) -> Tensor:
        """Compute log probability of given actions.

        Must be called after act() to have a valid distribution.

        Args:
            actions: Actions to evaluate [B, action_dim].

        Returns:
            log_probs: Log probabilities [B].
        """
        if self._distribution is None:
            raise RuntimeError("Must call act() before get_actions_log_prob()")
        return self._distribution.log_prob(actions).sum(dim=-1)

    def reset_memory(self, env_ids: Tensor) -> None:
        """Reset memory for specified environments.

        Called when episodes terminate to clear stale memory.

        Args:
            env_ids: Tensor of environment indices to reset.
        """
        if self._mems is None or len(env_ids) == 0:
            return

        new_mems = []
        for mem in self._mems:
            if mem is not None and mem.numel() > 0:
                # Create new tensor with zeros at specified indices (immutable pattern)
                new_mem = mem.clone()
                mask = torch.zeros(mem.shape[0], dtype=torch.bool, device=mem.device)
                mask[env_ids] = True
                new_mem[mask] = 0.0
                new_mems.append(new_mem)
            else:
                new_mems.append(mem)
        self._mems = new_mems

    def detach_memory(self) -> None:
        """Detach memory from computation graph.

        Called between training segments to prevent gradient flow
        through time beyond the segment boundary. Also clears cached
        distribution state to prevent memory leaks.
        """
        if self._mems is not None:
            self._mems = [
                mem.detach() if mem is not None else None
                for mem in self._mems
            ]

        # Clear cached distribution to prevent memory leaks
        self._distribution = None
        self._action_mean = None

    @property
    def distribution(self) -> Optional[Normal]:
        """Get the current action distribution."""
        return self._distribution

    @property
    def action_mean(self) -> Tensor:
        """Get the mean of the action distribution."""
        if self._action_mean is None:
            raise RuntimeError("Must call act() before accessing action_mean")
        return self._action_mean

    @property
    def action_std(self) -> Tensor:
        """Get the standard deviation of the action distribution."""
        if self._distribution is None:
            raise RuntimeError("Must call act() before accessing action_std")
        return self._distribution.stddev

    @property
    def entropy(self) -> Tensor:
        """Get the entropy of the action distribution."""
        if self._distribution is None:
            raise RuntimeError("Must call act() before accessing entropy")
        # Sum entropy over action dimensions
        return self._distribution.entropy().sum(dim=-1)
