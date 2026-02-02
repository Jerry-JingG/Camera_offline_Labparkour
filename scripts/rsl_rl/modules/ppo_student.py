"""PPOStudent algorithm for student policy with Transformer-XL memory.

This module implements the PPO algorithm adapted for image-based observations
and Transformer-XL memory management. It handles sequence-aware batching
and proper memory reset on episode termination.

Architecture:
    - Uses StudentActorCritic for policy and value estimation
    - Uses StudentRolloutStorage for sequence-aware data storage
    - Implements PPO with clipped surrogate objective
    - Supports adaptive learning rate based on KL divergence
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn, Tensor
import torch.optim as optim

# Handle both relative and absolute imports for testing
try:
    from .student_actor_critic import StudentActorCritic
    from .student_rollout_storage import StudentRolloutStorage
except ImportError:
    from student_actor_critic import StudentActorCritic
    from student_rollout_storage import StudentRolloutStorage


@dataclass
class Transition:
    """Container for transition data during rollout collection."""

    observations: Optional[Tensor] = None
    depth: Optional[Tensor] = None
    actions: Optional[Tensor] = None
    rewards: Optional[Tensor] = None
    values: Optional[Tensor] = None
    log_probs: Optional[Tensor] = None
    dones: Optional[Tensor] = None
    mu: Optional[Tensor] = None
    sigma: Optional[Tensor] = None
    mems: Optional[List[Tensor]] = None


class PPOStudent:
    """PPO algorithm for student policy with Transformer-XL.

    Implements Proximal Policy Optimization adapted for:
    - Image-based observations (proprioception + depth)
    - Transformer-XL memory management
    - Sequence-aware mini-batch generation

    Args:
        actor_critic: StudentActorCritic model for policy and value.
        num_learning_epochs: Number of epochs per update.
        num_mini_batches: Number of mini-batches per epoch.
        clip_param: PPO clipping parameter.
        gamma: Discount factor.
        lam: GAE lambda parameter.
        value_loss_coef: Value loss coefficient.
        entropy_coef: Entropy bonus coefficient.
        learning_rate: Optimizer learning rate.
        max_grad_norm: Maximum gradient norm for clipping.
        use_clipped_value_loss: Whether to clip value loss.
        schedule: Learning rate schedule ('fixed' or 'adaptive').
        desired_kl: Target KL divergence for adaptive schedule.
        device: Device to run on.
    """

    def __init__(
        self,
        actor_critic: StudentActorCritic,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 1e-4,
        max_grad_norm: float = 1.0,
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        device: str = "cuda",
    ) -> None:
        # Input validation
        if clip_param < 0:
            raise ValueError(f"clip_param must be non-negative, got {clip_param}")
        if gamma < 0 or gamma > 1:
            raise ValueError(f"gamma must be in [0, 1], got {gamma}")
        if lam < 0 or lam > 1:
            raise ValueError(f"lam must be in [0, 1], got {lam}")

        self.actor_critic = actor_critic
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.clip_param = clip_param
        self.gamma = gamma
        self.lam = lam
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.schedule = schedule
        self.desired_kl = desired_kl
        self.device = device

        # Create optimizer
        self.optimizer = optim.AdamW(
            actor_critic.parameters(),
            lr=learning_rate,
        )

        # Storage and state (initialized later)
        self.storage: Optional[StudentRolloutStorage] = None
        self.transition = Transition()
        self.current_mems: Optional[List[Tensor]] = None
        self.step = 0

        # Move actor_critic to device
        self.actor_critic.to(device)

    def init_storage(
        self,
        num_envs: int,
        num_steps: int,
        proprio_dim: int,
        depth_shape: Tuple[int, int, int],
        action_dim: int,
    ) -> None:
        """Initialize rollout storage.

        Args:
            num_envs: Number of parallel environments.
            num_steps: Number of steps per rollout.
            proprio_dim: Dimension of proprioceptive observations.
            depth_shape: Shape of depth images (depth_hist_len, H, W).
            action_dim: Dimension of action space.
        """
        self.storage = StudentRolloutStorage(
            num_steps=num_steps,
            num_envs=num_envs,
            proprio_dim=proprio_dim,
            action_dim=action_dim,
            depth_shape=depth_shape,
            device=self.device,
        )

        # Reset state
        self.step = 0
        self.current_mems = None
        self.transition = Transition()

    def act(self, proprio: Tensor, depth: Tensor) -> Tensor:
        """Sample actions from policy.

        Args:
            proprio: Proprioceptive observations [num_envs, proprio_dim].
            depth: Depth images [num_envs, depth_hist_len, H, W].

        Returns:
            actions: Sampled actions [num_envs, action_dim].
        """
        # Forward through actor-critic with memory
        actions, log_probs, values, new_mems = self.actor_critic.act(
            proprio, depth, mems=self.current_mems
        )

        # Store transition data (detached for storage)
        self.transition.observations = proprio.detach()
        self.transition.depth = depth.detach()
        self.transition.actions = actions.detach()
        self.transition.values = values.detach()
        self.transition.log_probs = log_probs.detach().unsqueeze(-1)
        self.transition.mu = self.actor_critic.action_mean.detach()
        self.transition.sigma = self.actor_critic.action_std.detach()
        self.transition.mems = (
            [m.detach() if m is not None else None for m in new_mems]
            if new_mems is not None
            else None
        )

        # Update current memory (detached to prevent gradient accumulation)
        self.current_mems = (
            [m.detach() if m is not None else None for m in new_mems]
            if new_mems is not None
            else None
        )

        return actions.detach()

    def process_env_step(
        self,
        rewards: Tensor,
        dones: Tensor,
        infos: Dict,
    ) -> None:
        """Process environment step and store transition.

        Args:
            rewards: Rewards from environment [num_envs, 1].
            dones: Done flags from environment [num_envs, 1].
            infos: Additional info from environment.
        """
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage first.")

        # Store transition in storage
        self.storage.add_transition(
            step=self.step,
            proprio=self.transition.observations,
            depth=self.transition.depth,
            actions=self.transition.actions,
            rewards=rewards,
            values=self.transition.values,
            log_probs=self.transition.log_probs,
            dones=dones,
            mu=self.transition.mu,
            sigma=self.transition.sigma,
            mems=self.transition.mems,
        )

        # Reset memory for done environments
        done_env_ids = dones.squeeze(-1).nonzero(as_tuple=False).squeeze(-1)
        if len(done_env_ids) > 0:
            self._reset_memory_for_envs(done_env_ids)

        # Increment step counter
        self.step += 1

    def _reset_memory_for_envs(self, env_ids: Tensor) -> None:
        """Reset TXL memory for specified environments.

        Args:
            env_ids: Tensor of environment indices to reset.
        """
        if self.current_mems is None or len(env_ids) == 0:
            return

        # Create new memory list with reset values (immutable pattern)
        new_mems = []
        for mem in self.current_mems:
            if mem is not None and mem.numel() > 0:
                new_mem = mem.clone()
                new_mem[env_ids] = 0.0
                new_mems.append(new_mem)
            else:
                new_mems.append(mem)

        self.current_mems = new_mems

    def compute_returns(self, last_values: Tensor) -> None:
        """Compute returns and advantages using GAE.

        Args:
            last_values: Bootstrap values for final step [num_envs, 1].
        """
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage first.")

        self.storage.compute_returns(
            last_values=last_values,
            gamma=self.gamma,
            lam=self.lam,
        )

    def update(self) -> Dict[str, float]:
        """Update policy using PPO.

        Returns:
            Dictionary containing loss metrics.
        """
        if self.storage is None:
            raise RuntimeError("Storage not initialized. Call init_storage first.")

        # Initialize loss accumulators
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_kl = 0.0
        num_updates = 0

        # Generate mini-batches using sequence-aware generator
        generator = self.storage.sequence_mini_batch_generator(
            num_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        for batch in generator:
            # Extract batch data
            proprio_batch = batch["proprio"]  # [num_steps, batch_size, proprio_dim]
            depth_batch = batch["depth"]  # [num_steps, batch_size, ...]
            actions_batch = batch["actions"]  # [num_steps, batch_size, action_dim]
            old_values_batch = batch["values"]  # [num_steps, batch_size, 1]
            returns_batch = batch["returns"]  # [num_steps, batch_size, 1]
            advantages_batch = batch["advantages"]  # [num_steps, batch_size, 1]
            old_log_probs_batch = batch["log_probs"]  # [num_steps, batch_size, 1]
            old_mu_batch = batch["mu"]  # [num_steps, batch_size, action_dim]
            old_sigma_batch = batch["sigma"]  # [num_steps, batch_size, action_dim]
            initial_mems = batch["initial_mems"]

            # Flatten for processing
            num_steps, batch_size = proprio_batch.shape[:2]
            total_samples = num_steps * batch_size

            proprio_flat = proprio_batch.reshape(total_samples, -1)
            depth_flat = depth_batch.reshape(total_samples, *depth_batch.shape[2:])
            actions_flat = actions_batch.reshape(total_samples, -1)
            old_values_flat = old_values_batch.reshape(total_samples, 1)
            returns_flat = returns_batch.reshape(total_samples, 1)
            advantages_flat = advantages_batch.reshape(total_samples, 1)
            old_log_probs_flat = old_log_probs_batch.reshape(total_samples, 1)
            old_mu_flat = old_mu_batch.reshape(total_samples, -1)
            old_sigma_flat = old_sigma_batch.reshape(total_samples, -1)

            # Normalize advantages
            advantages_flat = (advantages_flat - advantages_flat.mean()) / (
                advantages_flat.std() + 1e-8
            )

            # Forward pass through actor-critic
            # Note: For simplicity, we process all samples at once
            # A more sophisticated implementation would process sequentially
            _, log_probs, values, _ = self.actor_critic.act(
                proprio_flat, depth_flat, mems=None
            )
            log_probs = log_probs.unsqueeze(-1)

            # Get current distribution parameters
            mu = self.actor_critic.action_mean
            sigma = self.actor_critic.action_std
            entropy = self.actor_critic.entropy

            # Compute PPO surrogate loss
            ratio = torch.exp(log_probs - old_log_probs_flat)
            surrogate1 = ratio * advantages_flat
            surrogate2 = (
                torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
                * advantages_flat
            )
            surrogate_loss = -torch.min(surrogate1, surrogate2).mean()

            # Compute value loss
            if self.use_clipped_value_loss:
                value_clipped = old_values_flat + torch.clamp(
                    values - old_values_flat,
                    -self.clip_param,
                    self.clip_param,
                )
                value_loss1 = (values - returns_flat).pow(2)
                value_loss2 = (value_clipped - returns_flat).pow(2)
                value_loss = torch.max(value_loss1, value_loss2).mean()
            else:
                value_loss = (values - returns_flat).pow(2).mean()

            # Compute entropy bonus
            entropy_loss = -entropy.mean()

            # Total loss
            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                + self.entropy_coef * entropy_loss
            )

            # Compute KL divergence for adaptive LR
            with torch.no_grad():
                kl = torch.sum(
                    torch.log(sigma / old_sigma_flat + 1e-5)
                    + (old_sigma_flat.pow(2) + (old_mu_flat - mu).pow(2))
                    / (2.0 * sigma.pow(2))
                    - 0.5,
                    dim=-1,
                )
                kl_mean = kl.mean()

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            nn.utils.clip_grad_norm_(
                self.actor_critic.parameters(),
                self.max_grad_norm,
            )

            # Optimizer step
            self.optimizer.step()

            # Accumulate losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            mean_kl += kl_mean.item()
            num_updates += 1

        # Average losses
        if num_updates > 0:
            mean_value_loss /= num_updates
            mean_surrogate_loss /= num_updates
            mean_entropy /= num_updates
            mean_kl /= num_updates

        # Adaptive learning rate
        if self.schedule == "adaptive" and self.desired_kl is not None:
            if mean_kl > self.desired_kl * 2.0:
                self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif mean_kl < self.desired_kl / 2.0 and mean_kl > 0.0:
                self.learning_rate = min(1e-2, self.learning_rate * 1.5)

            for param_group in self.optimizer.param_groups:
                param_group["lr"] = self.learning_rate

        # Clear storage
        self.storage.clear()
        self.step = 0

        return {
            "value_loss": mean_value_loss,
            "surrogate_loss": mean_surrogate_loss,
            "policy_loss": mean_surrogate_loss,  # Alias
            "entropy": mean_entropy,
            "kl": mean_kl,
            "learning_rate": self.learning_rate,
        }
