"""StudentRolloutStorage for PPO training with depth images and TXL memory.

This module stores rollout data including depth images (uint8 for memory efficiency)
and Transformer-XL memory states for sequence-aware mini-batch generation.

Architecture:
    - Pre-allocated buffers for all rollout data
    - uint8 depth storage (4x memory savings vs float32)
    - Sequence-aware batching: preserves temporal ordering for TXL
    - GAE computation for advantage estimation
"""
from __future__ import annotations

from typing import Dict, Generator, List, Optional, Tuple

import torch
from torch import Tensor


class StudentRolloutStorage:
    """Storage for PPO rollouts with depth images and TXL memory states.

    Stores rollout data including proprioceptive observations, depth images,
    actions, rewards, values, and TXL memory states. Supports sequence-aware
    mini-batch generation for Transformer-XL training.

    Args:
        num_steps: Number of steps per rollout.
        num_envs: Number of parallel environments.
        proprio_dim: Dimension of proprioceptive observations.
        action_dim: Dimension of action space.
        depth_shape: Shape of depth images (depth_hist_len, H, W).
        device: Device to store tensors on.

    Attributes:
        proprio: Proprioceptive observations [num_steps, num_envs, proprio_dim].
        depth: Depth images [num_steps, num_envs, depth_hist_len, H, W] (uint8).
        actions: Actions [num_steps, num_envs, action_dim].
        rewards: Rewards [num_steps, num_envs, 1].
        values: State values [num_steps + 1, num_envs, 1].
        returns: Computed returns [num_steps, num_envs, 1].
        advantages: Computed advantages [num_steps, num_envs, 1].
        log_probs: Action log probabilities [num_steps, num_envs, 1].
        dones: Episode termination flags [num_steps, num_envs, 1].
        mu: Action distribution means [num_steps, num_envs, action_dim].
        sigma: Action distribution stds [num_steps, num_envs, action_dim].
        mems_at_step: TXL memory states at each step.
    """

    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        proprio_dim: int,
        action_dim: int,
        depth_shape: Tuple[int, int, int],
        device: str = "cpu",
    ) -> None:
        # Input validation
        if num_steps <= 0:
            raise ValueError(f"num_steps must be positive, got {num_steps}")
        if num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {num_envs}")
        if proprio_dim <= 0:
            raise ValueError(f"proprio_dim must be positive, got {proprio_dim}")
        if action_dim <= 0:
            raise ValueError(f"action_dim must be positive, got {action_dim}")

        self.num_steps = num_steps
        self.num_envs = num_envs
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.depth_shape = depth_shape
        self.device = device

        # Step counter
        self.step = 0

        # Memory states list
        self.mems_at_step: List[List[Tensor]] = []

        # Allocate buffers
        self._allocate_buffers()

    def _allocate_buffers(self) -> None:
        """Allocate all storage buffers."""
        depth_hist_len, depth_h, depth_w = self.depth_shape

        # Proprioceptive observations: [num_steps, num_envs, proprio_dim]
        self.proprio = torch.zeros(
            self.num_steps, self.num_envs, self.proprio_dim,
            dtype=torch.float32, device=self.device
        )

        # Depth images: [num_steps, num_envs, depth_hist_len, H, W] (uint8)
        self.depth = torch.zeros(
            self.num_steps, self.num_envs, depth_hist_len, depth_h, depth_w,
            dtype=torch.uint8, device=self.device
        )

        # Actions: [num_steps, num_envs, action_dim]
        self.actions = torch.zeros(
            self.num_steps, self.num_envs, self.action_dim,
            dtype=torch.float32, device=self.device
        )

        # Rewards: [num_steps, num_envs, 1]
        self.rewards = torch.zeros(
            self.num_steps, self.num_envs, 1,
            dtype=torch.float32, device=self.device
        )

        # Values: [num_steps + 1, num_envs, 1] (extra for bootstrap)
        self.values = torch.zeros(
            self.num_steps + 1, self.num_envs, 1,
            dtype=torch.float32, device=self.device
        )

        # Returns: [num_steps, num_envs, 1]
        self.returns = torch.zeros(
            self.num_steps, self.num_envs, 1,
            dtype=torch.float32, device=self.device
        )

        # Advantages: [num_steps, num_envs, 1]
        self.advantages = torch.zeros(
            self.num_steps, self.num_envs, 1,
            dtype=torch.float32, device=self.device
        )

        # Log probabilities: [num_steps, num_envs, 1]
        self.log_probs = torch.zeros(
            self.num_steps, self.num_envs, 1,
            dtype=torch.float32, device=self.device
        )

        # Dones: [num_steps, num_envs, 1]
        self.dones = torch.zeros(
            self.num_steps, self.num_envs, 1,
            dtype=torch.float32, device=self.device
        )

        # Mu (action means): [num_steps, num_envs, action_dim]
        self.mu = torch.zeros(
            self.num_steps, self.num_envs, self.action_dim,
            dtype=torch.float32, device=self.device
        )

        # Sigma (action stds): [num_steps, num_envs, action_dim]
        self.sigma = torch.zeros(
            self.num_steps, self.num_envs, self.action_dim,
            dtype=torch.float32, device=self.device
        )

    def add_transition(
        self,
        step: int,
        proprio: Tensor,
        depth: Tensor,
        actions: Tensor,
        rewards: Tensor,
        values: Tensor,
        log_probs: Tensor,
        dones: Tensor,
        mu: Tensor,
        sigma: Tensor,
        mems: Optional[List[Tensor]] = None,
    ) -> None:
        """Add a transition to storage at the specified step.

        Args:
            step: Step index to store at.
            proprio: Proprioceptive observations [num_envs, proprio_dim].
            depth: Depth images [num_envs, depth_hist_len, H, W].
            actions: Actions [num_envs, action_dim].
            rewards: Rewards [num_envs, 1].
            values: State values [num_envs, 1].
            log_probs: Action log probabilities [num_envs, 1].
            dones: Episode termination flags [num_envs, 1].
            mu: Action distribution means [num_envs, action_dim].
            sigma: Action distribution stds [num_envs, action_dim].
            mems: Optional TXL memory states.
        """
        # Store proprioceptive observations
        self.proprio[step].copy_(proprio)

        # Store depth (convert to uint8 if needed)
        if depth.dtype == torch.uint8:
            self.depth[step].copy_(depth)
        else:
            # Convert float to uint8 (assume values in 0-255 range)
            self.depth[step].copy_(depth.to(torch.uint8))

        # Store other tensors
        self.actions[step].copy_(actions)
        # Ensure rewards has correct shape [num_envs, 1]
        if rewards.dim() == 1:
            rewards = rewards.unsqueeze(-1)
        self.rewards[step].copy_(rewards)
        # Ensure values has correct shape [num_envs, 1]
        if values.dim() == 1:
            values = values.unsqueeze(-1)
        self.values[step].copy_(values)
        # Ensure log_probs has correct shape [num_envs, 1]
        if log_probs.dim() == 1:
            log_probs = log_probs.unsqueeze(-1)
        self.log_probs[step].copy_(log_probs)
        # Ensure dones has correct shape [num_envs, 1]
        if dones.dim() == 1:
            dones = dones.unsqueeze(-1)
        self.dones[step].copy_(dones)
        self.mu[step].copy_(mu)
        self.sigma[step].copy_(sigma)

        # Store memory states if provided
        if mems is not None:
            # Clone memories to avoid reference issues
            cloned_mems = [m.clone() if m is not None else None for m in mems]
            self.mems_at_step.append(cloned_mems)

        # Increment step counter
        self.step = step + 1

    def compute_returns(
        self,
        last_values: Tensor,
        gamma: float,
        lam: float,
    ) -> None:
        """Compute returns and advantages using Generalized Advantage Estimation.

        GAE formula:
            delta_t = r_t + gamma * V(s_{t+1}) * (1 - done_t) - V(s_t)
            A_t = delta_t + gamma * lam * (1 - done_t) * A_{t+1}
            R_t = A_t + V(s_t)

        Args:
            last_values: Bootstrap values for final step [num_envs, 1].
            gamma: Discount factor.
            lam: GAE lambda parameter.
        """
        # Store last values for bootstrapping
        self.values[self.num_steps].copy_(last_values)

        # Initialize advantage for backward pass
        advantage = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)

        # Compute GAE backwards
        for step in reversed(range(self.num_steps)):
            # Mask for non-terminal states
            not_done = 1.0 - self.dones[step]

            # TD error: delta = r + gamma * V(s') * (1 - done) - V(s)
            delta = (
                self.rewards[step]
                + gamma * self.values[step + 1] * not_done
                - self.values[step]
            )

            # GAE: A = delta + gamma * lam * (1 - done) * A'
            advantage = delta + gamma * lam * not_done * advantage

            # Store advantage
            self.advantages[step].copy_(advantage)

            # Compute return: R = A + V
            self.returns[step].copy_(advantage + self.values[step])

    def sequence_mini_batch_generator(
        self,
        num_batches: int,
        num_epochs: int,
    ) -> Generator[Dict[str, Tensor], None, None]:
        """Generate mini-batches that preserve sequence structure.

        Strategy: Split environments into mini-batches, keep full sequences intact.
        Each mini-batch contains: [num_steps, num_envs // num_batches, ...]

        Args:
            num_batches: Number of mini-batches per epoch.
            num_epochs: Number of epochs to iterate.

        Yields:
            Dictionary containing batch data with keys:
                - proprio: [num_steps, batch_size, proprio_dim]
                - depth: [num_steps, batch_size, depth_hist_len, H, W] (float32)
                - actions: [num_steps, batch_size, action_dim]
                - rewards: [num_steps, batch_size, 1]
                - values: [num_steps, batch_size, 1]
                - returns: [num_steps, batch_size, 1]
                - advantages: [num_steps, batch_size, 1]
                - log_probs: [num_steps, batch_size, 1]
                - dones: [num_steps, batch_size, 1]
                - mu: [num_steps, batch_size, action_dim]
                - sigma: [num_steps, batch_size, action_dim]
                - initial_mems: List of memory tensors from step 0
        """
        batch_size = self.num_envs // num_batches

        for epoch in range(num_epochs):
            # Shuffle environment indices at start of each epoch
            env_indices = torch.randperm(self.num_envs, device=self.device)

            for batch_idx in range(num_batches):
                start = batch_idx * batch_size
                end = start + batch_size

                # Handle uneven splits - last batch gets remaining envs
                if batch_idx == num_batches - 1:
                    end = self.num_envs

                batch_env_ids = env_indices[start:end]

                # Extract initial memories for this batch
                initial_mems = None
                if len(self.mems_at_step) > 0 and self.mems_at_step[0] is not None:
                    initial_mems = [
                        mem[batch_env_ids] if mem is not None else None
                        for mem in self.mems_at_step[0]
                    ]

                yield {
                    "proprio": self.proprio[:, batch_env_ids],
                    "depth": self.depth[:, batch_env_ids].float(),  # Convert to float32
                    "actions": self.actions[:, batch_env_ids],
                    "rewards": self.rewards[:, batch_env_ids],
                    "values": self.values[:self.num_steps, batch_env_ids],
                    "returns": self.returns[:, batch_env_ids],
                    "advantages": self.advantages[:, batch_env_ids],
                    "log_probs": self.log_probs[:, batch_env_ids],
                    "dones": self.dones[:, batch_env_ids],
                    "mu": self.mu[:, batch_env_ids],
                    "sigma": self.sigma[:, batch_env_ids],
                    "initial_mems": initial_mems,
                }

    def clear(self) -> None:
        """Reset storage for next rollout.

        Zeros all buffers, clears memory states, and resets step counter.
        """
        # Reset step counter
        self.step = 0

        # Clear memory states
        self.mems_at_step = []

        # Zero all buffers
        self.proprio.zero_()
        self.depth.zero_()
        self.actions.zero_()
        self.rewards.zero_()
        self.values.zero_()
        self.returns.zero_()
        self.advantages.zero_()
        self.log_probs.zero_()
        self.dones.zero_()
        self.mu.zero_()
        self.sigma.zero_()
