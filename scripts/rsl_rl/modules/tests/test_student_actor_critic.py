"""Unit tests for StudentActorCritic module.

TDD RED phase: These tests are written BEFORE implementation.
All tests should FAIL initially until StudentActorCritic is implemented.

Test cases:
- test_student_actor_critic_act_shapes: Verify output shapes
- test_student_actor_critic_evaluate_shape: Verify value output shape
- test_student_actor_critic_memory_reset: Verify memory reset on done envs
- test_student_actor_critic_memory_detach: Verify memory detachment
- test_student_actor_critic_gradient_flow: Verify gradients flow correctly
- test_student_actor_critic_freeze_encoders: Verify encoder freezing works
- test_student_actor_critic_gaussian_distribution: Verify action distribution
- test_student_actor_critic_log_prob_computation: Verify log prob calculation
"""
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import nn, Tensor


def _ensure_modules_on_path() -> None:
    """Ensure the project root is available on sys.path."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to scripts/rsl_rl/modules
    modules_dir = os.path.dirname(file_dir)
    if modules_dir not in sys.path:
        sys.path.insert(0, modules_dir)


_ensure_modules_on_path()

# Direct import from the module file to avoid __init__.py issues
from student_actor_critic import StudentActorCritic


# Test configuration constants
BATCH_SIZE = 4
PROPRIO_DIM = 48
ACTION_DIM = 12
PROP_HIST_LEN = 5
DEPTH_HIST_LEN = 3
CAMERA_HEIGHT = 64
CAMERA_WIDTH = 64
TOKEN_DIM = 128
MEM_LEN = 64
N_LAYERS = 3


class MockMultiModalStudentPolicy(nn.Module):
    """Mock student policy for testing without full model dependencies."""

    def __init__(
        self,
        proprio_dim: int = PROPRIO_DIM,
        action_dim: int = ACTION_DIM,
        prop_hist_len: int = PROP_HIST_LEN,
        depth_hist_len: int = DEPTH_HIST_LEN,
        token_dim: int = TOKEN_DIM,
        n_layers: int = N_LAYERS,
        mem_len: int = MEM_LEN,
    ) -> None:
        super().__init__()
        self.proprio_dim = proprio_dim
        self.action_dim = action_dim
        self.prop_hist_len = prop_hist_len
        self.depth_hist_len = depth_hist_len
        self.token_dim = token_dim
        self.n_layers = n_layers
        self.mem_len = mem_len

        # Mock encoders
        self.proprio_encoder = nn.Linear(proprio_dim * prop_hist_len, token_dim)
        self.depth_encoder = nn.Sequential(
            nn.Conv2d(depth_hist_len, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(32, token_dim),
        )

        # Mock fusion
        self.fusion_transformer = nn.Linear(token_dim * 2, token_dim)

        # Mock temporal model
        self.temporal_model = nn.Linear(token_dim, token_dim)

        # Mock action head with log_std
        self.action_head = nn.ModuleDict({
            "mu_head": nn.Linear(token_dim, action_dim),
        })
        self.action_head_log_std = nn.Parameter(torch.zeros(action_dim))

    def forward_step(
        self,
        proprio_step: Tensor,
        depth_step: Tensor,
        mems: Optional[List[Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[Tensor], Optional[List[Tensor]]]:
        """Single step forward pass returning action means."""
        batch_size = proprio_step.shape[0]

        # Encode
        prop_enc = self.proprio_encoder(proprio_step)
        depth_enc = self.depth_encoder(depth_step)

        # Fuse
        fused = self.fusion_transformer(torch.cat([prop_enc, depth_enc], dim=-1))

        # Temporal (simplified - no real memory)
        temporal_out = self.temporal_model(fused)

        # Action
        action_mean = self.action_head["mu_head"](temporal_out)

        # Create new mems (simplified)
        if mems is None:
            new_mems = [
                torch.zeros(batch_size, 0, self.token_dim, device=proprio_step.device)
                for _ in range(self.n_layers)
            ]
        else:
            # Simulate memory update
            new_mems = []
            for mem in mems:
                if mem is None or mem.size(1) == 0:
                    new_mem = temporal_out.unsqueeze(1)
                else:
                    new_mem = torch.cat([mem, temporal_out.unsqueeze(1)], dim=1)
                    if new_mem.size(1) > self.mem_len:
                        new_mem = new_mem[:, -self.mem_len:, :]
                new_mems.append(new_mem.detach())

        return action_mean, None, new_mems

    def get_temporal_output(
        self,
        proprio_step: Tensor,
        depth_step: Tensor,
        mems: Optional[List[Optional[Tensor]]] = None,
    ) -> Tuple[Tensor, Optional[List[Tensor]]]:
        """Get temporal features for value head."""
        batch_size = proprio_step.shape[0]

        # Encode
        prop_enc = self.proprio_encoder(proprio_step)
        depth_enc = self.depth_encoder(depth_step)

        # Fuse
        fused = self.fusion_transformer(torch.cat([prop_enc, depth_enc], dim=-1))

        # Temporal
        temporal_out = self.temporal_model(fused)

        # Create new mems
        if mems is None:
            new_mems = [
                torch.zeros(batch_size, 0, self.token_dim, device=proprio_step.device)
                for _ in range(self.n_layers)
            ]
        else:
            new_mems = []
            for mem in mems:
                if mem is None or mem.size(1) == 0:
                    new_mem = temporal_out.unsqueeze(1)
                else:
                    new_mem = torch.cat([mem, temporal_out.unsqueeze(1)], dim=1)
                    if new_mem.size(1) > self.mem_len:
                        new_mem = new_mem[:, -self.mem_len:, :]
                new_mems.append(new_mem.detach())

        return temporal_out, new_mems


def create_mock_policy() -> MockMultiModalStudentPolicy:
    """Factory function to create mock policy."""
    return MockMultiModalStudentPolicy()


def create_test_inputs(
    batch_size: int = BATCH_SIZE,
) -> Tuple[Tensor, Tensor]:
    """Create test input tensors."""
    proprio = torch.randn(batch_size, PROP_HIST_LEN * PROPRIO_DIM)
    depth = torch.randn(batch_size, DEPTH_HIST_LEN, CAMERA_HEIGHT, CAMERA_WIDTH)
    return proprio, depth


class TestStudentActorCriticActShapes:
    """Test act() method output shapes."""

    def test_act_returns_correct_shapes(self) -> None:
        """Verify act() returns (actions, log_probs, values, new_mems) with correct shapes."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(
            student_policy=policy,
            value_hidden_dims=(256, 256),
            init_noise_std=1.0,
        )

        proprio, depth = create_test_inputs()
        actions, log_probs, values, new_mems = actor_critic.act(proprio, depth, mems=None)

        # Actions: [B, action_dim]
        assert actions.shape == (BATCH_SIZE, ACTION_DIM), (
            f"Expected actions shape ({BATCH_SIZE}, {ACTION_DIM}), got {tuple(actions.shape)}"
        )

        # Log probs: [B]
        assert log_probs.shape == (BATCH_SIZE,), (
            f"Expected log_probs shape ({BATCH_SIZE},), got {tuple(log_probs.shape)}"
        )

        # Values: [B, 1]
        assert values.shape == (BATCH_SIZE, 1), (
            f"Expected values shape ({BATCH_SIZE}, 1), got {tuple(values.shape)}"
        )

        # New mems: List of [B, mem_len, d_model]
        assert new_mems is not None, "new_mems should not be None"
        assert len(new_mems) == N_LAYERS, (
            f"Expected {N_LAYERS} memory tensors, got {len(new_mems)}"
        )

    def test_act_with_existing_mems(self) -> None:
        """Verify act() works with pre-existing memory tensors."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()

        # Create initial mems
        initial_mems = [
            torch.randn(BATCH_SIZE, 5, TOKEN_DIM) for _ in range(N_LAYERS)
        ]

        actions, log_probs, values, new_mems = actor_critic.act(
            proprio, depth, mems=initial_mems
        )

        assert actions.shape == (BATCH_SIZE, ACTION_DIM)
        assert log_probs.shape == (BATCH_SIZE,)
        assert values.shape == (BATCH_SIZE, 1)
        assert new_mems is not None

    def test_act_outputs_are_finite(self) -> None:
        """Verify all outputs contain no NaN or Inf values."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        actions, log_probs, values, new_mems = actor_critic.act(proprio, depth, mems=None)

        assert torch.isfinite(actions).all(), "Actions contain NaN or Inf"
        assert torch.isfinite(log_probs).all(), "Log probs contain NaN or Inf"
        assert torch.isfinite(values).all(), "Values contain NaN or Inf"
        for i, mem in enumerate(new_mems):
            if mem.numel() > 0:
                assert torch.isfinite(mem).all(), f"Memory {i} contains NaN or Inf"


class TestStudentActorCriticEvaluateShape:
    """Test evaluate() method output shape."""

    def test_evaluate_returns_correct_shape(self) -> None:
        """Verify evaluate() returns values with shape [B, 1]."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        values = actor_critic.evaluate(proprio, depth, mems=None)

        assert values.shape == (BATCH_SIZE, 1), (
            f"Expected values shape ({BATCH_SIZE}, 1), got {tuple(values.shape)}"
        )

    def test_evaluate_with_mems(self) -> None:
        """Verify evaluate() works with memory tensors."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        mems = [torch.randn(BATCH_SIZE, 5, TOKEN_DIM) for _ in range(N_LAYERS)]

        values = actor_critic.evaluate(proprio, depth, mems=mems)

        assert values.shape == (BATCH_SIZE, 1)
        assert torch.isfinite(values).all()

    def test_evaluate_is_finite(self) -> None:
        """Verify evaluate output contains no NaN or Inf."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        values = actor_critic.evaluate(proprio, depth, mems=None)

        assert torch.isfinite(values).all(), "Values contain NaN or Inf"


class TestStudentActorCriticMemoryReset:
    """Test memory reset functionality on episode termination."""

    def test_reset_memory_zeros_specified_envs(self) -> None:
        """Verify reset_memory zeros out memory for specified environment IDs."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        # Initialize memory by running act
        proprio, depth = create_test_inputs()
        _, _, _, mems = actor_critic.act(proprio, depth, mems=None)

        # Run a few more steps to build up memory
        for _ in range(3):
            _, _, _, mems = actor_critic.act(proprio, depth, mems=mems)

        # Store memory state before reset
        actor_critic._mems = mems

        # Reset envs 0 and 2
        env_ids = torch.tensor([0, 2])
        actor_critic.reset_memory(env_ids)

        # Check that envs 0 and 2 have zeroed memory
        for layer_mem in actor_critic._mems:
            if layer_mem is not None and layer_mem.numel() > 0:
                # Envs 0 and 2 should be zero or empty
                assert layer_mem[0].abs().sum() == 0 or layer_mem.size(1) == 0, (
                    "Env 0 memory should be reset"
                )
                assert layer_mem[2].abs().sum() == 0 or layer_mem.size(1) == 0, (
                    "Env 2 memory should be reset"
                )

    def test_reset_memory_preserves_other_envs(self) -> None:
        """Verify reset_memory preserves memory for non-specified environments."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()

        # Build up memory
        _, _, _, mems = actor_critic.act(proprio, depth, mems=None)
        for _ in range(3):
            _, _, _, mems = actor_critic.act(proprio, depth, mems=mems)

        actor_critic._mems = mems

        # Store env 1 and 3 memory before reset
        env1_mem_before = [m[1].clone() if m is not None else None for m in mems]
        env3_mem_before = [m[3].clone() if m is not None else None for m in mems]

        # Reset only envs 0 and 2
        env_ids = torch.tensor([0, 2])
        actor_critic.reset_memory(env_ids)

        # Check envs 1 and 3 are preserved
        for i, layer_mem in enumerate(actor_critic._mems):
            if layer_mem is not None and env1_mem_before[i] is not None:
                if layer_mem.size(1) > 0 and env1_mem_before[i].size(0) > 0:
                    assert torch.allclose(layer_mem[1], env1_mem_before[i], atol=1e-6), (
                        f"Env 1 memory in layer {i} should be preserved"
                    )

    def test_reset_memory_empty_env_ids(self) -> None:
        """Verify reset_memory handles empty env_ids gracefully."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        _, _, _, mems = actor_critic.act(proprio, depth, mems=None)
        actor_critic._mems = mems

        # Reset with empty tensor - should not raise
        env_ids = torch.tensor([], dtype=torch.long)
        actor_critic.reset_memory(env_ids)  # Should not raise


class TestStudentActorCriticMemoryDetach:
    """Test memory detachment from computation graph."""

    def test_detach_memory_removes_grad_fn(self) -> None:
        """Verify detach_memory removes gradient tracking from memories."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        proprio.requires_grad_(True)

        _, _, _, mems = actor_critic.act(proprio, depth, mems=None)
        actor_critic._mems = mems

        actor_critic.detach_memory()

        # Check all memories are detached
        for i, mem in enumerate(actor_critic._mems):
            if mem is not None:
                assert not mem.requires_grad, f"Memory {i} should not require grad"
                assert mem.grad_fn is None, f"Memory {i} should have no grad_fn"

    def test_detach_memory_preserves_values(self) -> None:
        """Verify detach_memory preserves memory values."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        _, _, _, mems = actor_critic.act(proprio, depth, mems=None)

        # Store values before detach
        mems_before = [m.clone() if m is not None else None for m in mems]
        actor_critic._mems = mems

        actor_critic.detach_memory()

        # Check values are preserved
        for i, (before, after) in enumerate(zip(mems_before, actor_critic._mems)):
            if before is not None and after is not None:
                assert torch.allclose(before, after, atol=1e-6), (
                    f"Memory {i} values changed after detach"
                )


class TestStudentActorCriticGradientFlow:
    """Test gradient flow through the network."""

    def test_gradient_flow_through_value_head(self) -> None:
        """Verify gradients flow through value head."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        proprio.requires_grad_(True)

        _, _, values, _ = actor_critic.act(proprio, depth, mems=None)
        loss = values.sum()
        loss.backward()

        # Check value head has gradients
        for name, param in actor_critic.value_head.named_parameters():
            assert param.grad is not None, f"No gradient for value_head.{name}"
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient for value_head.{name}"
            )

    def test_gradient_flow_through_policy(self) -> None:
        """Verify gradients flow through policy when not frozen."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(
            student_policy=policy,
            freeze_encoders=False,
            freeze_fusion=False,
            freeze_temporal=False,
        )

        proprio, depth = create_test_inputs()
        proprio.requires_grad_(True)

        actions, log_probs, values, _ = actor_critic.act(proprio, depth, mems=None)
        loss = log_probs.sum() + values.sum()
        loss.backward()

        # Check policy has gradients
        has_grad = False
        for name, param in actor_critic.student_policy.named_parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients flowing through policy"

    def test_gradient_blocked_when_frozen(self) -> None:
        """Verify gradients are blocked when encoders are frozen."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(
            student_policy=policy,
            freeze_encoders=True,
        )

        proprio, depth = create_test_inputs()
        proprio.requires_grad_(True)

        actions, log_probs, values, _ = actor_critic.act(proprio, depth, mems=None)
        loss = values.sum()
        loss.backward()

        # Check encoder parameters have no gradients
        for name, param in actor_critic.student_policy.proprio_encoder.named_parameters():
            assert param.grad is None or param.grad.abs().sum() == 0, (
                f"Frozen encoder {name} should have no gradient"
            )


class TestStudentActorCriticFreezeEncoders:
    """Test encoder freezing functionality."""

    def test_freeze_encoders_sets_requires_grad_false(self) -> None:
        """Verify freeze_encoders=True sets requires_grad=False on encoders."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(
            student_policy=policy,
            freeze_encoders=True,
        )

        # Check proprio encoder is frozen
        for param in actor_critic.student_policy.proprio_encoder.parameters():
            assert not param.requires_grad, "Proprio encoder should be frozen"

        # Check depth encoder is frozen
        for param in actor_critic.student_policy.depth_encoder.parameters():
            assert not param.requires_grad, "Depth encoder should be frozen"

    def test_freeze_fusion_sets_requires_grad_false(self) -> None:
        """Verify freeze_fusion=True sets requires_grad=False on fusion."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(
            student_policy=policy,
            freeze_fusion=True,
        )

        for param in actor_critic.student_policy.fusion_transformer.parameters():
            assert not param.requires_grad, "Fusion transformer should be frozen"

    def test_freeze_temporal_sets_requires_grad_false(self) -> None:
        """Verify freeze_temporal=True sets requires_grad=False on temporal."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(
            student_policy=policy,
            freeze_temporal=True,
        )

        for param in actor_critic.student_policy.temporal_model.parameters():
            assert not param.requires_grad, "Temporal model should be frozen"

    def test_value_head_always_trainable(self) -> None:
        """Verify value head is always trainable regardless of freeze settings."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(
            student_policy=policy,
            freeze_encoders=True,
            freeze_fusion=True,
            freeze_temporal=True,
        )

        for param in actor_critic.value_head.parameters():
            assert param.requires_grad, "Value head should always be trainable"


class TestStudentActorCriticGaussianDistribution:
    """Test Gaussian action distribution."""

    def test_distribution_is_normal(self) -> None:
        """Verify action distribution is Normal/Gaussian."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        actor_critic.act(proprio, depth, mems=None)

        dist = actor_critic.distribution
        assert dist is not None, "Distribution should be set after act()"
        assert hasattr(dist, "mean"), "Distribution should have mean"
        assert hasattr(dist, "stddev"), "Distribution should have stddev"

    def test_action_mean_property(self) -> None:
        """Verify action_mean property returns correct shape."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        actor_critic.act(proprio, depth, mems=None)

        action_mean = actor_critic.action_mean
        assert action_mean.shape == (BATCH_SIZE, ACTION_DIM), (
            f"Expected action_mean shape ({BATCH_SIZE}, {ACTION_DIM}), "
            f"got {tuple(action_mean.shape)}"
        )

    def test_action_std_property(self) -> None:
        """Verify action_std property returns correct shape."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        actor_critic.act(proprio, depth, mems=None)

        action_std = actor_critic.action_std
        assert action_std.shape == (BATCH_SIZE, ACTION_DIM), (
            f"Expected action_std shape ({BATCH_SIZE}, {ACTION_DIM}), "
            f"got {tuple(action_std.shape)}"
        )
        assert (action_std > 0).all(), "Standard deviation must be positive"

    def test_entropy_property(self) -> None:
        """Verify entropy property returns correct shape."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        actor_critic.act(proprio, depth, mems=None)

        entropy = actor_critic.entropy
        assert entropy.shape == (BATCH_SIZE,), (
            f"Expected entropy shape ({BATCH_SIZE},), got {tuple(entropy.shape)}"
        )
        assert torch.isfinite(entropy).all(), "Entropy should be finite"


class TestStudentActorCriticLogProbComputation:
    """Test log probability computation."""

    def test_get_actions_log_prob_shape(self) -> None:
        """Verify get_actions_log_prob returns correct shape."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        actions, _, _, _ = actor_critic.act(proprio, depth, mems=None)

        log_probs = actor_critic.get_actions_log_prob(actions)
        assert log_probs.shape == (BATCH_SIZE,), (
            f"Expected log_probs shape ({BATCH_SIZE},), got {tuple(log_probs.shape)}"
        )

    def test_log_prob_is_finite(self) -> None:
        """Verify log probabilities are finite."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        actions, _, _, _ = actor_critic.act(proprio, depth, mems=None)

        log_probs = actor_critic.get_actions_log_prob(actions)
        assert torch.isfinite(log_probs).all(), "Log probs contain NaN or Inf"

    def test_log_prob_consistency(self) -> None:
        """Verify log_prob from act() matches get_actions_log_prob()."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        actions, log_probs_act, _, _ = actor_critic.act(proprio, depth, mems=None)

        log_probs_method = actor_critic.get_actions_log_prob(actions)

        assert torch.allclose(log_probs_act, log_probs_method, atol=1e-5), (
            "Log probs from act() should match get_actions_log_prob()"
        )

    def test_log_prob_negative_for_unlikely_actions(self) -> None:
        """Verify log probabilities are negative (probabilities < 1)."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs()
        actor_critic.act(proprio, depth, mems=None)

        # Create actions far from mean
        extreme_actions = torch.ones(BATCH_SIZE, ACTION_DIM) * 100
        log_probs = actor_critic.get_actions_log_prob(extreme_actions)

        # Log probs should be very negative for unlikely actions
        assert (log_probs < 0).all(), "Log probs should be negative"


class TestStudentActorCriticInitialization:
    """Test proper initialization."""

    def test_init_with_default_params(self) -> None:
        """Verify initialization with default parameters."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        assert actor_critic.student_policy is policy
        assert actor_critic.value_head is not None
        assert hasattr(actor_critic, "log_std")

    def test_init_with_custom_value_hidden_dims(self) -> None:
        """Verify custom value_hidden_dims are respected."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(
            student_policy=policy,
            value_hidden_dims=(128, 64),
        )

        # Count linear layers in value head
        linear_layers = [
            m for m in actor_critic.value_head.modules() if isinstance(m, nn.Linear)
        ]
        # Expected: 2 hidden + 1 output = 3
        assert len(linear_layers) == 3, (
            f"Expected 3 linear layers, got {len(linear_layers)}"
        )

    def test_init_noise_std(self) -> None:
        """Verify init_noise_std sets initial log_std correctly."""
        policy = create_mock_policy()
        init_std = 0.5
        actor_critic = StudentActorCritic(
            student_policy=policy,
            init_noise_std=init_std,
        )

        expected_log_std = torch.log(torch.tensor(init_std))
        actual_log_std = actor_critic.log_std

        assert torch.allclose(
            actual_log_std, torch.full_like(actual_log_std, expected_log_std.item()),
            atol=1e-5
        ), f"Expected log_std ~{expected_log_std.item()}, got {actual_log_std}"


class TestStudentActorCriticBatchSizes:
    """Test with various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
    def test_act_various_batch_sizes(self, batch_size: int) -> None:
        """Verify act() works with various batch sizes."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs(batch_size=batch_size)
        actions, log_probs, values, new_mems = actor_critic.act(
            proprio, depth, mems=None
        )

        assert actions.shape == (batch_size, ACTION_DIM)
        assert log_probs.shape == (batch_size,)
        assert values.shape == (batch_size, 1)

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32])
    def test_evaluate_various_batch_sizes(self, batch_size: int) -> None:
        """Verify evaluate() works with various batch sizes."""
        policy = create_mock_policy()
        actor_critic = StudentActorCritic(student_policy=policy)

        proprio, depth = create_test_inputs(batch_size=batch_size)
        values = actor_critic.evaluate(proprio, depth, mems=None)

        assert values.shape == (batch_size, 1)
        assert torch.isfinite(values).all()


if __name__ == "__main__":
    # Simple test runner to avoid pytest plugin conflicts
    import traceback

    test_classes = [
        TestStudentActorCriticActShapes,
        TestStudentActorCriticEvaluateShape,
        TestStudentActorCriticMemoryReset,
        TestStudentActorCriticMemoryDetach,
        TestStudentActorCriticGradientFlow,
        TestStudentActorCriticFreezeEncoders,
        TestStudentActorCriticGaussianDistribution,
        TestStudentActorCriticLogProbComputation,
        TestStudentActorCriticInitialization,
    ]

    passed = 0
    failed = 0
    errors = []

    for test_class in test_classes:
        instance = test_class()
        for method_name in dir(instance):
            if method_name.startswith("test_"):
                try:
                    getattr(instance, method_name)()
                    passed += 1
                    print(f"PASSED: {test_class.__name__}.{method_name}")
                except Exception as e:
                    failed += 1
                    errors.append((test_class.__name__, method_name, str(e)))
                    print(f"FAILED: {test_class.__name__}.{method_name}: {e}")

    # Run parametrized tests manually
    batch_sizes = [1, 2, 4, 8, 16, 32]

    test_batch = TestStudentActorCriticBatchSizes()
    for bs in batch_sizes:
        try:
            test_batch.test_act_various_batch_sizes(bs)
            passed += 1
            print(f"PASSED: TestStudentActorCriticBatchSizes.test_act_various_batch_sizes[{bs}]")
        except Exception as e:
            failed += 1
            errors.append(("TestStudentActorCriticBatchSizes", f"test_act_various_batch_sizes[{bs}]", str(e)))
            print(f"FAILED: TestStudentActorCriticBatchSizes.test_act_various_batch_sizes[{bs}]: {e}")

        try:
            test_batch.test_evaluate_various_batch_sizes(bs)
            passed += 1
            print(f"PASSED: TestStudentActorCriticBatchSizes.test_evaluate_various_batch_sizes[{bs}]")
        except Exception as e:
            failed += 1
            errors.append(("TestStudentActorCriticBatchSizes", f"test_evaluate_various_batch_sizes[{bs}]", str(e)))
            print(f"FAILED: TestStudentActorCriticBatchSizes.test_evaluate_various_batch_sizes[{bs}]: {e}")

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for cls, method, err in errors:
            print(f"  - {cls}.{method}: {err}")
    print(f"{'='*60}")
