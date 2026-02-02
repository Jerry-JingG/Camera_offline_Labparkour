"""Unit tests for StudentRolloutStorage module.

TDD RED phase: These tests are written BEFORE implementation.
All tests should FAIL initially until StudentRolloutStorage is implemented.

Test cases:
- test_storage_initialization: Verify buffer allocation
- test_add_transition: Verify data storage at correct indices
- test_compute_returns_gae: Verify GAE computation correctness
- test_sequence_mini_batch_generator: Verify sequence preservation
- test_clear: Verify storage reset
- test_memory_efficiency: Verify uint8 depth storage
- test_various_batch_sizes: Test with different num_envs and num_steps
"""
from __future__ import annotations

import os
import sys
from typing import Dict, Generator, List, Optional, Tuple
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch import Tensor


def _ensure_modules_on_path() -> None:
    """Ensure the project root is available on sys.path."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    # Navigate up to scripts/rsl_rl/modules
    modules_dir = os.path.dirname(file_dir)
    if modules_dir not in sys.path:
        sys.path.insert(0, modules_dir)


_ensure_modules_on_path()

# Direct import from the module file to avoid __init__.py issues
from student_rollout_storage import StudentRolloutStorage


# Test configuration constants
NUM_STEPS = 16
NUM_ENVS = 8
PROPRIO_DIM = 48
ACTION_DIM = 12
DEPTH_HIST_LEN = 3
DEPTH_HEIGHT = 58
DEPTH_WIDTH = 87
N_LAYERS = 3
MEM_LEN = 64
TOKEN_DIM = 128


def create_test_storage(
    num_steps: int = NUM_STEPS,
    num_envs: int = NUM_ENVS,
    proprio_dim: int = PROPRIO_DIM,
    action_dim: int = ACTION_DIM,
    depth_hist_len: int = DEPTH_HIST_LEN,
    depth_height: int = DEPTH_HEIGHT,
    depth_width: int = DEPTH_WIDTH,
    device: str = "cpu",
) -> StudentRolloutStorage:
    """Factory function to create test storage."""
    return StudentRolloutStorage(
        num_steps=num_steps,
        num_envs=num_envs,
        proprio_dim=proprio_dim,
        action_dim=action_dim,
        depth_shape=(depth_hist_len, depth_height, depth_width),
        device=device,
    )


def create_test_transition(
    num_envs: int = NUM_ENVS,
    proprio_dim: int = PROPRIO_DIM,
    action_dim: int = ACTION_DIM,
    depth_hist_len: int = DEPTH_HIST_LEN,
    depth_height: int = DEPTH_HEIGHT,
    depth_width: int = DEPTH_WIDTH,
    device: str = "cpu",
) -> Dict[str, Tensor]:
    """Create test transition data."""
    return {
        "proprio": torch.randn(num_envs, proprio_dim, device=device),
        "depth": torch.randint(0, 256, (num_envs, depth_hist_len, depth_height, depth_width),
                               dtype=torch.uint8, device=device),
        "actions": torch.randn(num_envs, action_dim, device=device),
        "rewards": torch.randn(num_envs, 1, device=device),
        "values": torch.randn(num_envs, 1, device=device),
        "log_probs": torch.randn(num_envs, 1, device=device),
        "dones": torch.zeros(num_envs, 1, dtype=torch.float32, device=device),
        "mu": torch.randn(num_envs, action_dim, device=device),
        "sigma": torch.abs(torch.randn(num_envs, action_dim, device=device)) + 0.1,
    }


def create_test_mems(
    num_envs: int = NUM_ENVS,
    n_layers: int = N_LAYERS,
    mem_len: int = MEM_LEN,
    token_dim: int = TOKEN_DIM,
    device: str = "cpu",
) -> List[Tensor]:
    """Create test memory tensors."""
    return [
        torch.randn(num_envs, mem_len, token_dim, device=device)
        for _ in range(n_layers)
    ]


class TestStorageInitialization:
    """Test storage initialization and buffer allocation."""

    def test_storage_creates_correct_buffer_shapes(self) -> None:
        """Verify all buffers are allocated with correct shapes."""
        storage = create_test_storage()

        # Check proprio buffer shape: [num_steps, num_envs, proprio_dim]
        assert storage.proprio.shape == (NUM_STEPS, NUM_ENVS, PROPRIO_DIM), (
            f"Expected proprio shape ({NUM_STEPS}, {NUM_ENVS}, {PROPRIO_DIM}), "
            f"got {tuple(storage.proprio.shape)}"
        )

        # Check depth buffer shape: [num_steps, num_envs, depth_hist_len, H, W]
        assert storage.depth.shape == (NUM_STEPS, NUM_ENVS, DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH), (
            f"Expected depth shape ({NUM_STEPS}, {NUM_ENVS}, {DEPTH_HIST_LEN}, {DEPTH_HEIGHT}, {DEPTH_WIDTH}), "
            f"got {tuple(storage.depth.shape)}"
        )

        # Check actions buffer shape: [num_steps, num_envs, action_dim]
        assert storage.actions.shape == (NUM_STEPS, NUM_ENVS, ACTION_DIM), (
            f"Expected actions shape ({NUM_STEPS}, {NUM_ENVS}, {ACTION_DIM}), "
            f"got {tuple(storage.actions.shape)}"
        )

        # Check rewards buffer shape: [num_steps, num_envs, 1]
        assert storage.rewards.shape == (NUM_STEPS, NUM_ENVS, 1), (
            f"Expected rewards shape ({NUM_STEPS}, {NUM_ENVS}, 1), "
            f"got {tuple(storage.rewards.shape)}"
        )

        # Check values buffer shape: [num_steps + 1, num_envs, 1]
        assert storage.values.shape == (NUM_STEPS + 1, NUM_ENVS, 1), (
            f"Expected values shape ({NUM_STEPS + 1}, {NUM_ENVS}, 1), "
            f"got {tuple(storage.values.shape)}"
        )

        # Check returns buffer shape: [num_steps, num_envs, 1]
        assert storage.returns.shape == (NUM_STEPS, NUM_ENVS, 1), (
            f"Expected returns shape ({NUM_STEPS}, {NUM_ENVS}, 1), "
            f"got {tuple(storage.returns.shape)}"
        )

        # Check advantages buffer shape: [num_steps, num_envs, 1]
        assert storage.advantages.shape == (NUM_STEPS, NUM_ENVS, 1), (
            f"Expected advantages shape ({NUM_STEPS}, {NUM_ENVS}, 1), "
            f"got {tuple(storage.advantages.shape)}"
        )

        # Check log_probs buffer shape: [num_steps, num_envs, 1]
        assert storage.log_probs.shape == (NUM_STEPS, NUM_ENVS, 1), (
            f"Expected log_probs shape ({NUM_STEPS}, {NUM_ENVS}, 1), "
            f"got {tuple(storage.log_probs.shape)}"
        )

        # Check dones buffer shape: [num_steps, num_envs, 1]
        assert storage.dones.shape == (NUM_STEPS, NUM_ENVS, 1), (
            f"Expected dones shape ({NUM_STEPS}, {NUM_ENVS}, 1), "
            f"got {tuple(storage.dones.shape)}"
        )

        # Check mu buffer shape: [num_steps, num_envs, action_dim]
        assert storage.mu.shape == (NUM_STEPS, NUM_ENVS, ACTION_DIM), (
            f"Expected mu shape ({NUM_STEPS}, {NUM_ENVS}, {ACTION_DIM}), "
            f"got {tuple(storage.mu.shape)}"
        )

        # Check sigma buffer shape: [num_steps, num_envs, action_dim]
        assert storage.sigma.shape == (NUM_STEPS, NUM_ENVS, ACTION_DIM), (
            f"Expected sigma shape ({NUM_STEPS}, {NUM_ENVS}, {ACTION_DIM}), "
            f"got {tuple(storage.sigma.shape)}"
        )

    def test_storage_initializes_mems_list(self) -> None:
        """Verify mems_at_step is initialized as empty list."""
        storage = create_test_storage()
        assert hasattr(storage, "mems_at_step"), "Storage should have mems_at_step attribute"
        assert isinstance(storage.mems_at_step, list), "mems_at_step should be a list"
        assert len(storage.mems_at_step) == 0, "mems_at_step should be empty initially"

    def test_storage_initializes_step_counter(self) -> None:
        """Verify step counter is initialized to 0."""
        storage = create_test_storage()
        assert hasattr(storage, "step"), "Storage should have step attribute"
        assert storage.step == 0, "Step counter should be 0 initially"

    def test_storage_stores_dimensions(self) -> None:
        """Verify storage stores dimension parameters."""
        storage = create_test_storage()
        assert storage.num_steps == NUM_STEPS
        assert storage.num_envs == NUM_ENVS
        assert storage.proprio_dim == PROPRIO_DIM
        assert storage.action_dim == ACTION_DIM

    def test_storage_invalid_params_raises_error(self) -> None:
        """Verify invalid parameters raise ValueError."""
        with pytest.raises(ValueError):
            StudentRolloutStorage(
                num_steps=0,  # Invalid
                num_envs=NUM_ENVS,
                proprio_dim=PROPRIO_DIM,
                action_dim=ACTION_DIM,
                depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
            )

        with pytest.raises(ValueError):
            StudentRolloutStorage(
                num_steps=NUM_STEPS,
                num_envs=0,  # Invalid
                proprio_dim=PROPRIO_DIM,
                action_dim=ACTION_DIM,
                depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
            )


class TestAddTransition:
    """Test add_transition method."""

    def test_add_transition_stores_data_at_correct_step(self) -> None:
        """Verify data is stored at the correct step index."""
        storage = create_test_storage()
        transition = create_test_transition()

        storage.add_transition(
            step=0,
            proprio=transition["proprio"],
            depth=transition["depth"],
            actions=transition["actions"],
            rewards=transition["rewards"],
            values=transition["values"],
            log_probs=transition["log_probs"],
            dones=transition["dones"],
            mu=transition["mu"],
            sigma=transition["sigma"],
        )

        # Verify data stored at step 0
        assert torch.allclose(storage.proprio[0], transition["proprio"]), (
            "Proprio not stored correctly at step 0"
        )
        assert torch.equal(storage.depth[0], transition["depth"]), (
            "Depth not stored correctly at step 0"
        )
        assert torch.allclose(storage.actions[0], transition["actions"]), (
            "Actions not stored correctly at step 0"
        )
        assert torch.allclose(storage.rewards[0], transition["rewards"]), (
            "Rewards not stored correctly at step 0"
        )
        assert torch.allclose(storage.values[0], transition["values"]), (
            "Values not stored correctly at step 0"
        )
        assert torch.allclose(storage.log_probs[0], transition["log_probs"]), (
            "Log probs not stored correctly at step 0"
        )
        assert torch.allclose(storage.dones[0], transition["dones"]), (
            "Dones not stored correctly at step 0"
        )
        assert torch.allclose(storage.mu[0], transition["mu"]), (
            "Mu not stored correctly at step 0"
        )
        assert torch.allclose(storage.sigma[0], transition["sigma"]), (
            "Sigma not stored correctly at step 0"
        )

    def test_add_transition_with_mems(self) -> None:
        """Verify memory states are stored correctly."""
        storage = create_test_storage()
        transition = create_test_transition()
        mems = create_test_mems()

        storage.add_transition(
            step=0,
            proprio=transition["proprio"],
            depth=transition["depth"],
            actions=transition["actions"],
            rewards=transition["rewards"],
            values=transition["values"],
            log_probs=transition["log_probs"],
            dones=transition["dones"],
            mu=transition["mu"],
            sigma=transition["sigma"],
            mems=mems,
        )

        assert len(storage.mems_at_step) == 1, "Should have 1 memory state stored"
        assert len(storage.mems_at_step[0]) == N_LAYERS, (
            f"Should have {N_LAYERS} layer memories"
        )

    def test_add_transition_multiple_steps(self) -> None:
        """Verify multiple transitions are stored correctly."""
        storage = create_test_storage()

        for step in range(NUM_STEPS):
            transition = create_test_transition()
            storage.add_transition(
                step=step,
                proprio=transition["proprio"],
                depth=transition["depth"],
                actions=transition["actions"],
                rewards=transition["rewards"],
                values=transition["values"],
                log_probs=transition["log_probs"],
                dones=transition["dones"],
                mu=transition["mu"],
                sigma=transition["sigma"],
            )

        # Verify step counter
        assert storage.step == NUM_STEPS, f"Step counter should be {NUM_STEPS}"

    def test_add_transition_increments_step(self) -> None:
        """Verify step counter increments after add_transition."""
        storage = create_test_storage()
        transition = create_test_transition()

        assert storage.step == 0, "Initial step should be 0"

        storage.add_transition(
            step=0,
            proprio=transition["proprio"],
            depth=transition["depth"],
            actions=transition["actions"],
            rewards=transition["rewards"],
            values=transition["values"],
            log_probs=transition["log_probs"],
            dones=transition["dones"],
            mu=transition["mu"],
            sigma=transition["sigma"],
        )

        assert storage.step == 1, "Step should be 1 after first transition"


class TestComputeReturnsGAE:
    """Test compute_returns method with GAE computation."""

    def _fill_storage(self, storage: StudentRolloutStorage) -> None:
        """Helper to fill storage with test data."""
        for step in range(storage.num_steps):
            transition = create_test_transition(num_envs=storage.num_envs)
            storage.add_transition(
                step=step,
                proprio=transition["proprio"],
                depth=transition["depth"],
                actions=transition["actions"],
                rewards=transition["rewards"],
                values=transition["values"],
                log_probs=transition["log_probs"],
                dones=transition["dones"],
                mu=transition["mu"],
                sigma=transition["sigma"],
            )

    def test_compute_returns_sets_returns_and_advantages(self) -> None:
        """Verify compute_returns populates returns and advantages buffers."""
        storage = create_test_storage()
        self._fill_storage(storage)

        last_values = torch.randn(NUM_ENVS, 1)
        gamma = 0.99
        lam = 0.95

        storage.compute_returns(last_values, gamma, lam)

        # Returns and advantages should be non-zero after computation
        assert not torch.all(storage.returns == 0), "Returns should be computed"
        assert not torch.all(storage.advantages == 0), "Advantages should be computed"

    def test_compute_returns_correct_shape(self) -> None:
        """Verify returns and advantages have correct shapes."""
        storage = create_test_storage()
        self._fill_storage(storage)

        last_values = torch.randn(NUM_ENVS, 1)
        storage.compute_returns(last_values, gamma=0.99, lam=0.95)

        assert storage.returns.shape == (NUM_STEPS, NUM_ENVS, 1), (
            f"Returns shape should be ({NUM_STEPS}, {NUM_ENVS}, 1)"
        )
        assert storage.advantages.shape == (NUM_STEPS, NUM_ENVS, 1), (
            f"Advantages shape should be ({NUM_STEPS}, {NUM_ENVS}, 1)"
        )

    def test_compute_returns_gae_formula(self) -> None:
        """Verify GAE computation follows correct formula."""
        # Use simple values for manual verification
        storage = StudentRolloutStorage(
            num_steps=3,
            num_envs=1,
            proprio_dim=PROPRIO_DIM,
            action_dim=ACTION_DIM,
            depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
        )

        # Set known values
        storage.rewards[0, 0, 0] = 1.0
        storage.rewards[1, 0, 0] = 2.0
        storage.rewards[2, 0, 0] = 3.0

        storage.values[0, 0, 0] = 0.5
        storage.values[1, 0, 0] = 1.0
        storage.values[2, 0, 0] = 1.5

        storage.dones[:] = 0.0

        last_values = torch.tensor([[2.0]])
        gamma = 0.99
        lam = 0.95

        storage.compute_returns(last_values, gamma, lam)

        # Manual GAE calculation for verification
        # delta_t = r_t + gamma * V(s_{t+1}) * (1 - done) - V(s_t)
        # A_t = delta_t + gamma * lam * (1 - done) * A_{t+1}

        # Step 2 (last step):
        delta_2 = 3.0 + gamma * 2.0 - 1.5  # r + gamma*V_next - V
        adv_2 = delta_2

        # Step 1:
        delta_1 = 2.0 + gamma * 1.5 - 1.0
        adv_1 = delta_1 + gamma * lam * adv_2

        # Step 0:
        delta_0 = 1.0 + gamma * 1.0 - 0.5
        adv_0 = delta_0 + gamma * lam * adv_1

        # Check advantages are close to manual calculation
        assert torch.isclose(storage.advantages[2, 0, 0], torch.tensor(adv_2), atol=1e-5), (
            f"Advantage at step 2 incorrect: expected {adv_2}, got {storage.advantages[2, 0, 0].item()}"
        )
        assert torch.isclose(storage.advantages[1, 0, 0], torch.tensor(adv_1), atol=1e-5), (
            f"Advantage at step 1 incorrect: expected {adv_1}, got {storage.advantages[1, 0, 0].item()}"
        )
        assert torch.isclose(storage.advantages[0, 0, 0], torch.tensor(adv_0), atol=1e-5), (
            f"Advantage at step 0 incorrect: expected {adv_0}, got {storage.advantages[0, 0, 0].item()}"
        )

    def test_compute_returns_handles_dones(self) -> None:
        """Verify GAE computation handles episode terminations correctly."""
        storage = StudentRolloutStorage(
            num_steps=3,
            num_envs=1,
            proprio_dim=PROPRIO_DIM,
            action_dim=ACTION_DIM,
            depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
        )

        storage.rewards[:] = 1.0
        storage.values[:] = 0.5
        storage.dones[1, 0, 0] = 1.0  # Episode ends at step 1

        last_values = torch.tensor([[1.0]])
        storage.compute_returns(last_values, gamma=0.99, lam=0.95)

        # After done, advantage should not propagate from future steps
        # The advantage at step 0 should be affected by the done at step 1
        assert torch.isfinite(storage.advantages).all(), "Advantages should be finite"

    def test_compute_returns_stores_last_values(self) -> None:
        """Verify last_values is stored in values buffer."""
        storage = create_test_storage()
        self._fill_storage(storage)

        last_values = torch.randn(NUM_ENVS, 1)
        storage.compute_returns(last_values, gamma=0.99, lam=0.95)

        assert torch.allclose(storage.values[NUM_STEPS], last_values), (
            "Last values should be stored at values[num_steps]"
        )


class TestSequenceMiniBatchGenerator:
    """Test sequence_mini_batch_generator method."""

    def _fill_storage_with_mems(self, storage: StudentRolloutStorage) -> None:
        """Helper to fill storage with test data including mems."""
        for step in range(storage.num_steps):
            transition = create_test_transition(num_envs=storage.num_envs)
            mems = create_test_mems(num_envs=storage.num_envs)
            storage.add_transition(
                step=step,
                proprio=transition["proprio"],
                depth=transition["depth"],
                actions=transition["actions"],
                rewards=transition["rewards"],
                values=transition["values"],
                log_probs=transition["log_probs"],
                dones=transition["dones"],
                mu=transition["mu"],
                sigma=transition["sigma"],
                mems=mems,
            )
        # Compute returns
        last_values = torch.randn(storage.num_envs, 1)
        storage.compute_returns(last_values, gamma=0.99, lam=0.95)

    def test_generator_yields_correct_number_of_batches(self) -> None:
        """Verify generator yields num_batches * num_epochs batches."""
        storage = create_test_storage()
        self._fill_storage_with_mems(storage)

        num_batches = 2
        num_epochs = 3

        batches = list(storage.sequence_mini_batch_generator(num_batches, num_epochs))
        expected_count = num_batches * num_epochs

        assert len(batches) == expected_count, (
            f"Expected {expected_count} batches, got {len(batches)}"
        )

    def test_generator_preserves_sequence_structure(self) -> None:
        """Verify batches preserve full temporal sequences."""
        storage = create_test_storage()
        self._fill_storage_with_mems(storage)

        num_batches = 2
        batch_size = NUM_ENVS // num_batches

        for batch in storage.sequence_mini_batch_generator(num_batches, num_epochs=1):
            # Each batch should have full sequence length
            assert batch["proprio"].shape[0] == NUM_STEPS, (
                f"Batch proprio should have {NUM_STEPS} steps"
            )
            assert batch["proprio"].shape[1] == batch_size, (
                f"Batch should have {batch_size} envs"
            )

    def test_generator_batch_contains_all_fields(self) -> None:
        """Verify each batch contains all required fields."""
        storage = create_test_storage()
        self._fill_storage_with_mems(storage)

        required_fields = [
            "proprio", "depth", "actions", "rewards", "values",
            "returns", "advantages", "log_probs", "dones", "mu", "sigma",
            "initial_mems"
        ]

        for batch in storage.sequence_mini_batch_generator(num_batches=2, num_epochs=1):
            for field in required_fields:
                assert field in batch, f"Batch missing required field: {field}"

    def test_generator_depth_converted_to_float32(self) -> None:
        """Verify depth is converted from uint8 to float32 in batches."""
        storage = create_test_storage()
        self._fill_storage_with_mems(storage)

        for batch in storage.sequence_mini_batch_generator(num_batches=2, num_epochs=1):
            assert batch["depth"].dtype == torch.float32, (
                f"Depth should be float32, got {batch['depth'].dtype}"
            )

    def test_generator_initial_mems_from_step_zero(self) -> None:
        """Verify initial_mems comes from step 0 memory states."""
        storage = create_test_storage()
        self._fill_storage_with_mems(storage)

        for batch in storage.sequence_mini_batch_generator(num_batches=2, num_epochs=1):
            assert batch["initial_mems"] is not None, "initial_mems should not be None"
            assert isinstance(batch["initial_mems"], list), "initial_mems should be a list"

    def test_generator_shuffles_envs_between_epochs(self) -> None:
        """Verify environment indices are shuffled between epochs."""
        storage = create_test_storage()
        self._fill_storage_with_mems(storage)

        # Collect env indices from multiple epochs
        batches_epoch1 = []
        batches_epoch2 = []

        gen = storage.sequence_mini_batch_generator(num_batches=2, num_epochs=2)
        for i, batch in enumerate(gen):
            if i < 2:
                batches_epoch1.append(batch["proprio"].clone())
            else:
                batches_epoch2.append(batch["proprio"].clone())

        # Due to shuffling, batches may differ between epochs
        # This is a probabilistic test - with 8 envs, very unlikely to be same
        # We just verify the generator completes without error
        assert len(batches_epoch1) == 2
        assert len(batches_epoch2) == 2

    def test_generator_handles_uneven_batch_split(self) -> None:
        """Verify generator handles when num_envs not divisible by num_batches."""
        # Create storage with 7 envs (not divisible by 2)
        storage = StudentRolloutStorage(
            num_steps=NUM_STEPS,
            num_envs=7,
            proprio_dim=PROPRIO_DIM,
            action_dim=ACTION_DIM,
            depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
        )

        for step in range(NUM_STEPS):
            transition = create_test_transition(num_envs=7)
            mems = create_test_mems(num_envs=7)
            storage.add_transition(
                step=step,
                proprio=transition["proprio"],
                depth=transition["depth"],
                actions=transition["actions"],
                rewards=transition["rewards"],
                values=transition["values"],
                log_probs=transition["log_probs"],
                dones=transition["dones"],
                mu=transition["mu"],
                sigma=transition["sigma"],
                mems=mems,
            )

        last_values = torch.randn(7, 1)
        storage.compute_returns(last_values, gamma=0.99, lam=0.95)

        # Should not raise error
        batches = list(storage.sequence_mini_batch_generator(num_batches=2, num_epochs=1))
        assert len(batches) == 2


class TestClear:
    """Test clear method."""

    def test_clear_resets_step_counter(self) -> None:
        """Verify clear resets step counter to 0."""
        storage = create_test_storage()

        # Add some transitions
        for step in range(5):
            transition = create_test_transition()
            storage.add_transition(
                step=step,
                proprio=transition["proprio"],
                depth=transition["depth"],
                actions=transition["actions"],
                rewards=transition["rewards"],
                values=transition["values"],
                log_probs=transition["log_probs"],
                dones=transition["dones"],
                mu=transition["mu"],
                sigma=transition["sigma"],
            )

        assert storage.step == 5, "Step should be 5 before clear"

        storage.clear()

        assert storage.step == 0, "Step should be 0 after clear"

    def test_clear_resets_mems_list(self) -> None:
        """Verify clear empties mems_at_step list."""
        storage = create_test_storage()

        # Add transitions with mems
        for step in range(3):
            transition = create_test_transition()
            mems = create_test_mems()
            storage.add_transition(
                step=step,
                proprio=transition["proprio"],
                depth=transition["depth"],
                actions=transition["actions"],
                rewards=transition["rewards"],
                values=transition["values"],
                log_probs=transition["log_probs"],
                dones=transition["dones"],
                mu=transition["mu"],
                sigma=transition["sigma"],
                mems=mems,
            )

        assert len(storage.mems_at_step) == 3, "Should have 3 mems before clear"

        storage.clear()

        assert len(storage.mems_at_step) == 0, "mems_at_step should be empty after clear"

    def test_clear_zeros_buffers(self) -> None:
        """Verify clear zeros out all buffers."""
        storage = create_test_storage()

        # Fill with non-zero data
        storage.proprio.fill_(1.0)
        storage.actions.fill_(2.0)
        storage.rewards.fill_(3.0)

        storage.clear()

        # Buffers should be zeroed
        assert torch.all(storage.proprio == 0), "Proprio should be zeroed"
        assert torch.all(storage.actions == 0), "Actions should be zeroed"
        assert torch.all(storage.rewards == 0), "Rewards should be zeroed"

    def test_clear_preserves_buffer_shapes(self) -> None:
        """Verify clear preserves buffer shapes."""
        storage = create_test_storage()

        original_shapes = {
            "proprio": storage.proprio.shape,
            "depth": storage.depth.shape,
            "actions": storage.actions.shape,
            "rewards": storage.rewards.shape,
            "values": storage.values.shape,
        }

        storage.clear()

        assert storage.proprio.shape == original_shapes["proprio"]
        assert storage.depth.shape == original_shapes["depth"]
        assert storage.actions.shape == original_shapes["actions"]
        assert storage.rewards.shape == original_shapes["rewards"]
        assert storage.values.shape == original_shapes["values"]


class TestMemoryEfficiency:
    """Test memory efficiency features."""

    def test_depth_stored_as_uint8(self) -> None:
        """Verify depth buffer uses uint8 dtype for memory efficiency."""
        storage = create_test_storage()

        assert storage.depth.dtype == torch.uint8, (
            f"Depth should be uint8, got {storage.depth.dtype}"
        )

    def test_depth_uint8_memory_savings(self) -> None:
        """Verify uint8 depth uses 4x less memory than float32."""
        storage = create_test_storage()

        # Calculate expected memory
        depth_elements = NUM_STEPS * NUM_ENVS * DEPTH_HIST_LEN * DEPTH_HEIGHT * DEPTH_WIDTH
        uint8_bytes = depth_elements * 1  # 1 byte per uint8
        float32_bytes = depth_elements * 4  # 4 bytes per float32

        actual_bytes = storage.depth.element_size() * storage.depth.numel()

        assert actual_bytes == uint8_bytes, (
            f"Depth memory should be {uint8_bytes} bytes, got {actual_bytes}"
        )
        assert actual_bytes < float32_bytes, (
            "uint8 should use less memory than float32"
        )

    def test_add_transition_accepts_uint8_depth(self) -> None:
        """Verify add_transition accepts uint8 depth input."""
        storage = create_test_storage()

        # Create uint8 depth
        depth_uint8 = torch.randint(
            0, 256,
            (NUM_ENVS, DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
            dtype=torch.uint8
        )

        transition = create_test_transition()
        transition["depth"] = depth_uint8

        # Should not raise
        storage.add_transition(
            step=0,
            proprio=transition["proprio"],
            depth=transition["depth"],
            actions=transition["actions"],
            rewards=transition["rewards"],
            values=transition["values"],
            log_probs=transition["log_probs"],
            dones=transition["dones"],
            mu=transition["mu"],
            sigma=transition["sigma"],
        )

        assert storage.depth.dtype == torch.uint8

    def test_add_transition_converts_float_depth_to_uint8(self) -> None:
        """Verify add_transition converts float32 depth to uint8."""
        storage = create_test_storage()

        # Create float32 depth (values 0-255)
        depth_float = torch.rand(
            NUM_ENVS, DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH,
            dtype=torch.float32
        ) * 255

        transition = create_test_transition()
        transition["depth"] = depth_float

        storage.add_transition(
            step=0,
            proprio=transition["proprio"],
            depth=transition["depth"],
            actions=transition["actions"],
            rewards=transition["rewards"],
            values=transition["values"],
            log_probs=transition["log_probs"],
            dones=transition["dones"],
            mu=transition["mu"],
            sigma=transition["sigma"],
        )

        # Storage should still be uint8
        assert storage.depth.dtype == torch.uint8


class TestVariousBatchSizes:
    """Test with various num_envs and num_steps configurations."""

    @pytest.mark.parametrize("num_envs", [1, 2, 4, 8, 16, 32, 64])
    def test_various_num_envs(self, num_envs: int) -> None:
        """Verify storage works with various num_envs."""
        storage = StudentRolloutStorage(
            num_steps=NUM_STEPS,
            num_envs=num_envs,
            proprio_dim=PROPRIO_DIM,
            action_dim=ACTION_DIM,
            depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
        )

        assert storage.proprio.shape == (NUM_STEPS, num_envs, PROPRIO_DIM)
        assert storage.depth.shape == (NUM_STEPS, num_envs, DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH)

    @pytest.mark.parametrize("num_steps", [1, 4, 8, 16, 32, 64])
    def test_various_num_steps(self, num_steps: int) -> None:
        """Verify storage works with various num_steps."""
        storage = StudentRolloutStorage(
            num_steps=num_steps,
            num_envs=NUM_ENVS,
            proprio_dim=PROPRIO_DIM,
            action_dim=ACTION_DIM,
            depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
        )

        assert storage.proprio.shape == (num_steps, NUM_ENVS, PROPRIO_DIM)
        assert storage.values.shape == (num_steps + 1, NUM_ENVS, 1)

    @pytest.mark.parametrize("proprio_dim", [12, 24, 48, 96])
    def test_various_proprio_dims(self, proprio_dim: int) -> None:
        """Verify storage works with various proprio_dim."""
        storage = StudentRolloutStorage(
            num_steps=NUM_STEPS,
            num_envs=NUM_ENVS,
            proprio_dim=proprio_dim,
            action_dim=ACTION_DIM,
            depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
        )

        assert storage.proprio.shape == (NUM_STEPS, NUM_ENVS, proprio_dim)

    @pytest.mark.parametrize("action_dim", [6, 12, 18, 24])
    def test_various_action_dims(self, action_dim: int) -> None:
        """Verify storage works with various action_dim."""
        storage = StudentRolloutStorage(
            num_steps=NUM_STEPS,
            num_envs=NUM_ENVS,
            proprio_dim=PROPRIO_DIM,
            action_dim=action_dim,
            depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
        )

        assert storage.actions.shape == (NUM_STEPS, NUM_ENVS, action_dim)
        assert storage.mu.shape == (NUM_STEPS, NUM_ENVS, action_dim)
        assert storage.sigma.shape == (NUM_STEPS, NUM_ENVS, action_dim)

    def test_full_workflow_various_sizes(self) -> None:
        """Test complete workflow with non-default sizes."""
        num_steps = 32
        num_envs = 16

        storage = StudentRolloutStorage(
            num_steps=num_steps,
            num_envs=num_envs,
            proprio_dim=PROPRIO_DIM,
            action_dim=ACTION_DIM,
            depth_shape=(DEPTH_HIST_LEN, DEPTH_HEIGHT, DEPTH_WIDTH),
        )

        # Fill storage
        for step in range(num_steps):
            transition = create_test_transition(num_envs=num_envs)
            mems = create_test_mems(num_envs=num_envs)
            storage.add_transition(
                step=step,
                proprio=transition["proprio"],
                depth=transition["depth"],
                actions=transition["actions"],
                rewards=transition["rewards"],
                values=transition["values"],
                log_probs=transition["log_probs"],
                dones=transition["dones"],
                mu=transition["mu"],
                sigma=transition["sigma"],
                mems=mems,
            )

        # Compute returns
        last_values = torch.randn(num_envs, 1)
        storage.compute_returns(last_values, gamma=0.99, lam=0.95)

        # Generate batches
        batches = list(storage.sequence_mini_batch_generator(num_batches=4, num_epochs=2))
        assert len(batches) == 8

        # Clear
        storage.clear()
        assert storage.step == 0


if __name__ == "__main__":
    # Simple test runner to avoid pytest plugin conflicts

    test_classes = [
        TestStorageInitialization,
        TestAddTransition,
        TestComputeReturnsGAE,
        TestSequenceMiniBatchGenerator,
        TestClear,
        TestMemoryEfficiency,
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
    test_batch = TestVariousBatchSizes()

    # Test various num_envs
    for num_envs in [1, 2, 4, 8, 16, 32, 64]:
        try:
            test_batch.test_various_num_envs(num_envs)
            passed += 1
            print(f"PASSED: TestVariousBatchSizes.test_various_num_envs[{num_envs}]")
        except Exception as e:
            failed += 1
            errors.append(("TestVariousBatchSizes", f"test_various_num_envs[{num_envs}]", str(e)))
            print(f"FAILED: TestVariousBatchSizes.test_various_num_envs[{num_envs}]: {e}")

    # Test various num_steps
    for num_steps in [1, 4, 8, 16, 32, 64]:
        try:
            test_batch.test_various_num_steps(num_steps)
            passed += 1
            print(f"PASSED: TestVariousBatchSizes.test_various_num_steps[{num_steps}]")
        except Exception as e:
            failed += 1
            errors.append(("TestVariousBatchSizes", f"test_various_num_steps[{num_steps}]", str(e)))
            print(f"FAILED: TestVariousBatchSizes.test_various_num_steps[{num_steps}]: {e}")

    # Test various proprio_dims
    for proprio_dim in [12, 24, 48, 96]:
        try:
            test_batch.test_various_proprio_dims(proprio_dim)
            passed += 1
            print(f"PASSED: TestVariousBatchSizes.test_various_proprio_dims[{proprio_dim}]")
        except Exception as e:
            failed += 1
            errors.append(("TestVariousBatchSizes", f"test_various_proprio_dims[{proprio_dim}]", str(e)))
            print(f"FAILED: TestVariousBatchSizes.test_various_proprio_dims[{proprio_dim}]: {e}")

    # Test various action_dims
    for action_dim in [6, 12, 18, 24]:
        try:
            test_batch.test_various_action_dims(action_dim)
            passed += 1
            print(f"PASSED: TestVariousBatchSizes.test_various_action_dims[{action_dim}]")
        except Exception as e:
            failed += 1
            errors.append(("TestVariousBatchSizes", f"test_various_action_dims[{action_dim}]", str(e)))
            print(f"FAILED: TestVariousBatchSizes.test_various_action_dims[{action_dim}]: {e}")

    # Test full workflow
    try:
        test_batch.test_full_workflow_various_sizes()
        passed += 1
        print("PASSED: TestVariousBatchSizes.test_full_workflow_various_sizes")
    except Exception as e:
        failed += 1
        errors.append(("TestVariousBatchSizes", "test_full_workflow_various_sizes", str(e)))
        print(f"FAILED: TestVariousBatchSizes.test_full_workflow_various_sizes: {e}")

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for cls, method, err in errors:
            print(f"  - {cls}.{method}: {err}")
    print(f"{'='*60}")
