"""Unit tests for ValueHead module.

TDD RED phase: These tests are written BEFORE implementation.
All tests should FAIL initially until ValueHead is implemented.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
from torch import nn


def _ensure_modules_on_path() -> None:
    """Ensure the project root containing 'modules' is available on sys.path."""
    file_dir = os.path.dirname(os.path.abspath(__file__))
    curr = file_dir
    while True:
        modules_dir = os.path.join(curr, "modules")
        if os.path.isdir(modules_dir):
            if curr not in sys.path:
                sys.path.insert(0, curr)
            return
        parent = os.path.dirname(curr)
        if parent == curr:
            raise RuntimeError("Unable to locate a project root containing 'modules'.")
        curr = parent


_ensure_modules_on_path()

from modules.actionheads.value_head import ValueHead


class TestValueHeadForwardStepShape:
    """Test forward_step output shape [B, 1]."""

    def test_basic_shape(self) -> None:
        """Verify output shape is [B, 1] for single timestep input."""
        d_model = 128
        batch_size = 4
        value_head = ValueHead(d_model=d_model)

        h = torch.randn(batch_size, d_model)
        output = value_head.forward_step(h)

        assert output.shape == (batch_size, 1), (
            f"Expected shape ({batch_size}, 1), got {tuple(output.shape)}"
        )

    def test_output_is_finite(self) -> None:
        """Verify output contains no NaN or Inf values."""
        d_model = 128
        batch_size = 4
        value_head = ValueHead(d_model=d_model)

        h = torch.randn(batch_size, d_model)
        output = value_head.forward_step(h)

        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"


class TestValueHeadForwardSequenceShape:
    """Test forward_sequence output shape [B, S, 1]."""

    def test_basic_shape(self) -> None:
        """Verify output shape is [B, S, 1] for sequence input."""
        d_model = 128
        batch_size = 4
        seq_len = 8
        value_head = ValueHead(d_model=d_model)

        h_seq = torch.randn(batch_size, seq_len, d_model)
        output = value_head.forward_sequence(h_seq)

        assert output.shape == (batch_size, seq_len, 1), (
            f"Expected shape ({batch_size}, {seq_len}, 1), got {tuple(output.shape)}"
        )

    def test_output_is_finite(self) -> None:
        """Verify output contains no NaN or Inf values."""
        d_model = 128
        batch_size = 4
        seq_len = 8
        value_head = ValueHead(d_model=d_model)

        h_seq = torch.randn(batch_size, seq_len, d_model)
        output = value_head.forward_sequence(h_seq)

        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    def test_consistency_with_forward_step(self) -> None:
        """Verify forward_sequence last step matches forward_step."""
        d_model = 128
        batch_size = 4
        seq_len = 8
        value_head = ValueHead(d_model=d_model)

        h_seq = torch.randn(batch_size, seq_len, d_model)
        seq_output = value_head.forward_sequence(h_seq)
        step_output = value_head.forward_step(h_seq[:, -1, :])

        assert torch.allclose(seq_output[:, -1, :], step_output, atol=1e-6), (
            "forward_sequence last step does not match forward_step output"
        )


class TestValueHeadGradientFlow:
    """Test that gradients flow through all layers."""

    def test_gradient_flow_forward_step(self) -> None:
        """Verify gradients flow through all layers in forward_step."""
        d_model = 128
        batch_size = 4
        value_head = ValueHead(d_model=d_model)

        h = torch.randn(batch_size, d_model, requires_grad=True)
        output = value_head.forward_step(h)
        loss = output.sum()
        loss.backward()

        # Check input gradient exists
        assert h.grad is not None, "No gradient on input tensor"
        assert torch.isfinite(h.grad).all(), "Non-finite gradient on input"

        # Check all parameters have gradients
        for name, param in value_head.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient for parameter {name}"
            )

    def test_gradient_flow_forward_sequence(self) -> None:
        """Verify gradients flow through all layers in forward_sequence."""
        d_model = 128
        batch_size = 4
        seq_len = 8
        value_head = ValueHead(d_model=d_model)

        h_seq = torch.randn(batch_size, seq_len, d_model, requires_grad=True)
        output = value_head.forward_sequence(h_seq)
        loss = output.sum()
        loss.backward()

        # Check input gradient exists
        assert h_seq.grad is not None, "No gradient on input tensor"
        assert torch.isfinite(h_seq.grad).all(), "Non-finite gradient on input"

        # Check all parameters have gradients
        for name, param in value_head.named_parameters():
            assert param.grad is not None, f"No gradient for parameter {name}"
            assert torch.isfinite(param.grad).all(), (
                f"Non-finite gradient for parameter {name}"
            )

    def test_nonzero_gradients(self) -> None:
        """Verify gradients are non-zero (network is learning)."""
        d_model = 128
        batch_size = 4
        value_head = ValueHead(d_model=d_model)

        h = torch.randn(batch_size, d_model, requires_grad=True)
        output = value_head.forward_step(h)
        loss = output.sum()
        loss.backward()

        nonzero_grads = 0
        for param in value_head.parameters():
            if param.grad is not None and param.grad.abs().sum() > 0:
                nonzero_grads += 1

        assert nonzero_grads > 0, "All gradients are zero"


class TestValueHeadInitialization:
    """Test proper weight initialization."""

    def test_has_expected_layers(self) -> None:
        """Verify ValueHead has the expected MLP structure."""
        d_model = 128
        hidden_dims = (256, 256)
        value_head = ValueHead(d_model=d_model, hidden_dims=hidden_dims)

        # Count linear layers
        linear_layers = [
            m for m in value_head.modules() if isinstance(m, nn.Linear)
        ]
        # Expected: len(hidden_dims) + 1 (for output layer)
        expected_linear_count = len(hidden_dims) + 1
        assert len(linear_layers) == expected_linear_count, (
            f"Expected {expected_linear_count} Linear layers, got {len(linear_layers)}"
        )

    def test_layer_dimensions(self) -> None:
        """Verify layer dimensions match architecture spec."""
        d_model = 128
        hidden_dims = (256, 256)
        value_head = ValueHead(d_model=d_model, hidden_dims=hidden_dims)

        linear_layers = [
            m for m in value_head.modules() if isinstance(m, nn.Linear)
        ]

        # First layer: d_model -> hidden_dims[0]
        assert linear_layers[0].in_features == d_model, (
            f"First layer input should be {d_model}, got {linear_layers[0].in_features}"
        )
        assert linear_layers[0].out_features == hidden_dims[0], (
            f"First layer output should be {hidden_dims[0]}, got {linear_layers[0].out_features}"
        )

        # Second layer: hidden_dims[0] -> hidden_dims[1]
        assert linear_layers[1].in_features == hidden_dims[0], (
            f"Second layer input should be {hidden_dims[0]}, got {linear_layers[1].in_features}"
        )
        assert linear_layers[1].out_features == hidden_dims[1], (
            f"Second layer output should be {hidden_dims[1]}, got {linear_layers[1].out_features}"
        )

        # Output layer: hidden_dims[-1] -> 1
        assert linear_layers[-1].in_features == hidden_dims[-1], (
            f"Output layer input should be {hidden_dims[-1]}, got {linear_layers[-1].in_features}"
        )
        assert linear_layers[-1].out_features == 1, (
            f"Output layer output should be 1, got {linear_layers[-1].out_features}"
        )

    def test_weights_are_finite(self) -> None:
        """Verify all weights are finite after initialization."""
        d_model = 128
        value_head = ValueHead(d_model=d_model)

        for name, param in value_head.named_parameters():
            assert torch.isfinite(param).all(), (
                f"Parameter {name} contains non-finite values"
            )

    def test_custom_hidden_dims(self) -> None:
        """Verify custom hidden_dims are respected."""
        d_model = 64
        hidden_dims = (128, 64, 32)
        value_head = ValueHead(d_model=d_model, hidden_dims=hidden_dims)

        linear_layers = [
            m for m in value_head.modules() if isinstance(m, nn.Linear)
        ]

        expected_linear_count = len(hidden_dims) + 1
        assert len(linear_layers) == expected_linear_count, (
            f"Expected {expected_linear_count} Linear layers, got {len(linear_layers)}"
        )


class TestValueHeadDifferentBatchSizes:
    """Test with various batch sizes."""

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
    def test_forward_step_various_batch_sizes(self, batch_size: int) -> None:
        """Verify forward_step works with various batch sizes."""
        d_model = 128
        value_head = ValueHead(d_model=d_model)

        h = torch.randn(batch_size, d_model)
        output = value_head.forward_step(h)

        assert output.shape == (batch_size, 1), (
            f"Expected shape ({batch_size}, 1), got {tuple(output.shape)}"
        )
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    @pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16, 32, 64])
    def test_forward_sequence_various_batch_sizes(self, batch_size: int) -> None:
        """Verify forward_sequence works with various batch sizes."""
        d_model = 128
        seq_len = 8
        value_head = ValueHead(d_model=d_model)

        h_seq = torch.randn(batch_size, seq_len, d_model)
        output = value_head.forward_sequence(h_seq)

        assert output.shape == (batch_size, seq_len, 1), (
            f"Expected shape ({batch_size}, {seq_len}, 1), got {tuple(output.shape)}"
        )
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"

    @pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32])
    def test_forward_sequence_various_seq_lengths(self, seq_len: int) -> None:
        """Verify forward_sequence works with various sequence lengths."""
        d_model = 128
        batch_size = 4
        value_head = ValueHead(d_model=d_model)

        h_seq = torch.randn(batch_size, seq_len, d_model)
        output = value_head.forward_sequence(h_seq)

        assert output.shape == (batch_size, seq_len, 1), (
            f"Expected shape ({batch_size}, {seq_len}, 1), got {tuple(output.shape)}"
        )
        assert torch.isfinite(output).all(), "Output contains NaN or Inf values"


class TestValueHeadForwardDispatch:
    """Test the forward method dispatches correctly based on input rank."""

    def test_forward_dispatches_to_step(self) -> None:
        """Verify forward with 2D input calls forward_step."""
        d_model = 128
        batch_size = 4
        value_head = ValueHead(d_model=d_model)

        h = torch.randn(batch_size, d_model)
        output_forward = value_head.forward(h)
        output_step = value_head.forward_step(h)

        assert torch.allclose(output_forward, output_step, atol=1e-6), (
            "forward(2D) should match forward_step"
        )

    def test_forward_dispatches_to_sequence(self) -> None:
        """Verify forward with 3D input calls forward_sequence."""
        d_model = 128
        batch_size = 4
        seq_len = 8
        value_head = ValueHead(d_model=d_model)

        h_seq = torch.randn(batch_size, seq_len, d_model)
        output_forward = value_head.forward(h_seq)
        output_seq = value_head.forward_sequence(h_seq)

        assert torch.allclose(output_forward, output_seq, atol=1e-6), (
            "forward(3D) should match forward_sequence"
        )

    def test_forward_raises_on_invalid_rank(self) -> None:
        """Verify forward raises ValueError for invalid input rank."""
        d_model = 128
        value_head = ValueHead(d_model=d_model)

        # 1D tensor should raise
        h_1d = torch.randn(d_model)
        with pytest.raises(ValueError, match="rank 2 or 3"):
            value_head.forward(h_1d)

        # 4D tensor should raise
        h_4d = torch.randn(2, 4, 8, d_model)
        with pytest.raises(ValueError, match="rank 2 or 3"):
            value_head.forward(h_4d)


if __name__ == "__main__":
    # Simple test runner to avoid pytest plugin conflicts
    import traceback

    test_classes = [
        TestValueHeadForwardStepShape,
        TestValueHeadForwardSequenceShape,
        TestValueHeadGradientFlow,
        TestValueHeadInitialization,
        TestValueHeadForwardDispatch,
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
    batch_sizes = [1, 2, 4, 8, 16, 32, 64]
    seq_lens = [1, 2, 4, 8, 16, 32]

    test_batch = TestValueHeadDifferentBatchSizes()
    for bs in batch_sizes:
        try:
            test_batch.test_forward_step_various_batch_sizes(bs)
            passed += 1
            print(f"PASSED: TestValueHeadDifferentBatchSizes.test_forward_step_various_batch_sizes[{bs}]")
        except Exception as e:
            failed += 1
            errors.append(("TestValueHeadDifferentBatchSizes", f"test_forward_step_various_batch_sizes[{bs}]", str(e)))
            print(f"FAILED: TestValueHeadDifferentBatchSizes.test_forward_step_various_batch_sizes[{bs}]: {e}")

        try:
            test_batch.test_forward_sequence_various_batch_sizes(bs)
            passed += 1
            print(f"PASSED: TestValueHeadDifferentBatchSizes.test_forward_sequence_various_batch_sizes[{bs}]")
        except Exception as e:
            failed += 1
            errors.append(("TestValueHeadDifferentBatchSizes", f"test_forward_sequence_various_batch_sizes[{bs}]", str(e)))
            print(f"FAILED: TestValueHeadDifferentBatchSizes.test_forward_sequence_various_batch_sizes[{bs}]: {e}")

    for sl in seq_lens:
        try:
            test_batch.test_forward_sequence_various_seq_lengths(sl)
            passed += 1
            print(f"PASSED: TestValueHeadDifferentBatchSizes.test_forward_sequence_various_seq_lengths[{sl}]")
        except Exception as e:
            failed += 1
            errors.append(("TestValueHeadDifferentBatchSizes", f"test_forward_sequence_various_seq_lengths[{sl}]", str(e)))
            print(f"FAILED: TestValueHeadDifferentBatchSizes.test_forward_sequence_various_seq_lengths[{sl}]: {e}")

    print(f"\n{'='*60}")
    print(f"Test Results: {passed} passed, {failed} failed")
    if errors:
        print("\nFailed tests:")
        for cls, method, err in errors:
            print(f"  - {cls}.{method}: {err}")
    print(f"{'='*60}")