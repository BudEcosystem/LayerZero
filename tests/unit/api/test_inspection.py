"""Tests for LayerZero inspection APIs.

Tests for lz.select(), lz.explain(), lz.which(), lz.list_kernels(), etc.
"""
from __future__ import annotations

import pytest
import torch


class TestSelectAPI:
    """Tests for lz.select() public API."""

    def test_select_basic(self) -> None:
        """Basic kernel selection."""
        import layerzero as lz

        result = lz.select(
            operation="attention.causal",
            batch_size=2,
            seq_len=1024,
            num_heads=8,
            head_dim=64,
            dtype=torch.float16,
        )

        assert result is not None
        assert hasattr(result, 'kernel_id')
        assert hasattr(result, 'score')

    def test_select_returns_kernel_id(self) -> None:
        """Selection returns valid kernel ID."""
        import layerzero as lz

        result = lz.select(
            operation="attention.causal",
            batch_size=1,
            seq_len=512,
            num_heads=8,
            head_dim=64,
        )

        # Should return a string kernel ID
        assert isinstance(result.kernel_id, str)
        assert len(result.kernel_id) > 0

    def test_select_with_context(self) -> None:
        """Selection with full context."""
        import layerzero as lz

        result = lz.select(
            operation="attention.causal",
            batch_size=4,
            seq_len=2048,
            num_heads=32,
            head_dim=128,
            dtype=torch.bfloat16,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        assert result.kernel_id is not None

    def test_select_invalid_operation(self) -> None:
        """Selection with unknown operation returns fallback."""
        import layerzero as lz

        # LayerZero is lenient - returns fallback for unknown operations
        result = lz.select(operation="invalid.operation")

        # Should return fallback kernel (torch_sdpa)
        assert result.kernel_id == "torch_sdpa"
        assert "fallback" in result.reasons[0].lower() if result.reasons else True


class TestExplainAPI:
    """Tests for lz.explain() public API."""

    def test_explain_basic(self) -> None:
        """Basic selection explanation."""
        import layerzero as lz

        report = lz.explain(
            operation="attention.causal",
            batch_size=2,
            seq_len=1024,
            num_heads=8,
            head_dim=64,
        )

        assert report is not None
        assert hasattr(report, 'selected_kernel')
        assert hasattr(report, 'candidates')

    def test_explain_shows_candidates(self) -> None:
        """Explanation shows candidate kernels."""
        import layerzero as lz

        report = lz.explain(
            operation="attention.causal",
            batch_size=2,
            seq_len=1024,
            num_heads=8,
            head_dim=64,
        )

        # Should have list of candidates
        assert hasattr(report, 'candidates')
        assert isinstance(report.candidates, (list, tuple))

    def test_explain_shows_scores(self) -> None:
        """Explanation shows kernel scores."""
        import layerzero as lz

        report = lz.explain(
            operation="attention.causal",
            batch_size=2,
            seq_len=1024,
            num_heads=8,
            head_dim=64,
        )

        # Each candidate should have a score
        for candidate in report.candidates:
            assert hasattr(candidate, 'score') or 'score' in candidate

    def test_explain_shows_rejection_reasons(self) -> None:
        """Explanation shows why kernels were rejected."""
        import layerzero as lz

        report = lz.explain(
            operation="attention.causal",
            batch_size=2,
            seq_len=1024,
            num_heads=8,
            head_dim=64,
        )

        # Should have rejection reasons if any candidates rejected
        assert hasattr(report, 'rejected') or hasattr(report, 'rejection_reasons')


class TestWhichAPI:
    """Tests for lz.which() public API."""

    def test_which_basic(self) -> None:
        """Basic which() query."""
        import layerzero as lz

        kernel_id = lz.which("attention.causal")

        assert isinstance(kernel_id, str)

    def test_which_with_context(self) -> None:
        """which() with context parameters."""
        import layerzero as lz

        kernel_id = lz.which(
            "attention.causal",
            batch_size=4,
            seq_len=2048,
        )

        assert isinstance(kernel_id, str)

    def test_which_returns_current_default(self) -> None:
        """which() returns current default selection."""
        import layerzero as lz

        # Get current default
        kernel1 = lz.which("attention.causal")

        # Lock to different kernel
        lz.lock("attention.causal", "torch_sdpa")
        kernel2 = lz.which("attention.causal")

        assert kernel2 == "torch_sdpa"

        # Clean up
        lz.unlock("attention.causal")


class TestListKernelsAPI:
    """Tests for lz.list_kernels() public API."""

    def test_list_kernels_all(self) -> None:
        """List all available kernels."""
        import layerzero as lz

        kernels = lz.list_kernels()

        assert isinstance(kernels, list)
        assert len(kernels) > 0

    def test_list_kernels_for_operation(self) -> None:
        """List kernels for specific operation."""
        import layerzero as lz

        kernels = lz.list_kernels(operation="attention.causal")

        assert isinstance(kernels, list)
        # Should have at least torch SDPA
        assert len(kernels) > 0

    def test_list_kernels_returns_info(self) -> None:
        """Listed kernels have useful info."""
        import layerzero as lz

        kernels = lz.list_kernels()

        for kernel in kernels:
            # Each kernel should have id and operation
            assert hasattr(kernel, 'id') or 'id' in kernel
            assert hasattr(kernel, 'operation') or 'operation' in kernel


class TestValidateAPI:
    """Tests for lz.validate() public API."""

    def test_validate_basic(self) -> None:
        """Basic kernel validation."""
        import layerzero as lz

        is_valid = lz.validate(
            operation="attention.causal",
            kernel_id="torch_sdpa",
        )

        assert isinstance(is_valid, bool)

    def test_validate_with_context(self) -> None:
        """Validate kernel for specific context."""
        import layerzero as lz

        is_valid = lz.validate(
            operation="attention.causal",
            kernel_id="torch_sdpa",
            batch_size=2,
            seq_len=1024,
            dtype=torch.float16,
        )

        assert isinstance(is_valid, bool)

    def test_validate_invalid_kernel(self) -> None:
        """Validate returns False for invalid kernel."""
        import layerzero as lz

        is_valid = lz.validate(
            operation="attention.causal",
            kernel_id="nonexistent_kernel",
        )

        assert is_valid is False
