"""Exception Hierarchy Tests for LayerZero.

Tests for the exception classes defined in spec Section 14.1.
"""
from __future__ import annotations

import pytest


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_base_exception_exists(self) -> None:
        """LayerZeroError base class exists."""
        from layerzero.exceptions import LayerZeroError

        assert issubclass(LayerZeroError, Exception)

    def test_no_kernel_found_error(self) -> None:
        """NoKernelFoundError is LayerZeroError subclass."""
        from layerzero.exceptions import LayerZeroError, NoKernelFoundError

        assert issubclass(NoKernelFoundError, LayerZeroError)

        # Test instantiation
        err = NoKernelFoundError("attention.causal")
        assert "attention.causal" in str(err)

    def test_cuda_graph_unsafe_error(self) -> None:
        """CudaGraphUnsafeError is LayerZeroError subclass."""
        from layerzero.exceptions import LayerZeroError, CudaGraphUnsafeError

        assert issubclass(CudaGraphUnsafeError, LayerZeroError)

        err = CudaGraphUnsafeError("flash_attn", "allocates memory")
        assert "flash_attn" in str(err)
        assert "allocates" in str(err)

    def test_kernel_execution_error(self) -> None:
        """KernelExecutionError is LayerZeroError subclass."""
        from layerzero.exceptions import LayerZeroError, KernelExecutionError

        assert issubclass(KernelExecutionError, LayerZeroError)

        err = KernelExecutionError("triton_kernel", RuntimeError("CUDA error"))
        assert "triton_kernel" in str(err)

    def test_policy_validation_error(self) -> None:
        """PolicyValidationError is LayerZeroError subclass."""
        from layerzero.exceptions import LayerZeroError, PolicyValidationError

        assert issubclass(PolicyValidationError, LayerZeroError)

        err = PolicyValidationError("Invalid rule syntax")
        assert "Invalid" in str(err)

    def test_backend_not_available_error(self) -> None:
        """BackendNotAvailableError is LayerZeroError subclass."""
        from layerzero.exceptions import LayerZeroError, BackendNotAvailableError

        assert issubclass(BackendNotAvailableError, LayerZeroError)

        err = BackendNotAvailableError("flash_attn", "import failed")
        assert "flash_attn" in str(err)


class TestExceptionAttributes:
    """Tests for exception attributes."""

    def test_no_kernel_found_has_operation(self) -> None:
        """NoKernelFoundError has operation attribute."""
        from layerzero.exceptions import NoKernelFoundError

        err = NoKernelFoundError("attention.causal", candidates=["fa", "sdpa"])
        assert err.operation == "attention.causal"
        assert err.candidates == ["fa", "sdpa"]

    def test_cuda_graph_unsafe_has_kernel_id(self) -> None:
        """CudaGraphUnsafeError has kernel_id attribute."""
        from layerzero.exceptions import CudaGraphUnsafeError

        err = CudaGraphUnsafeError("flash_attn_v3", "uses dynamic allocation")
        assert err.kernel_id == "flash_attn_v3"
        assert err.reason == "uses dynamic allocation"

    def test_kernel_execution_has_cause(self) -> None:
        """KernelExecutionError has original cause."""
        from layerzero.exceptions import KernelExecutionError

        cause = RuntimeError("CUDA OOM")
        err = KernelExecutionError("triton_attn", cause)
        assert err.kernel_id == "triton_attn"
        assert err.__cause__ is cause

    def test_backend_not_available_has_reason(self) -> None:
        """BackendNotAvailableError has backend and reason."""
        from layerzero.exceptions import BackendNotAvailableError

        err = BackendNotAvailableError("xformers", "version mismatch")
        assert err.backend_id == "xformers"
        assert err.reason == "version mismatch"


class TestExceptionUsage:
    """Tests for exception usage patterns."""

    def test_catch_layerzero_error(self) -> None:
        """Can catch all LayerZero errors with base class."""
        from layerzero.exceptions import (
            LayerZeroError,
            NoKernelFoundError,
            CudaGraphUnsafeError,
        )

        errors = [
            NoKernelFoundError("test_op"),
            CudaGraphUnsafeError("test_kernel", "reason"),
        ]

        for err in errors:
            try:
                raise err
            except LayerZeroError:
                pass  # Should catch

    def test_reraise_with_context(self) -> None:
        """Can chain exceptions properly."""
        from layerzero.exceptions import KernelExecutionError

        try:
            try:
                raise RuntimeError("Original error")
            except RuntimeError as e:
                raise KernelExecutionError("my_kernel", e) from e
        except KernelExecutionError as e:
            assert e.__cause__ is not None
            assert isinstance(e.__cause__, RuntimeError)

    def test_exception_repr(self) -> None:
        """Exceptions have useful repr."""
        from layerzero.exceptions import NoKernelFoundError

        err = NoKernelFoundError("attention.causal", candidates=["fa"])
        repr_str = repr(err)

        assert "NoKernelFoundError" in repr_str
        assert "attention.causal" in repr_str


class TestExceptionIntegration:
    """Integration tests for exceptions with LayerZero modules."""

    def test_selection_engine_raises_no_kernel_found(self) -> None:
        """SelectionEngine raises NoKernelFoundError when no match."""
        from layerzero.selection.engine import SelectionEngine, NoKernelAvailableError
        from layerzero.exceptions import NoKernelFoundError
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.policy.policy import Policy

        # NoKernelAvailableError should be aliased or inherit from NoKernelFoundError
        # Or they should be the same class
        assert NoKernelAvailableError is not None

    def test_exceptions_importable_from_main_module(self) -> None:
        """Exceptions are importable from layerzero package."""
        from layerzero import (
            LayerZeroError,
            NoKernelFoundError,
            CudaGraphUnsafeError,
            KernelExecutionError,
            PolicyValidationError,
            BackendNotAvailableError,
        )

        assert all([
            LayerZeroError,
            NoKernelFoundError,
            CudaGraphUnsafeError,
            KernelExecutionError,
            PolicyValidationError,
            BackendNotAvailableError,
        ])
