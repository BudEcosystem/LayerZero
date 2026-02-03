"""Tests for Triton kernel registry."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.triton.registry import (
    TritonKernelRegistry,
    register_triton_kernel,
    get_registry,
)
from layerzero.backends.triton.version import is_triton_available
from layerzero.models.kernel_spec import KernelSpec


class TestTritonKernelRegistry:
    """Test Triton kernel registry."""

    def test_registry_is_singleton(self) -> None:
        """get_registry returns the same instance."""
        registry1 = get_registry()
        registry2 = get_registry()
        assert registry1 is registry2

    def test_registry_list_kernels_returns_list(self) -> None:
        """list_kernels returns a list."""
        registry = get_registry()
        result = registry.list_kernels()
        assert isinstance(result, list)

    def test_registry_get_returns_none_for_unknown(self) -> None:
        """get returns None for unknown kernel name."""
        registry = get_registry()
        result = registry.get("unknown_kernel_name_xyz")
        assert result is None


class TestTritonKernelRegistration:
    """Test kernel registration."""

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_register_simple_kernel(self) -> None:
        """Register a simple Triton kernel."""
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            y = tl.load(y_ptr + idx)
            tl.store(out_ptr + idx, x + y)

        spec = register_triton_kernel(
            name="test_add",
            kernel_fn=add_kernel,
            operation="math.add",
            supported_dtypes={torch.float16, torch.float32},
        )

        assert isinstance(spec, KernelSpec)
        assert "triton" in spec.kernel_id.lower()
        assert spec.operation == "math.add"

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_registered_kernel_retrievable(self) -> None:
        """Registered kernel can be retrieved."""
        import triton
        import triton.language as tl

        @triton.jit
        def mul_kernel(x_ptr, y_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            y = tl.load(y_ptr + idx)
            tl.store(out_ptr + idx, x * y)

        register_triton_kernel(
            name="test_mul",
            kernel_fn=mul_kernel,
            operation="math.mul",
            supported_dtypes={torch.float16},
        )

        registry = get_registry()
        adapter = registry.get("test_mul")
        assert adapter is not None

    def test_register_without_triton_raises(self) -> None:
        """Registering without Triton installed raises error."""
        if is_triton_available():
            pytest.skip("Triton is installed")

        def dummy_kernel():
            pass

        with pytest.raises((RuntimeError, ImportError)):
            register_triton_kernel(
                name="dummy",
                kernel_fn=dummy_kernel,
                operation="math.dummy",
                supported_dtypes={torch.float32},
            )


class TestKernelSpecGeneration:
    """Test KernelSpec generation from Triton kernels."""

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_kernel_spec_has_correct_source(self) -> None:
        """Generated KernelSpec has correct source."""
        import triton
        import triton.language as tl

        @triton.jit
        def spec_test_kernel(x_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            tl.store(out_ptr + idx, x)

        spec = register_triton_kernel(
            name="spec_test",
            kernel_fn=spec_test_kernel,
            operation="copy",
            supported_dtypes={torch.float16},
        )

        assert spec.source == "triton.custom"

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_kernel_spec_has_custom_dtypes(self) -> None:
        """Generated KernelSpec respects custom dtypes."""
        import triton
        import triton.language as tl

        @triton.jit
        def dtype_test_kernel(x_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            tl.store(out_ptr + idx, x)

        spec = register_triton_kernel(
            name="dtype_test",
            kernel_fn=dtype_test_kernel,
            operation="copy",
            supported_dtypes={torch.float16, torch.bfloat16},
        )

        assert torch.float16 in spec.supported_dtypes
        assert torch.bfloat16 in spec.supported_dtypes
        assert torch.float32 not in spec.supported_dtypes

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_kernel_spec_has_custom_priority(self) -> None:
        """Generated KernelSpec respects custom priority."""
        import triton
        import triton.language as tl

        @triton.jit
        def priority_test_kernel(x_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            tl.store(out_ptr + idx, x)

        spec = register_triton_kernel(
            name="priority_test",
            kernel_fn=priority_test_kernel,
            operation="copy",
            supported_dtypes={torch.float16},
            priority=80,
        )

        assert spec.priority == 80
