"""Tests for Triton kernel adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.triton.adapter import TritonKernelAdapter
from layerzero.backends.triton.version import is_triton_available
from layerzero.backends.base import BaseKernel
from layerzero.models.kernel_spec import KernelSpec
from layerzero.enums import Platform


class TestTritonKernelAdapter:
    """Test Triton kernel adapter base class."""

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_adapter_inherits_base_kernel(self) -> None:
        """Adapter inherits from BaseKernel."""
        import triton
        import triton.language as tl

        @triton.jit
        def dummy_kernel(x_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            tl.store(out_ptr + idx, x)

        adapter = TritonKernelAdapter(
            name="test_adapter",
            kernel_fn=dummy_kernel,
            operation="copy",
            supported_dtypes={torch.float16},
        )
        assert isinstance(adapter, BaseKernel)

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        import triton
        import triton.language as tl

        @triton.jit
        def spec_kernel(x_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            tl.store(out_ptr + idx, x)

        adapter = TritonKernelAdapter(
            name="test_spec",
            kernel_fn=spec_kernel,
            operation="copy",
            supported_dtypes={torch.float16},
        )
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_kernel_id_contains_name(self) -> None:
        """kernel_id contains the registered name."""
        import triton
        import triton.language as tl

        @triton.jit
        def name_kernel(x_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            tl.store(out_ptr + idx, x)

        adapter = TritonKernelAdapter(
            name="my_custom_kernel",
            kernel_fn=name_kernel,
            operation="copy",
            supported_dtypes={torch.float16},
        )
        spec = adapter.get_kernel_spec()
        assert "my_custom_kernel" in spec.kernel_id

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_source_is_triton_custom(self) -> None:
        """Source is 'triton.custom'."""
        import triton
        import triton.language as tl

        @triton.jit
        def source_kernel(x_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            tl.store(out_ptr + idx, x)

        adapter = TritonKernelAdapter(
            name="source_test",
            kernel_fn=source_kernel,
            operation="copy",
            supported_dtypes={torch.float16},
        )
        spec = adapter.get_kernel_spec()
        assert spec.source == "triton.custom"

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    def test_platform_is_cuda(self) -> None:
        """Platform is CUDA (or ROCm on AMD)."""
        import triton
        import triton.language as tl

        @triton.jit
        def platform_kernel(x_ptr, out_ptr, n: tl.constexpr):
            idx = tl.program_id(0)
            x = tl.load(x_ptr + idx)
            tl.store(out_ptr + idx, x)

        adapter = TritonKernelAdapter(
            name="platform_test",
            kernel_fn=platform_kernel,
            operation="copy",
            supported_dtypes={torch.float16},
        )
        spec = adapter.get_kernel_spec()
        assert spec.platform in (Platform.CUDA, Platform.ROCM)


class TestTritonKernelExecution:
    """Test Triton kernel execution."""

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_adapter_call_with_valid_input(
        self,
        sample_vectors: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Adapter callable with valid input."""
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(
            x_ptr, y_ptr, out_ptr,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        def grid_fn(meta):
            return (triton.cdiv(meta['n_elements'], meta['BLOCK_SIZE']),)

        adapter = TritonKernelAdapter(
            name="exec_test_add",
            kernel_fn=add_kernel,
            operation="math.add",
            supported_dtypes={torch.float16},
            grid_fn=grid_fn,
        )

        x, y = sample_vectors
        output = adapter(x, y, block_size=256)

        # Verify output
        assert output.shape == x.shape
        expected = x + y
        torch.testing.assert_close(output, expected, rtol=1e-2, atol=1e-2)

    @pytest.mark.skipif(
        not is_triton_available(),
        reason="Triton not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_adapter_correctness_fp16(
        self,
        device: torch.device,
    ) -> None:
        """Adapter produces correct results in fp16."""
        import triton
        import triton.language as tl

        @triton.jit
        def scale_kernel(
            x_ptr, out_ptr,
            scale,
            n_elements,
            BLOCK_SIZE: tl.constexpr,
        ):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x * scale, mask=mask)

        def grid_fn(meta):
            return (triton.cdiv(meta['n_elements'], meta['BLOCK_SIZE']),)

        adapter = TritonKernelAdapter(
            name="correctness_scale",
            kernel_fn=scale_kernel,
            operation="math.scale",
            supported_dtypes={torch.float16},
            grid_fn=grid_fn,
        )

        x = torch.randn(1024, device=device, dtype=torch.float16)
        scale = 2.5
        output = adapter(x, scale=scale, block_size=128)

        expected = x * scale
        max_diff = (output - expected).abs().max().item()
        assert max_diff < 0.01


class TestTritonKernelWithoutTriton:
    """Test behavior when Triton is not installed."""

    def test_adapter_creation_without_triton_raises(self) -> None:
        """Creating adapter without Triton raises error."""
        if is_triton_available():
            pytest.skip("Triton is installed")

        def dummy_kernel():
            pass

        with pytest.raises((RuntimeError, ImportError)):
            TritonKernelAdapter(
                name="dummy",
                kernel_fn=dummy_kernel,
                operation="dummy",
                supported_dtypes={torch.float32},
            )
