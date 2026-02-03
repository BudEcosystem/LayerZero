"""Tests for Intel Extension for PyTorch (IPEX) adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.cpu.ipex import (
    IPEXAttentionAdapter,
    IPEXMatmulAdapter,
    detect_ipex_version,
    is_ipex_available,
    is_xpu_available,
)
from layerzero.backends.base import BaseKernel
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec


class TestIPEXAvailability:
    """Test IPEX availability detection."""

    def test_is_ipex_available_returns_bool(self) -> None:
        """is_ipex_available returns boolean."""
        result = is_ipex_available()
        assert isinstance(result, bool)

    def test_is_xpu_available_returns_bool(self) -> None:
        """is_xpu_available returns boolean."""
        result = is_xpu_available()
        assert isinstance(result, bool)

    def test_detect_version_returns_tuple_or_none(self) -> None:
        """detect_ipex_version returns tuple or None."""
        result = detect_ipex_version()
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 3 and
            all(isinstance(x, int) for x in result)
        )

    def test_version_consistent_with_availability(self) -> None:
        """Version is only available when package is available."""
        available = is_ipex_available()
        version = detect_ipex_version()

        if not available:
            assert version is None


class TestIPEXMatmulAdapter:
    """Test IPEX matmul adapter."""

    def test_adapter_inherits_base_kernel(self) -> None:
        """IPEXMatmulAdapter inherits from BaseKernel."""
        adapter = IPEXMatmulAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = IPEXMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_spec_platform_is_cpu(self) -> None:
        """KernelSpec platform is CPU (or XPU if available)."""
        adapter = IPEXMatmulAdapter()
        spec = adapter.get_kernel_spec()
        # IPEX primarily targets CPU, but can also use XPU
        assert spec.platform in (Platform.CPU, Platform.XPU)

    def test_kernel_spec_source(self) -> None:
        """KernelSpec source is ipex."""
        adapter = IPEXMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "ipex"

    def test_kernel_spec_operation(self) -> None:
        """KernelSpec operation is matmul."""
        adapter = IPEXMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.operation == "matmul"

    def test_supported_dtypes(self) -> None:
        """Adapter supports expected dtypes."""
        adapter = IPEXMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float32 in spec.supported_dtypes

    @pytest.mark.skipif(
        not is_ipex_available(),
        reason="IPEX not available"
    )
    def test_matmul_execution(
        self,
        sample_cpu_matrices: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Matmul execution produces correct shape."""
        adapter = IPEXMatmulAdapter()
        a, b = sample_cpu_matrices

        result = adapter(a, b)

        # Result shape should be (batch, M, N)
        expected_shape = (a.shape[0], a.shape[1], b.shape[2])
        assert result.shape == expected_shape

    @pytest.mark.skipif(
        not is_ipex_available(),
        reason="IPEX not available"
    )
    @pytest.mark.correctness
    def test_matmul_correctness(
        self,
        sample_cpu_matrices: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """IPEX matmul matches PyTorch."""
        adapter = IPEXMatmulAdapter()
        a, b = sample_cpu_matrices

        result = adapter(a, b)
        expected = torch.bmm(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


class TestIPEXAttentionAdapter:
    """Test IPEX attention adapter."""

    def test_adapter_inherits_base_kernel(self) -> None:
        """IPEXAttentionAdapter inherits from BaseKernel."""
        adapter = IPEXAttentionAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = IPEXAttentionAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_spec_operation(self) -> None:
        """KernelSpec operation is attention."""
        adapter = IPEXAttentionAdapter()
        spec = adapter.get_kernel_spec()
        assert "attention" in spec.operation

    @pytest.mark.skipif(
        not is_ipex_available(),
        reason="IPEX not available"
    )
    def test_attention_execution(
        self,
        sample_cpu_attention_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Attention execution produces correct shape."""
        adapter = IPEXAttentionAdapter()
        query, key, value = sample_cpu_attention_inputs

        result = adapter(query, key, value)

        # Result shape should match query shape
        assert result.shape == query.shape

    @pytest.mark.skipif(
        not is_ipex_available(),
        reason="IPEX not available"
    )
    @pytest.mark.correctness
    def test_attention_correctness(
        self,
        sample_cpu_attention_inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """IPEX attention matches PyTorch SDPA."""
        adapter = IPEXAttentionAdapter()
        query, key, value = sample_cpu_attention_inputs

        result = adapter(query, key, value)
        expected = torch.nn.functional.scaled_dot_product_attention(
            query, key, value
        )

        torch.testing.assert_close(result, expected, rtol=1e-3, atol=1e-3)


class TestIPEXOptimizations:
    """Test IPEX-specific optimizations."""

    def test_adapter_has_bf16_support(self) -> None:
        """Adapter reports bf16 support."""
        adapter = IPEXMatmulAdapter()
        assert hasattr(adapter, "supports_bf16")

    def test_adapter_has_compile_flag(self) -> None:
        """Adapter has torch.compile integration flag."""
        adapter = IPEXMatmulAdapter()
        assert hasattr(adapter, "use_torch_compile")


class TestIPEXWhenUnavailable:
    """Test behavior when IPEX is not available."""

    def test_adapters_still_instantiable(self) -> None:
        """Adapters can still be instantiated when IPEX unavailable."""
        adapter = IPEXMatmulAdapter()
        assert adapter is not None

    def test_is_available_method(self) -> None:
        """Adapter has is_available method."""
        adapter = IPEXMatmulAdapter()
        assert hasattr(adapter, "is_available")
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_fallback_to_pytorch(self) -> None:
        """Adapter falls back to PyTorch when IPEX unavailable."""
        adapter = IPEXMatmulAdapter()
        if not adapter.is_available():
            assert adapter._use_fallback is True
