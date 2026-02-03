"""Tests for AMD ZenDNN adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.cpu.zendnn import (
    ZenDNNMatmulAdapter,
    detect_zendnn_version,
    is_aocl_blas_available,
    is_zendnn_available,
)
from layerzero.backends.base import BaseKernel
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec


class TestZenDNNAvailability:
    """Test ZenDNN availability detection."""

    def test_is_zendnn_available_returns_bool(self) -> None:
        """is_zendnn_available returns boolean."""
        result = is_zendnn_available()
        assert isinstance(result, bool)

    def test_is_aocl_blas_available_returns_bool(self) -> None:
        """is_aocl_blas_available returns boolean."""
        result = is_aocl_blas_available()
        assert isinstance(result, bool)

    def test_detect_version_returns_tuple_or_none(self) -> None:
        """detect_zendnn_version returns tuple or None."""
        result = detect_zendnn_version()
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 3 and
            all(isinstance(x, int) for x in result)
        )

    def test_version_consistent_with_availability(self) -> None:
        """Version is only available when package is available."""
        available = is_zendnn_available()
        version = detect_zendnn_version()

        if not available:
            assert version is None


class TestZenDNNMatmulAdapter:
    """Test ZenDNN matmul adapter."""

    def test_adapter_inherits_base_kernel(self) -> None:
        """ZenDNNMatmulAdapter inherits from BaseKernel."""
        adapter = ZenDNNMatmulAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = ZenDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_spec_platform_is_cpu(self) -> None:
        """KernelSpec platform is CPU."""
        adapter = ZenDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.platform == Platform.CPU

    def test_kernel_spec_source(self) -> None:
        """KernelSpec source is zendnn."""
        adapter = ZenDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "zendnn"

    def test_kernel_spec_operation(self) -> None:
        """KernelSpec operation is matmul."""
        adapter = ZenDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.operation == "matmul"

    def test_supported_dtypes(self) -> None:
        """Adapter supports expected dtypes."""
        adapter = ZenDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float32 in spec.supported_dtypes

    @pytest.mark.skipif(
        not is_zendnn_available(),
        reason="ZenDNN not available"
    )
    def test_matmul_execution(
        self,
        sample_cpu_matrices: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Matmul execution produces correct shape."""
        adapter = ZenDNNMatmulAdapter()
        a, b = sample_cpu_matrices

        result = adapter(a, b)

        # Result shape should be (batch, M, N)
        expected_shape = (a.shape[0], a.shape[1], b.shape[2])
        assert result.shape == expected_shape

    @pytest.mark.skipif(
        not is_zendnn_available(),
        reason="ZenDNN not available"
    )
    @pytest.mark.correctness
    def test_matmul_correctness(
        self,
        sample_cpu_matrices: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """ZenDNN matmul matches PyTorch."""
        adapter = ZenDNNMatmulAdapter()
        a, b = sample_cpu_matrices

        result = adapter(a, b)
        expected = torch.bmm(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


class TestZenDNNEPYCOptimization:
    """Test ZenDNN EPYC-specific optimizations."""

    def test_adapter_has_epyc_flag(self) -> None:
        """Adapter has EPYC optimization flag."""
        adapter = ZenDNNMatmulAdapter()
        assert hasattr(adapter, "is_epyc_optimized")

    def test_epyc_flag_returns_bool(self) -> None:
        """EPYC optimization flag returns boolean."""
        adapter = ZenDNNMatmulAdapter()
        result = adapter.is_epyc_optimized()
        assert isinstance(result, bool)

    def test_thread_count_configurable(self) -> None:
        """Thread count is configurable."""
        adapter = ZenDNNMatmulAdapter(num_threads=4)
        assert hasattr(adapter, "_num_threads")
        assert adapter._num_threads == 4


class TestZenDNNWhenUnavailable:
    """Test behavior when ZenDNN is not available."""

    def test_adapters_still_instantiable(self) -> None:
        """Adapters can still be instantiated when ZenDNN unavailable."""
        adapter = ZenDNNMatmulAdapter()
        assert adapter is not None

    def test_is_available_method(self) -> None:
        """Adapter has is_available method."""
        adapter = ZenDNNMatmulAdapter()
        assert hasattr(adapter, "is_available")
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_fallback_to_pytorch(self) -> None:
        """Adapter falls back to PyTorch when ZenDNN unavailable."""
        adapter = ZenDNNMatmulAdapter()
        if not adapter.is_available():
            assert adapter._use_fallback is True
