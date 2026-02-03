"""Tests for Intel oneDNN adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.cpu.onednn import (
    OneDNNLayerNormAdapter,
    OneDNNMatmulAdapter,
    detect_onednn_version,
    is_onednn_available,
)
from layerzero.backends.base import BaseKernel
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec


class TestOneDNNAvailability:
    """Test oneDNN availability detection."""

    def test_is_onednn_available_returns_bool(self) -> None:
        """is_onednn_available returns boolean."""
        result = is_onednn_available()
        assert isinstance(result, bool)

    def test_detect_version_returns_tuple_or_none(self) -> None:
        """detect_onednn_version returns tuple or None."""
        result = detect_onednn_version()
        assert result is None or (
            isinstance(result, tuple) and
            len(result) == 3 and
            all(isinstance(x, int) for x in result)
        )

    def test_version_consistent_with_availability(self) -> None:
        """Version is only available when package is available."""
        available = is_onednn_available()
        version = detect_onednn_version()

        if not available:
            assert version is None
        # Note: version can be None even if available (detection might fail)


class TestOneDNNMatmulAdapter:
    """Test oneDNN matmul adapter."""

    def test_adapter_inherits_base_kernel(self) -> None:
        """OneDNNMatmulAdapter inherits from BaseKernel."""
        adapter = OneDNNMatmulAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = OneDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_spec_platform_is_cpu(self) -> None:
        """KernelSpec platform is CPU."""
        adapter = OneDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.platform == Platform.CPU

    def test_kernel_spec_source(self) -> None:
        """KernelSpec source is onednn."""
        adapter = OneDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "onednn"

    def test_kernel_spec_operation(self) -> None:
        """KernelSpec operation is matmul."""
        adapter = OneDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.operation == "matmul"

    def test_supported_dtypes(self) -> None:
        """Adapter supports expected dtypes."""
        adapter = OneDNNMatmulAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float32 in spec.supported_dtypes

    @pytest.mark.skipif(
        not is_onednn_available(),
        reason="oneDNN not available"
    )
    def test_matmul_execution(
        self,
        sample_cpu_matrices: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Matmul execution produces correct shape."""
        adapter = OneDNNMatmulAdapter()
        a, b = sample_cpu_matrices

        result = adapter(a, b)

        # Result shape should be (batch, M, N)
        expected_shape = (a.shape[0], a.shape[1], b.shape[2])
        assert result.shape == expected_shape

    @pytest.mark.skipif(
        not is_onednn_available(),
        reason="oneDNN not available"
    )
    @pytest.mark.correctness
    def test_matmul_correctness(
        self,
        sample_cpu_matrices: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """oneDNN matmul matches PyTorch."""
        adapter = OneDNNMatmulAdapter()
        a, b = sample_cpu_matrices

        result = adapter(a, b)
        expected = torch.bmm(a, b)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


class TestOneDNNLayerNormAdapter:
    """Test oneDNN LayerNorm adapter."""

    def test_adapter_inherits_base_kernel(self) -> None:
        """OneDNNLayerNormAdapter inherits from BaseKernel."""
        adapter = OneDNNLayerNormAdapter(normalized_shape=512)
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = OneDNNLayerNormAdapter(normalized_shape=512)
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_spec_operation(self) -> None:
        """KernelSpec operation is layer_norm."""
        adapter = OneDNNLayerNormAdapter(normalized_shape=512)
        spec = adapter.get_kernel_spec()
        assert spec.operation == "layer_norm"

    @pytest.mark.skipif(
        not is_onednn_available(),
        reason="oneDNN not available"
    )
    def test_layernorm_execution(
        self,
        sample_cpu_layernorm_input: torch.Tensor,
    ) -> None:
        """LayerNorm execution produces correct shape."""
        hidden_dim = sample_cpu_layernorm_input.shape[-1]
        adapter = OneDNNLayerNormAdapter(normalized_shape=hidden_dim)

        result = adapter(sample_cpu_layernorm_input)

        assert result.shape == sample_cpu_layernorm_input.shape

    @pytest.mark.skipif(
        not is_onednn_available(),
        reason="oneDNN not available"
    )
    @pytest.mark.correctness
    def test_layernorm_correctness(
        self,
        sample_cpu_layernorm_input: torch.Tensor,
    ) -> None:
        """oneDNN LayerNorm matches PyTorch."""
        hidden_dim = sample_cpu_layernorm_input.shape[-1]
        adapter = OneDNNLayerNormAdapter(normalized_shape=hidden_dim)

        result = adapter(sample_cpu_layernorm_input)

        # Use PyTorch LayerNorm as reference
        layer_norm = torch.nn.LayerNorm(hidden_dim)
        # Use same weights
        if hasattr(adapter, "_weight") and adapter._weight is not None:
            layer_norm.weight.data = adapter._weight
        if hasattr(adapter, "_bias") and adapter._bias is not None:
            layer_norm.bias.data = adapter._bias
        expected = layer_norm(sample_cpu_layernorm_input)

        torch.testing.assert_close(result, expected, rtol=1e-4, atol=1e-4)


class TestOneDNNWhenUnavailable:
    """Test behavior when oneDNN is not available."""

    def test_adapters_still_instantiable(self) -> None:
        """Adapters can still be instantiated when oneDNN unavailable."""
        # This tests that the adapter doesn't crash on import/instantiation
        adapter = OneDNNMatmulAdapter()
        assert adapter is not None

    def test_is_available_method(self) -> None:
        """Adapter has is_available method."""
        adapter = OneDNNMatmulAdapter()
        assert hasattr(adapter, "is_available")
        result = adapter.is_available()
        assert isinstance(result, bool)
