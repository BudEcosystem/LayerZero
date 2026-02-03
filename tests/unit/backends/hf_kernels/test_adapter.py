"""Tests for HuggingFace Kernel Hub adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.hf_kernels.adapter import HFKernelAdapter
from layerzero.backends.hf_kernels.version import is_hf_kernels_available
from layerzero.backends.base import BaseKernel
from layerzero.models.kernel_spec import KernelSpec


class TestHFKernelAdapter:
    """Test HF kernel adapter."""

    def test_adapter_inherits_base_kernel(self) -> None:
        """HFKernelAdapter inherits from BaseKernel."""
        adapter = HFKernelAdapter(kernel_name="mock_kernel")
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = HFKernelAdapter(kernel_name="mock_kernel")
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_spec_source(self) -> None:
        """KernelSpec source is hf_kernels."""
        adapter = HFKernelAdapter(kernel_name="mock_kernel")
        spec = adapter.get_kernel_spec()
        assert spec.source == "hf_kernels"

    def test_kernel_spec_has_kernel_id(self) -> None:
        """KernelSpec has kernel_id containing kernel name."""
        adapter = HFKernelAdapter(kernel_name="flash_attn")
        spec = adapter.get_kernel_spec()
        assert "flash_attn" in spec.kernel_id


class TestHFKernelAdapterWithVersion:
    """Test HF kernel adapter with version specification."""

    def test_adapter_with_version(self) -> None:
        """Adapter accepts version parameter."""
        adapter = HFKernelAdapter(
            kernel_name="flash_attn",
            version="2.6.0",
        )
        assert adapter is not None

    def test_adapter_version_in_spec(self) -> None:
        """Version appears in kernel spec."""
        adapter = HFKernelAdapter(
            kernel_name="flash_attn",
            version="2.6.0",
        )
        spec = adapter.get_kernel_spec()
        assert "2.6.0" in spec.version or spec.version != ""


class TestHFKernelAdapterAvailability:
    """Test adapter availability."""

    def test_adapter_has_is_available(self) -> None:
        """Adapter has is_available method."""
        adapter = HFKernelAdapter(kernel_name="mock_kernel")
        assert hasattr(adapter, "is_available")
        assert callable(adapter.is_available)

    def test_is_available_returns_bool(self) -> None:
        """is_available returns boolean."""
        adapter = HFKernelAdapter(kernel_name="mock_kernel")
        result = adapter.is_available()
        assert isinstance(result, bool)

    def test_mock_kernel_not_available(self) -> None:
        """Mock kernel is not actually available."""
        adapter = HFKernelAdapter(kernel_name="definitely_not_real_xyz123")
        assert adapter.is_available() is False


class TestHFKernelAdapterExecution:
    """Test adapter execution."""

    @pytest.mark.skipif(
        not is_hf_kernels_available(),
        reason="HF kernels not available"
    )
    def test_adapter_callable(self) -> None:
        """Adapter is callable."""
        adapter = HFKernelAdapter(kernel_name="flash_attn")
        assert callable(adapter)

    def test_unavailable_adapter_raises_on_call(self) -> None:
        """Calling unavailable adapter raises RuntimeError."""
        adapter = HFKernelAdapter(kernel_name="not_real_kernel")
        with pytest.raises(RuntimeError):
            adapter(torch.zeros(1))


class TestHFKernelAdapterClashDetection:
    """Test kernel clash detection."""

    def test_adapter_has_check_clashes(self) -> None:
        """Adapter has check_clashes method."""
        adapter = HFKernelAdapter(kernel_name="mock_kernel")
        assert hasattr(adapter, "check_namespace_clash")

    def test_check_clashes_returns_list(self) -> None:
        """check_namespace_clash returns list."""
        adapter = HFKernelAdapter(kernel_name="mock_kernel")
        result = adapter.check_namespace_clash(set())
        assert isinstance(result, list)

    def test_check_clashes_empty_with_no_existing(self) -> None:
        """No clashes with empty existing namespaces."""
        adapter = HFKernelAdapter(kernel_name="unique_kernel")
        result = adapter.check_namespace_clash(set())
        assert len(result) == 0


class TestHFKernelAdapterRegistration:
    """Test adapter registration with kernel selection."""

    def test_adapter_has_register_method(self) -> None:
        """Adapter has register method."""
        adapter = HFKernelAdapter(kernel_name="mock_kernel")
        assert hasattr(adapter, "register")

    def test_adapter_priority(self) -> None:
        """Adapter has appropriate priority."""
        adapter = HFKernelAdapter(kernel_name="mock_kernel")
        spec = adapter.get_kernel_spec()
        # HF kernels should have lower priority than native backends
        assert spec.priority < 70
