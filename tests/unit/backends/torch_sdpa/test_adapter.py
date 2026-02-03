"""Tests for TorchSDPAAdapter class."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.torch_sdpa.adapter import TorchSDPAAdapter
from layerzero.models.kernel_spec import KernelSpec
from layerzero.enums import Platform, MaskType


class TestTorchSDPAAdapterInit:
    """Test TorchSDPAAdapter initialization."""

    def test_adapter_creation(self) -> None:
        """Test adapter can be created."""
        adapter = TorchSDPAAdapter()
        assert adapter is not None

    def test_adapter_with_backend_hint(self) -> None:
        """Test adapter can be created with backend hint."""
        adapter = TorchSDPAAdapter(backend_hint="flash")
        assert adapter.backend_hint == "flash"

    def test_adapter_default_backend_hint_none(self) -> None:
        """Test default backend hint is None (auto)."""
        adapter = TorchSDPAAdapter()
        assert adapter.backend_hint is None


class TestTorchSDPAAdapterKernelSpec:
    """Test KernelSpec from TorchSDPAAdapter."""

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """Test get_kernel_spec returns KernelSpec."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_spec_kernel_id(self) -> None:
        """Test kernel_id is correct."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.kernel_id.startswith("torch.sdpa")

    def test_kernel_spec_operation(self) -> None:
        """Test operation is attention."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.operation == "attention"

    def test_kernel_spec_source(self) -> None:
        """Test source is torch."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "torch"

    def test_kernel_spec_platform(self) -> None:
        """Test platform is CUDA."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.platform == Platform.CUDA

    def test_kernel_spec_supported_dtypes(self) -> None:
        """Test supported dtypes include fp16, bf16, fp32."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float16 in spec.supported_dtypes
        assert torch.bfloat16 in spec.supported_dtypes
        assert torch.float32 in spec.supported_dtypes

    def test_kernel_spec_head_dim_constraints(self) -> None:
        """Test head_dim constraints."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.min_head_dim >= 1
        assert spec.max_head_dim >= 128

    def test_kernel_spec_supports_gqa(self) -> None:
        """Test GQA support."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.supports_gqa is True

    def test_kernel_spec_supports_attn_mask(self) -> None:
        """Test attention mask support."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.supports_attn_mask is True
        assert MaskType.BOOL in spec.supported_attn_mask_types
        assert MaskType.FLOAT in spec.supported_attn_mask_types

    def test_kernel_spec_cuda_graph_safe(self) -> None:
        """Test CUDA graph safety."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.is_cuda_graph_safe is True

    def test_kernel_spec_supports_dropout(self) -> None:
        """Test dropout support."""
        adapter = TorchSDPAAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.supports_dropout is True


class TestTorchSDPAAdapterExecution:
    """Test TorchSDPAAdapter execution."""

    def test_call_basic(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test basic execution."""
        adapter = TorchSDPAAdapter()
        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
        )

        assert output.shape == query_tensor.shape
        assert output.dtype == query_tensor.dtype

    def test_call_with_causal(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test execution with causal masking."""
        adapter = TorchSDPAAdapter()
        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            is_causal=True,
        )

        assert output.shape == query_tensor.shape

    def test_call_with_dropout(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test execution with dropout (training mode)."""
        adapter = TorchSDPAAdapter()
        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            dropout_p=0.1,
        )

        assert output.shape == query_tensor.shape

    def test_call_with_scale(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test execution with custom scale."""
        adapter = TorchSDPAAdapter()
        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            scale=0.125,
        )

        assert output.shape == query_tensor.shape

    def test_call_output_device_matches_input(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test output device matches input."""
        adapter = TorchSDPAAdapter()
        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
        )

        assert output.device == query_tensor.device
