"""Tests for FlashAttention adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.flash_attn.adapter import FlashAttnAdapter
from layerzero.backends.flash_attn.version import is_flash_attn_available
from layerzero.models.kernel_spec import KernelSpec
from layerzero.enums import Platform, MaskType


class TestFlashAttnAdapterInit:
    """Test FlashAttnAdapter initialization."""

    def test_adapter_creation(self) -> None:
        """Test adapter can be created."""
        adapter = FlashAttnAdapter()
        assert adapter is not None

    def test_adapter_is_available_property(self) -> None:
        """Test is_available property matches detection."""
        adapter = FlashAttnAdapter()
        expected = is_flash_attn_available()
        assert adapter.is_available == expected

    def test_adapter_version_property(self) -> None:
        """Test version property."""
        adapter = FlashAttnAdapter()
        if adapter.is_available:
            assert adapter.version is not None
            assert len(adapter.version) == 3
        else:
            assert adapter.version is None


class TestFlashAttnAdapterKernelSpec:
    """Test KernelSpec from FlashAttnAdapter."""

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """Test get_kernel_spec returns KernelSpec."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_spec_kernel_id(self) -> None:
        """Test kernel_id starts with flash_attn."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.kernel_id.startswith("flash_attn")

    def test_kernel_spec_operation(self) -> None:
        """Test operation is attention."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.operation == "attention"

    def test_kernel_spec_source(self) -> None:
        """Test source is flash_attn."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "flash_attn"

    def test_kernel_spec_platform(self) -> None:
        """Test platform is CUDA."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.platform == Platform.CUDA

    def test_kernel_spec_min_sm(self) -> None:
        """Test min SM is 8.0."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.min_sm == (8, 0)

    def test_kernel_spec_supported_dtypes(self) -> None:
        """Test supported dtypes are fp16 and bf16."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float16 in spec.supported_dtypes
        assert torch.bfloat16 in spec.supported_dtypes
        # fp32 not supported
        assert torch.float32 not in spec.supported_dtypes

    def test_kernel_spec_head_dim_constraints(self) -> None:
        """Test head_dim constraints."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.max_head_dim == 256
        assert spec.head_dim_multiple == 8

    def test_kernel_spec_supports_gqa(self) -> None:
        """Test GQA support."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.supports_gqa is True

    def test_kernel_spec_cuda_graph_safe(self) -> None:
        """Test CUDA graph safety."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.is_cuda_graph_safe is True

    def test_kernel_spec_high_priority(self) -> None:
        """Test high priority (Flash is preferred)."""
        adapter = FlashAttnAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.priority >= 80


class TestFlashAttnAdapterExecution:
    """Test FlashAttnAdapter execution (requires flash_attn)."""

    @pytest.mark.skipif(
        not is_flash_attn_available(),
        reason="flash_attn not installed"
    )
    def test_call_basic_bshd(
        self,
        query_bshd: torch.Tensor,
        key_bshd: torch.Tensor,
        value_bshd: torch.Tensor,
    ) -> None:
        """Test basic execution with BSHD layout."""
        adapter = FlashAttnAdapter()
        output = adapter(
            query=query_bshd,
            key=key_bshd,
            value=value_bshd,
        )

        assert output.shape == query_bshd.shape
        assert output.dtype == query_bshd.dtype

    @pytest.mark.skipif(
        not is_flash_attn_available(),
        reason="flash_attn not installed"
    )
    def test_call_causal(
        self,
        query_bshd: torch.Tensor,
        key_bshd: torch.Tensor,
        value_bshd: torch.Tensor,
    ) -> None:
        """Test execution with causal masking."""
        adapter = FlashAttnAdapter()
        output = adapter(
            query=query_bshd,
            key=key_bshd,
            value=value_bshd,
            is_causal=True,
        )

        assert output.shape == query_bshd.shape

    @pytest.mark.skipif(
        not is_flash_attn_available(),
        reason="flash_attn not installed"
    )
    def test_call_bf16(
        self,
        qkv_bf16_bshd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Test execution with bf16."""
        q, k, v = qkv_bf16_bshd
        adapter = FlashAttnAdapter()
        output = adapter(query=q, key=k, value=v)

        assert output.shape == q.shape
        assert output.dtype == torch.bfloat16

    @pytest.mark.skipif(
        not is_flash_attn_available(),
        reason="flash_attn not installed"
    )
    def test_call_gqa(
        self,
        query_bshd: torch.Tensor,
        gqa_kv_bshd: tuple[torch.Tensor, torch.Tensor],
    ) -> None:
        """Test execution with GQA (different KV heads)."""
        k, v = gqa_kv_bshd
        adapter = FlashAttnAdapter()
        output = adapter(query=query_bshd, key=k, value=v)

        # Output should match query shape
        assert output.shape == query_bshd.shape


class TestFlashAttnAdapterFallback:
    """Test FlashAttnAdapter fallback when not available."""

    def test_graceful_when_not_installed(self) -> None:
        """Test adapter can be created even if flash_attn not installed."""
        # This should not raise
        adapter = FlashAttnAdapter()
        assert adapter is not None

    def test_call_raises_when_not_available(
        self,
        query_bshd: torch.Tensor,
        key_bshd: torch.Tensor,
        value_bshd: torch.Tensor,
    ) -> None:
        """Test __call__ raises appropriate error when not available."""
        adapter = FlashAttnAdapter()

        if not adapter.is_available:
            with pytest.raises(RuntimeError, match="[Ff]lash"):
                adapter(query=query_bshd, key=key_bshd, value=value_bshd)
