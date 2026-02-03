"""Tests for xFormers adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.xformers.adapter import XFormersAdapter
from layerzero.backends.xformers.version import is_xformers_available
from layerzero.backends.base import BaseKernel
from layerzero.enums import Layout, MaskType, Platform
from layerzero.models.kernel_spec import KernelSpec


class TestXFormersAdapter:
    """Test xFormers adapter."""

    def test_inherits_base_kernel(self) -> None:
        """Adapter inherits from BaseKernel."""
        adapter = XFormersAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_id_contains_xformers(self) -> None:
        """kernel_id contains xformers."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert "xformers" in spec.kernel_id.lower()

    def test_operation_is_attention(self) -> None:
        """Operation is attention-related."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert "attention" in spec.operation

    def test_source_is_xformers(self) -> None:
        """Source is xformers."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "xformers"

    def test_platform_is_cuda(self) -> None:
        """Platform is CUDA."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.platform == Platform.CUDA

    def test_min_sm_is_70(self) -> None:
        """Minimum SM is 7.0 (Volta)."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.min_sm == (7, 0)

    def test_supports_fp16_bf16(self) -> None:
        """Supports fp16 and bf16."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float16 in spec.supported_dtypes
        assert torch.bfloat16 in spec.supported_dtypes

    def test_no_fp32_support(self) -> None:
        """Does not support fp32."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float32 not in spec.supported_dtypes

    def test_requires_bshd_layout(self) -> None:
        """Requires BSHD layout."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert Layout.BSHD in spec.requires_layouts

    def test_produces_bshd_layout(self) -> None:
        """Produces BSHD layout."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.produces_layout == Layout.BSHD

    def test_supports_attn_mask(self) -> None:
        """Supports attention mask."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.supports_attn_mask is True
        assert MaskType.FLOAT in spec.supported_attn_mask_types

    def test_is_available_property(self) -> None:
        """is_available returns bool."""
        adapter = XFormersAdapter()
        assert isinstance(adapter.is_available, bool)

    def test_version_property(self) -> None:
        """version returns tuple or None."""
        adapter = XFormersAdapter()
        version = adapter.version
        assert version is None or isinstance(version, tuple)

    def test_priority_is_lower_than_flash(self) -> None:
        """Priority is lower than FlashAttention."""
        adapter = XFormersAdapter()
        spec = adapter.get_kernel_spec()
        # FlashAttn priority is 90, FlashInfer is 85
        assert spec.priority < 85  # Lower than FlashInfer

    @pytest.mark.skipif(
        not is_xformers_available(),
        reason="xFormers not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_call_with_valid_input(
        self,
        sample_qkv_bshd: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> None:
        """Adapter callable with valid input."""
        adapter = XFormersAdapter()
        q, k, v = sample_qkv_bshd

        output = adapter(query=q, key=k, value=v, is_causal=True)

        # Output same shape as query
        assert output.shape == q.shape

    def test_call_without_xformers_raises(self) -> None:
        """Calling without xformers raises RuntimeError."""
        if is_xformers_available():
            pytest.skip("xFormers is installed")

        adapter = XFormersAdapter()
        q = torch.randn(2, 16, 8, 64, dtype=torch.float16)
        k = torch.randn(2, 16, 8, 64, dtype=torch.float16)
        v = torch.randn(2, 16, 8, 64, dtype=torch.float16)

        with pytest.raises(RuntimeError, match="[xX]formers"):
            adapter(query=q, key=k, value=v, is_causal=True)


class TestXFormersWithAttnBias:
    """Test xFormers with attention bias."""

    @pytest.mark.skipif(
        not is_xformers_available(),
        reason="xFormers not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_with_expanded_bias(
        self,
        device: torch.device,
        sample_qkv_bshd: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        sample_attn_bias_full: torch.Tensor,
    ) -> None:
        """Adapter works with fully expanded attn_bias."""
        adapter = XFormersAdapter()
        q, k, v = sample_qkv_bshd

        output = adapter(
            query=q,
            key=k,
            value=v,
            attn_bias=sample_attn_bias_full,
            is_causal=False,
        )

        assert output.shape == q.shape

    @pytest.mark.skipif(
        not is_xformers_available(),
        reason="xFormers not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_bias_auto_expansion(
        self,
        device: torch.device,
    ) -> None:
        """Adapter auto-expands broadcast bias."""
        adapter = XFormersAdapter()

        batch, seq, heads, dim = 2, 64, 4, 64
        q = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)

        # Broadcast batch dim
        bias = torch.zeros(1, heads, seq, seq, device=device, dtype=torch.float16)

        # Adapter should auto-expand
        output = adapter(
            query=q,
            key=k,
            value=v,
            attn_bias=bias,
            is_causal=False,
            auto_expand_bias=True,
        )

        assert output.shape == q.shape


class TestXFormersGQA:
    """Test GQA support in xFormers."""

    @pytest.mark.skipif(
        not is_xformers_available(),
        reason="xFormers not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gqa_5d_inputs(self, device: torch.device) -> None:
        """GQA with 5D K/V inputs works."""
        adapter = XFormersAdapter()

        batch, seq, heads_q, dim = 2, 64, 8, 64
        heads_kv = 2
        groups = heads_q // heads_kv

        q = torch.randn(batch, seq, heads_q, dim, device=device, dtype=torch.float16)
        # 5D format: (B, S, G, Hkv, D)
        k = torch.randn(batch, seq, groups, heads_kv, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, seq, groups, heads_kv, dim, device=device, dtype=torch.float16)

        output = adapter(
            query=q,
            key=k,
            value=v,
            is_causal=True,
            use_5d_gqa=True,
        )

        assert output.shape == q.shape

    @pytest.mark.skipif(
        not is_xformers_available(),
        reason="xFormers not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_mqa_single_kv_head(self, device: torch.device) -> None:
        """MQA with single KV head works."""
        adapter = XFormersAdapter()

        batch, seq, heads_q, dim = 2, 64, 8, 64
        heads_kv = 1
        groups = heads_q // heads_kv

        q = torch.randn(batch, seq, heads_q, dim, device=device, dtype=torch.float16)
        # 5D format for MQA: (B, S, 8, 1, D)
        k = torch.randn(batch, seq, groups, heads_kv, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, seq, groups, heads_kv, dim, device=device, dtype=torch.float16)

        output = adapter(
            query=q,
            key=k,
            value=v,
            is_causal=True,
            use_5d_gqa=True,
        )

        assert output.shape == q.shape


class TestXFormersCorrectness:
    """Correctness tests comparing xFormers to reference."""

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_xformers_available(),
        reason="xFormers not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_xformers_vs_sdpa_fp16(self, device: torch.device) -> None:
        """xFormers matches SDPA within fp16 tolerance."""
        adapter = XFormersAdapter()

        batch, seq, heads, dim = 1, 64, 4, 64
        torch.manual_seed(42)

        q = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)

        # xFormers output (BSHD layout)
        xf_output = adapter(query=q, key=k, value=v, is_causal=True)

        # SDPA reference (BSHD -> BHSD for SDPA)
        q_bhsd = q.transpose(1, 2)
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            sdpa_output = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=True
            )
        sdpa_output = sdpa_output.transpose(1, 2)  # Back to BSHD

        # Compare within tolerance
        max_diff = (xf_output - sdpa_output).abs().max().item()
        assert max_diff < 0.01, f"xFormers vs SDPA max diff: {max_diff}"

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_xformers_available(),
        reason="xFormers not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_xformers_vs_sdpa_bf16(self, device: torch.device) -> None:
        """xFormers matches SDPA within bf16 tolerance."""
        adapter = XFormersAdapter()

        batch, seq, heads, dim = 1, 64, 4, 64
        torch.manual_seed(42)

        q = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.bfloat16)

        # xFormers output
        xf_output = adapter(query=q, key=k, value=v, is_causal=True)

        # SDPA reference
        q_bhsd = q.transpose(1, 2)
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            sdpa_output = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=True
            )
        sdpa_output = sdpa_output.transpose(1, 2)

        # BF16 has lower precision
        max_diff = (xf_output - sdpa_output).abs().max().item()
        assert max_diff < 0.02, f"xFormers vs SDPA max diff: {max_diff}"
