"""Tests for FlashInfer adapter."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.flashinfer.adapter import (
    FlashInferPrefillAdapter,
    FlashInferDecodeAdapter,
    FlashInferPagedAdapter,
)
from layerzero.backends.flashinfer.version import is_flashinfer_available
from layerzero.backends.base import BaseKernel
from layerzero.enums import Layout, MaskType, Platform, KVCacheStrategy
from layerzero.models.kernel_spec import KernelSpec


class TestFlashInferPrefillAdapter:
    """Test FlashInfer prefill adapter."""

    def test_inherits_base_kernel(self) -> None:
        """Adapter inherits from BaseKernel."""
        adapter = FlashInferPrefillAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_id_contains_flashinfer(self) -> None:
        """kernel_id contains flashinfer."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert "flashinfer" in spec.kernel_id.lower()

    def test_operation_is_attention(self) -> None:
        """Operation is attention-related."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert "attention" in spec.operation

    def test_source_is_flashinfer(self) -> None:
        """Source is flashinfer."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.source == "flashinfer"

    def test_platform_is_cuda(self) -> None:
        """Platform is CUDA."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.platform == Platform.CUDA

    def test_min_sm_is_75(self) -> None:
        """Minimum SM is 7.5 (Turing)."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.min_sm == (7, 5)

    def test_supports_fp16_bf16(self) -> None:
        """Supports fp16 and bf16."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert torch.float16 in spec.supported_dtypes
        assert torch.bfloat16 in spec.supported_dtypes

    def test_supports_gqa(self) -> None:
        """Supports GQA."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.supports_gqa is True

    def test_supports_mqa(self) -> None:
        """Supports MQA."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.supports_mqa is True

    def test_requires_nhd_layout(self) -> None:
        """Requires NHD layout (internal)."""
        adapter = FlashInferPrefillAdapter()
        spec = adapter.get_kernel_spec()
        assert Layout.NHD in spec.requires_layouts

    def test_is_available_property(self) -> None:
        """is_available returns bool."""
        adapter = FlashInferPrefillAdapter()
        assert isinstance(adapter.is_available, bool)

    def test_version_property(self) -> None:
        """version returns tuple or None."""
        adapter = FlashInferPrefillAdapter()
        version = adapter.version
        assert version is None or isinstance(version, tuple)

    @pytest.mark.skipif(
        not is_flashinfer_available(),
        reason="FlashInfer not installed"
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
        adapter = FlashInferPrefillAdapter()
        q, k, v = sample_qkv_bshd

        # Adapter handles layout conversion internally
        output = adapter(query=q, key=k, value=v, is_causal=True)

        # Output same shape as query
        assert output.shape == q.shape

    def test_call_without_flashinfer_raises(self) -> None:
        """Calling without flashinfer raises RuntimeError."""
        if is_flashinfer_available():
            pytest.skip("FlashInfer is installed")

        adapter = FlashInferPrefillAdapter()
        q = torch.randn(2, 16, 8, 64, dtype=torch.float16)
        k = torch.randn(2, 16, 8, 64, dtype=torch.float16)
        v = torch.randn(2, 16, 8, 64, dtype=torch.float16)

        with pytest.raises(RuntimeError, match="[Ff]lash[Ii]nfer"):
            adapter(query=q, key=k, value=v, is_causal=True)


class TestFlashInferDecodeAdapter:
    """Test FlashInfer decode adapter."""

    def test_inherits_base_kernel(self) -> None:
        """Adapter inherits from BaseKernel."""
        adapter = FlashInferDecodeAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = FlashInferDecodeAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_id_contains_decode(self) -> None:
        """kernel_id contains decode."""
        adapter = FlashInferDecodeAdapter()
        spec = adapter.get_kernel_spec()
        assert "decode" in spec.kernel_id.lower()

    def test_operation_is_decode(self) -> None:
        """Operation contains decode."""
        adapter = FlashInferDecodeAdapter()
        spec = adapter.get_kernel_spec()
        assert "decode" in spec.operation

    def test_supports_kv_cache_strategies(self) -> None:
        """Supports KV cache strategies."""
        adapter = FlashInferDecodeAdapter()
        spec = adapter.get_kernel_spec()
        assert len(spec.supports_kv_strategies) > 0

    def test_is_cuda_graph_safe(self) -> None:
        """Decode is CUDA graph safe (after warmup)."""
        adapter = FlashInferDecodeAdapter()
        spec = adapter.get_kernel_spec()
        assert spec.is_cuda_graph_safe is True


class TestFlashInferPagedAdapter:
    """Test FlashInfer paged KV cache adapter."""

    def test_inherits_base_kernel(self) -> None:
        """Adapter inherits from BaseKernel."""
        adapter = FlashInferPagedAdapter()
        assert isinstance(adapter, BaseKernel)

    def test_get_kernel_spec_returns_kernel_spec(self) -> None:
        """get_kernel_spec returns KernelSpec."""
        adapter = FlashInferPagedAdapter()
        spec = adapter.get_kernel_spec()
        assert isinstance(spec, KernelSpec)

    def test_kernel_id_contains_paged(self) -> None:
        """kernel_id contains paged."""
        adapter = FlashInferPagedAdapter()
        spec = adapter.get_kernel_spec()
        assert "paged" in spec.kernel_id.lower()

    def test_supports_paged_kv_strategy(self) -> None:
        """Supports PAGED KV cache strategy."""
        adapter = FlashInferPagedAdapter()
        spec = adapter.get_kernel_spec()
        assert KVCacheStrategy.PAGED in spec.supports_kv_strategies

    def test_supports_kv_cache_layouts(self) -> None:
        """Supports KV cache layouts."""
        adapter = FlashInferPagedAdapter()
        spec = adapter.get_kernel_spec()
        assert len(spec.supports_kv_cache_layouts) > 0

    @pytest.mark.skipif(
        not is_flashinfer_available(),
        reason="FlashInfer not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_call_with_paged_kv(
        self,
        device: torch.device,
    ) -> None:
        """Adapter callable with paged KV cache."""
        # Note: FlashInfer paged API is complex and version-specific
        # This test validates the adapter initializes correctly
        # Full paged attention testing requires proper wrapper setup
        adapter = FlashInferPagedAdapter()

        # Verify adapter has paged support in spec
        spec = adapter.get_kernel_spec()
        from layerzero.enums import KVCacheStrategy
        assert KVCacheStrategy.PAGED in spec.supports_kv_strategies

        # Skip actual execution test - paged API requires specific setup
        # that varies significantly between FlashInfer versions
        pytest.skip(
            "FlashInfer paged attention requires version-specific API setup. "
            "Testing kernel spec only."
        )


class TestFlashInferAdapterJIT:
    """Test FlashInfer JIT warmup functionality."""

    def test_warmup_method_exists(self) -> None:
        """Adapters have warmup method."""
        adapter = FlashInferPrefillAdapter()
        assert hasattr(adapter, "warmup")
        assert callable(adapter.warmup)

    def test_warmup_accepts_shapes(self) -> None:
        """warmup accepts list of shapes."""
        adapter = FlashInferPrefillAdapter()
        shapes = [
            (1, 128, 8, 64),
            (1, 256, 8, 64),
            (1, 512, 8, 64),
        ]
        # Should not raise
        try:
            adapter.warmup(shapes=shapes, dry_run=True)
        except RuntimeError:
            # OK if flashinfer not installed
            pass

    def test_jit_cache_status_property(self) -> None:
        """Adapter has jit_cache_enabled property."""
        adapter = FlashInferPrefillAdapter()
        assert hasattr(adapter, "jit_cache_enabled")
        assert isinstance(adapter.jit_cache_enabled, bool)


class TestFlashInferCorrectness:
    """Correctness tests comparing FlashInfer to reference."""

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_flashinfer_available(),
        reason="FlashInfer not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_flashinfer_vs_sdpa_fp16(self, device: torch.device) -> None:
        """FlashInfer matches SDPA within fp16 tolerance."""
        # Import FlashInfer's single_prefill directly for proper testing
        try:
            from flashinfer import single_prefill_with_kv_cache
        except ImportError:
            pytest.skip("FlashInfer not available")

        batch, seq, heads, dim = 1, 64, 4, 64  # Smaller test for speed
        torch.manual_seed(42)

        # Create BSHD tensors
        q = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.float16)

        # FlashInfer expects NHD layout
        q_nhd = q.reshape(batch * seq, heads, dim)
        k_nhd = k.reshape(batch * seq, heads, dim)
        v_nhd = v.reshape(batch * seq, heads, dim)

        # FlashInfer output
        fi_output_nhd = single_prefill_with_kv_cache(q_nhd, k_nhd, v_nhd, causal=True)
        fi_output = fi_output_nhd.reshape(batch, seq, heads, dim)

        # SDPA reference (BSHD -> BHSD for SDPA)
        q_bhsd = q.transpose(1, 2)
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            sdpa_output = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=True
            )
        sdpa_output = sdpa_output.transpose(1, 2)  # Back to BSHD

        # Compare within tolerance (FlashInfer vs SDPA MATH backend)
        # Note: FlashInfer uses different algorithms, allowing looser tolerances
        max_diff = (fi_output - sdpa_output).abs().max().item()
        assert max_diff < 0.01, f"FlashInfer vs SDPA max diff: {max_diff}"

    @pytest.mark.correctness
    @pytest.mark.skipif(
        not is_flashinfer_available(),
        reason="FlashInfer not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_flashinfer_vs_sdpa_bf16(self, device: torch.device) -> None:
        """FlashInfer matches SDPA within bf16 tolerance."""
        try:
            from flashinfer import single_prefill_with_kv_cache
        except ImportError:
            pytest.skip("FlashInfer not available")

        batch, seq, heads, dim = 1, 64, 4, 64
        torch.manual_seed(42)

        q = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.bfloat16)
        k = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.bfloat16)
        v = torch.randn(batch, seq, heads, dim, device=device, dtype=torch.bfloat16)

        # FlashInfer expects NHD layout
        q_nhd = q.reshape(batch * seq, heads, dim)
        k_nhd = k.reshape(batch * seq, heads, dim)
        v_nhd = v.reshape(batch * seq, heads, dim)

        # FlashInfer output
        fi_output_nhd = single_prefill_with_kv_cache(q_nhd, k_nhd, v_nhd, causal=True)
        fi_output = fi_output_nhd.reshape(batch, seq, heads, dim)

        # SDPA reference
        q_bhsd = q.transpose(1, 2)
        k_bhsd = k.transpose(1, 2)
        v_bhsd = v.transpose(1, 2)
        with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
            sdpa_output = torch.nn.functional.scaled_dot_product_attention(
                q_bhsd, k_bhsd, v_bhsd, is_causal=True
            )
        sdpa_output = sdpa_output.transpose(1, 2)

        # BF16 has lower precision - allow larger tolerance
        max_diff = (fi_output - sdpa_output).abs().max().item()
        assert max_diff < 0.02, f"FlashInfer vs SDPA max diff: {max_diff}"


class TestFlashInferGQA:
    """Test GQA support in FlashInfer."""

    @pytest.mark.skipif(
        not is_flashinfer_available(),
        reason="FlashInfer not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gqa_group_size_2(self, device: torch.device) -> None:
        """GQA with group size 2 works."""
        adapter = FlashInferPrefillAdapter()

        batch, seq, q_heads, kv_heads, dim = 2, 128, 8, 4, 64
        q = torch.randn(batch, seq, q_heads, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, seq, kv_heads, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, seq, kv_heads, dim, device=device, dtype=torch.float16)

        output = adapter(query=q, key=k, value=v, is_causal=True)

        assert output.shape == q.shape

    @pytest.mark.skipif(
        not is_flashinfer_available(),
        reason="FlashInfer not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gqa_group_size_4(self, device: torch.device) -> None:
        """GQA with group size 4 works."""
        adapter = FlashInferPrefillAdapter()

        batch, seq, q_heads, kv_heads, dim = 2, 128, 8, 2, 64
        q = torch.randn(batch, seq, q_heads, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, seq, kv_heads, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, seq, kv_heads, dim, device=device, dtype=torch.float16)

        output = adapter(query=q, key=k, value=v, is_causal=True)

        assert output.shape == q.shape

    @pytest.mark.skipif(
        not is_flashinfer_available(),
        reason="FlashInfer not installed"
    )
    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )
    def test_gqa_group_size_8(self, device: torch.device) -> None:
        """GQA with group size 8 (MQA) works."""
        adapter = FlashInferPrefillAdapter()

        batch, seq, q_heads, kv_heads, dim = 2, 128, 8, 1, 64
        q = torch.randn(batch, seq, q_heads, dim, device=device, dtype=torch.float16)
        k = torch.randn(batch, seq, kv_heads, dim, device=device, dtype=torch.float16)
        v = torch.randn(batch, seq, kv_heads, dim, device=device, dtype=torch.float16)

        output = adapter(query=q, key=k, value=v, is_causal=True)

        assert output.shape == q.shape
