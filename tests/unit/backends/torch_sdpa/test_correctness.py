"""Correctness tests for Torch SDPA adapter.

Compare SDPA output against naive reference implementation.
"""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.torch_sdpa.adapter import TorchSDPAAdapter

from .conftest import reference_attention


class TestSDPACorrectnessFP16:
    """Test SDPA correctness with fp16."""

    def test_basic_attention_matches_reference(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test basic attention matches reference."""
        adapter = TorchSDPAAdapter()

        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
        )

        reference = reference_attention(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
        )

        # Relative tolerance for fp16
        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

    def test_causal_attention_matches_reference(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test causal attention matches reference."""
        adapter = TorchSDPAAdapter()

        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            is_causal=True,
        )

        reference = reference_attention(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            is_causal=True,
        )

        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

    def test_scaled_attention_matches_reference(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test scaled attention matches reference."""
        adapter = TorchSDPAAdapter()
        scale = 0.125

        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            scale=scale,
        )

        reference = reference_attention(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            scale=scale,
        )

        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


class TestSDPACorrectnessFP32:
    """Test SDPA correctness with fp32."""

    def test_fp32_attention_matches_reference(
        self,
        qkv_fp32: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Test fp32 attention matches reference."""
        q, k, v = qkv_fp32
        adapter = TorchSDPAAdapter()

        output = adapter(query=q, key=k, value=v)
        reference = reference_attention(query=q, key=k, value=v)

        # Tighter tolerance for fp32
        torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)

    def test_fp32_causal_matches_reference(
        self,
        qkv_fp32: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Test fp32 causal attention matches reference."""
        q, k, v = qkv_fp32
        adapter = TorchSDPAAdapter()

        output = adapter(query=q, key=k, value=v, is_causal=True)
        reference = reference_attention(query=q, key=k, value=v, is_causal=True)

        torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)


class TestSDPACorrectnessBF16:
    """Test SDPA correctness with bf16."""

    def test_bf16_attention_matches_reference(
        self,
        qkv_bf16: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Test bf16 attention matches reference."""
        q, k, v = qkv_bf16
        adapter = TorchSDPAAdapter()

        output = adapter(query=q, key=k, value=v)
        reference = reference_attention(query=q, key=k, value=v)

        # bf16 has lower precision
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


class TestSDPACorrectnessWithMask:
    """Test SDPA correctness with attention masks."""

    def test_bool_mask_matches_reference(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        bool_mask: torch.Tensor,
    ) -> None:
        """Test boolean mask matches reference."""
        adapter = TorchSDPAAdapter()

        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            attn_mask=bool_mask,
        )

        reference = reference_attention(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            attn_mask=bool_mask,
        )

        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

    def test_float_mask_matches_reference(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        float_mask: torch.Tensor,
    ) -> None:
        """Test float mask matches reference."""
        adapter = TorchSDPAAdapter()

        output = adapter(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            attn_mask=float_mask,
        )

        reference = reference_attention(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            attn_mask=float_mask,
        )

        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)


class TestSDPACorrectnessGQA:
    """Test SDPA correctness with GQA."""

    def test_gqa_output_shape(
        self,
        query_tensor: torch.Tensor,
        gqa_key_tensor: torch.Tensor,
        gqa_value_tensor: torch.Tensor,
    ) -> None:
        """Test GQA produces correct output shape."""
        adapter = TorchSDPAAdapter()

        output = adapter(
            query=query_tensor,  # 8 heads
            key=gqa_key_tensor,  # 2 heads
            value=gqa_value_tensor,  # 2 heads
            enable_gqa=True,
        )

        # Output should match query shape
        assert output.shape == query_tensor.shape


class TestSDPACorrectnessEdgeCases:
    """Test SDPA correctness edge cases."""

    def test_head_dim_84(
        self,
        head_dim_84_qkv: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Test non-power-of-2 head_dim=84."""
        q, k, v = head_dim_84_qkv
        adapter = TorchSDPAAdapter()

        output = adapter(query=q, key=k, value=v)
        reference = reference_attention(query=q, key=k, value=v)

        torch.testing.assert_close(output, reference, rtol=1e-2, atol=1e-2)

    def test_noncontiguous_input(
        self,
        noncontiguous_query: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test with non-contiguous query tensor."""
        adapter = TorchSDPAAdapter()

        # Verify input is actually non-contiguous
        assert not noncontiguous_query.is_contiguous()

        output = adapter(
            query=noncontiguous_query,
            key=key_tensor,
            value=value_tensor,
        )

        # Should produce valid output
        assert output.shape == noncontiguous_query.shape
        assert not torch.isnan(output).any()
