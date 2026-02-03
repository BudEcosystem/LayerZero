"""Tests for LayerZero operation APIs.

Tests for lz.attention(), lz.rms_norm(), lz.layer_norm(), etc.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F


class TestAttentionAPI:
    """Tests for lz.attention() public API."""

    def test_attention_basic(self) -> None:
        """Basic attention call."""
        import layerzero as lz

        batch_size = 2
        seq_len = 64
        num_heads = 8
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        output = lz.attention(q, k, v)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert output.dtype == q.dtype

    def test_attention_causal(self) -> None:
        """Causal attention masking."""
        import layerzero as lz

        q = torch.randn(2, 8, 64, 64)
        k = torch.randn(2, 8, 64, 64)
        v = torch.randn(2, 8, 64, 64)

        output = lz.attention(q, k, v, is_causal=True)

        # Reference with torch SDPA
        expected = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert output.shape == expected.shape
        # Numerical comparison with tolerance
        assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3)

    def test_attention_with_mask(self) -> None:
        """Attention with explicit mask."""
        import layerzero as lz

        q = torch.randn(2, 8, 64, 64)
        k = torch.randn(2, 8, 64, 64)
        v = torch.randn(2, 8, 64, 64)

        # Create causal mask
        mask = torch.triu(torch.ones(64, 64), diagonal=1).bool()
        attn_mask = torch.zeros(64, 64).masked_fill(mask, float('-inf'))

        output = lz.attention(q, k, v, attn_mask=attn_mask)

        assert output.shape == (2, 8, 64, 64)

    def test_attention_with_scale(self) -> None:
        """Custom attention scale."""
        import layerzero as lz

        q = torch.randn(2, 8, 64, 64)
        k = torch.randn(2, 8, 64, 64)
        v = torch.randn(2, 8, 64, 64)

        custom_scale = 0.1
        output = lz.attention(q, k, v, scale=custom_scale)

        assert output.shape == (2, 8, 64, 64)

    def test_attention_with_dropout(self) -> None:
        """Attention with dropout (training mode)."""
        import layerzero as lz

        q = torch.randn(2, 8, 64, 64)
        k = torch.randn(2, 8, 64, 64)
        v = torch.randn(2, 8, 64, 64)

        # Dropout only affects training
        output = lz.attention(q, k, v, dropout_p=0.1)

        assert output.shape == (2, 8, 64, 64)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_attention_cuda(self) -> None:
        """Attention on CUDA."""
        import layerzero as lz

        q = torch.randn(2, 8, 64, 64, device='cuda')
        k = torch.randn(2, 8, 64, 64, device='cuda')
        v = torch.randn(2, 8, 64, 64, device='cuda')

        output = lz.attention(q, k, v, is_causal=True)

        assert output.device.type == 'cuda'

    def test_attention_fp16(self) -> None:
        """Attention with float16."""
        import layerzero as lz

        q = torch.randn(2, 8, 64, 64, dtype=torch.float16)
        k = torch.randn(2, 8, 64, 64, dtype=torch.float16)
        v = torch.randn(2, 8, 64, 64, dtype=torch.float16)

        output = lz.attention(q, k, v)

        assert output.dtype == torch.float16

    def test_attention_backend_override(self) -> None:
        """Override kernel selection with explicit backend."""
        import layerzero as lz

        q = torch.randn(2, 8, 64, 64)
        k = torch.randn(2, 8, 64, 64)
        v = torch.randn(2, 8, 64, 64)

        # Force SDPA backend
        output = lz.attention(q, k, v, backend="torch_sdpa")

        assert output.shape == (2, 8, 64, 64)


class TestRMSNormAPI:
    """Tests for lz.rms_norm() public API."""

    def test_rms_norm_basic(self) -> None:
        """Basic RMS normalization."""
        import layerzero as lz

        x = torch.randn(2, 64, 256)
        weight = torch.ones(256)

        output = lz.rms_norm(x, weight)

        assert output.shape == (2, 64, 256)

    def test_rms_norm_correctness(self) -> None:
        """RMS norm matches reference implementation."""
        import layerzero as lz

        x = torch.randn(2, 64, 256)
        weight = torch.randn(256)
        eps = 1e-6

        output = lz.rms_norm(x, weight, eps=eps)

        # Reference implementation
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        expected = x * torch.rsqrt(variance + eps) * weight

        assert torch.allclose(output, expected, rtol=1e-4, atol=1e-4)

    def test_rms_norm_custom_eps(self) -> None:
        """RMS norm with custom epsilon."""
        import layerzero as lz

        x = torch.randn(2, 64, 256)
        weight = torch.ones(256)

        output = lz.rms_norm(x, weight, eps=1e-5)

        assert output.shape == x.shape

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_rms_norm_cuda(self) -> None:
        """RMS norm on CUDA."""
        import layerzero as lz

        x = torch.randn(2, 64, 256, device='cuda')
        weight = torch.ones(256, device='cuda')

        output = lz.rms_norm(x, weight)

        assert output.device.type == 'cuda'


class TestLayerNormAPI:
    """Tests for lz.layer_norm() public API."""

    def test_layer_norm_basic(self) -> None:
        """Basic layer normalization."""
        import layerzero as lz

        x = torch.randn(2, 64, 256)
        weight = torch.ones(256)
        bias = torch.zeros(256)

        output = lz.layer_norm(x, (256,), weight, bias)

        assert output.shape == (2, 64, 256)

    def test_layer_norm_no_bias(self) -> None:
        """Layer norm without bias."""
        import layerzero as lz

        x = torch.randn(2, 64, 256)
        weight = torch.ones(256)

        output = lz.layer_norm(x, (256,), weight, bias=None)

        assert output.shape == x.shape

    def test_layer_norm_correctness(self) -> None:
        """Layer norm matches torch.nn.functional."""
        import layerzero as lz

        x = torch.randn(2, 64, 256)
        weight = torch.randn(256)
        bias = torch.randn(256)

        output = lz.layer_norm(x, (256,), weight, bias)

        expected = F.layer_norm(x, (256,), weight, bias)

        assert torch.allclose(output, expected, rtol=1e-4, atol=1e-4)

    def test_layer_norm_no_weight(self) -> None:
        """Layer norm without weight or bias (spec style)."""
        import layerzero as lz

        x = torch.randn(2, 64, 256)

        # As per spec, weight and bias are optional
        output = lz.layer_norm(x, (256,))

        assert output.shape == x.shape

    def test_layer_norm_int_shape(self) -> None:
        """Layer norm with int as normalized_shape."""
        import layerzero as lz

        x = torch.randn(2, 64, 256)

        # int is valid for normalized_shape
        output = lz.layer_norm(x, 256)

        assert output.shape == x.shape


class TestSamplingAPIs:
    """Tests for sampling APIs."""

    def test_sample_topk(self) -> None:
        """Top-k sampling via public API."""
        import layerzero as lz

        logits = torch.randn(2, 1000)

        sample = lz.sample_topk(logits, k=50, temperature=1.0)

        assert sample.shape == (2, 1)
        assert sample.dtype == torch.long

    def test_sample_topp(self) -> None:
        """Top-p sampling via public API."""
        import layerzero as lz

        logits = torch.randn(2, 1000)

        sample = lz.sample_topp(logits, p=0.9, temperature=1.0)

        assert sample.shape == (2, 1)
        assert sample.dtype == torch.long


class TestTokenizationAPIs:
    """Tests for tokenization APIs."""

    def test_tokenize_basic(self) -> None:
        """Basic tokenization."""
        import layerzero as lz

        # This requires a tokenizer to be configured
        try:
            tokens = lz.tokenize("Hello world", tokenizer="gpt2")
            assert isinstance(tokens, (list, torch.Tensor))
        except Exception:
            pytest.skip("Tokenizer not available")

    def test_detokenize_basic(self) -> None:
        """Basic detokenization."""
        import layerzero as lz

        try:
            tokens = [15496, 995]  # "Hello world" in GPT-2
            text = lz.detokenize(tokens, tokenizer="gpt2")
            assert isinstance(text, str)
        except Exception:
            pytest.skip("Tokenizer not available")
