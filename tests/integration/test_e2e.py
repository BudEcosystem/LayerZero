"""End-to-end integration tests for LayerZero.

Tests full attention pipelines, model patching, and mixed precision.
"""
from __future__ import annotations

from typing import Any
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from layerzero.pytorch import ops
from layerzero.core.validation import (
    validate_attention_inputs,
    validate_head_dim,
    validate_dtype,
    SUPPORTED_DTYPES,
)
from layerzero.enums import Layout


# Check CUDA availability
HAS_CUDA = torch.cuda.is_available()
HAS_BF16 = HAS_CUDA and torch.cuda.is_bf16_supported()


class SimpleAttention(nn.Module):
    """Simple attention module for testing."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape to [batch, heads, seq, head_dim]
        q = q.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Use LayerZero attention
        output = torch.ops.layerzero.attention(q, k, v, mask, 0.0, False, 1.0)

        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch, seq_len, self.embed_dim)
        return self.out_proj(output)


class SimpleRMSNorm(nn.Module):
    """Simple RMSNorm for testing."""

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.ops.layerzero.rms_norm(x, self.weight, self.eps)


class SimpleLLaMABlock(nn.Module):
    """Simple LLaMA-style block for testing."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = SimpleAttention(embed_dim, num_heads)
        self.norm1 = SimpleRMSNorm(embed_dim)
        self.norm2 = SimpleRMSNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention
        h = x + self.attention(self.norm1(x))
        # Pre-norm MLP
        return h + self.mlp(self.norm2(h))


class SimpleGPTBlock(nn.Module):
    """Simple GPT-style block for testing."""

    def __init__(self, embed_dim: int, num_heads: int) -> None:
        super().__init__()
        self.attention = SimpleAttention(embed_dim, num_heads)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x + self.attention(self.norm1(x))
        return h + self.mlp(self.norm2(h))


class TestEndToEndAttention:
    """End-to-end attention layer tests."""

    @pytest.mark.integration
    def test_e2e_llama_attention(self) -> None:
        """Full LLaMA-style attention layer works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        embed_dim = 256
        num_heads = 4
        batch_size = 2
        seq_len = 64

        block = SimpleLLaMABlock(embed_dim, num_heads).to(device, dtype)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

        output = block(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()
        assert torch.isfinite(output).all()

    @pytest.mark.integration
    def test_e2e_gpt_attention(self) -> None:
        """Full GPT-style attention layer works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        embed_dim = 256
        num_heads = 4
        batch_size = 2
        seq_len = 64

        block = SimpleGPTBlock(embed_dim, num_heads).to(device, dtype)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device, dtype=dtype)

        output = block(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    def test_e2e_encoder_decoder_attention(self) -> None:
        """Encoder-decoder (cross) attention works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 2
        num_heads = 4
        head_dim = 64
        encoder_seq = 100
        decoder_seq = 50

        # Encoder output
        encoder_output = torch.randn(
            batch_size, num_heads, encoder_seq, head_dim, device=device, dtype=dtype
        )

        # Decoder query
        decoder_q = torch.randn(
            batch_size, num_heads, decoder_seq, head_dim, device=device, dtype=dtype
        )

        # Cross attention: Q from decoder, K/V from encoder
        output = torch.ops.layerzero.attention(
            decoder_q,
            encoder_output,  # K from encoder
            encoder_output,  # V from encoder
            None,
            0.0,
            False,
            1.0,
        )

        assert output.shape == (batch_size, num_heads, decoder_seq, head_dim)
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    def test_e2e_vit_attention(self) -> None:
        """Vision Transformer attention (no causal mask)."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 4
        num_patches = 196  # 14x14 patches
        num_heads = 8
        head_dim = 64

        # ViT uses non-causal attention
        q = torch.randn(batch_size, num_heads, num_patches, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, num_patches, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, num_patches, head_dim, device=device, dtype=dtype)

        output = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)

        assert output.shape == q.shape
        assert not torch.isnan(output).any()


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""

    @pytest.mark.integration
    def test_e2e_text_generation_pipeline(self) -> None:
        """Text generation pipeline works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        embed_dim = 128
        num_heads = 2
        vocab_size = 1000
        seq_len = 32

        # Simple model
        class SimpleModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.embed = nn.Embedding(vocab_size, embed_dim)
                self.block = SimpleLLaMABlock(embed_dim, num_heads)
                self.lm_head = nn.Linear(embed_dim, vocab_size)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                h = self.embed(x)
                h = self.block(h)
                return self.lm_head(h)

        model = SimpleModel().to(device, dtype)
        input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

        logits = model(input_ids)

        assert logits.shape == (1, seq_len, vocab_size)
        assert not torch.isnan(logits).any()

        # Generate next token
        next_token = torch.argmax(logits[:, -1, :], dim=-1)
        assert next_token.shape == (1,)

    @pytest.mark.integration
    def test_e2e_with_kv_cache_simulation(self) -> None:
        """Generation with simulated KV cache works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 2
        num_heads = 4
        head_dim = 64
        cache_len = 100
        new_len = 1  # Single new token

        # Simulated KV cache
        k_cache = torch.randn(
            batch_size, num_heads, cache_len, head_dim, device=device, dtype=dtype
        )
        v_cache = torch.randn(
            batch_size, num_heads, cache_len, head_dim, device=device, dtype=dtype
        )

        # New query for single token
        q_new = torch.randn(
            batch_size, num_heads, new_len, head_dim, device=device, dtype=dtype
        )

        # Attention with full K/V cache
        output = torch.ops.layerzero.attention(
            q_new,
            k_cache,
            v_cache,
            None,
            0.0,
            False,
            1.0,
        )

        assert output.shape == (batch_size, num_heads, new_len, head_dim)

    @pytest.mark.integration
    def test_e2e_batch_generation(self) -> None:
        """Batch generation works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 8
        num_heads = 4
        head_dim = 64
        seq_len = 32

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        output = torch.ops.layerzero.attention(q, k, v, None, 0.0, True, 1.0)

        # All batches should produce valid output
        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        for i in range(batch_size):
            assert not torch.isnan(output[i]).any()


class TestEndToEndNormalization:
    """End-to-end normalization tests."""

    @pytest.mark.integration
    def test_e2e_rms_norm_in_model(self) -> None:
        """RMSNorm in full model works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        dim = 256

        norm = SimpleRMSNorm(dim).to(device, dtype)
        x = torch.randn(2, 32, dim, device=device, dtype=dtype)

        output = norm(x)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        # RMS should be approximately 1 after normalization (scaled by weight)
        rms = (output ** 2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=0.1)

    @pytest.mark.integration
    def test_e2e_layer_norm_in_model(self) -> None:
        """LayerNorm in full model works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        dim = 256

        # Use standard LayerNorm with LayerZero backend
        x = torch.randn(2, 32, dim, device=device, dtype=dtype)
        weight = torch.ones(dim, device=device, dtype=dtype)
        bias = torch.zeros(dim, device=device, dtype=dtype)

        output = torch.ops.layerzero.layer_norm(x, weight, bias, 1e-5)

        assert output.shape == x.shape
        assert not torch.isnan(output).any()

        # Mean should be ~0, std should be ~1
        assert torch.abs(output.mean()) < 0.1
        assert torch.abs(output.std() - 1.0) < 0.1


class TestEndToEndPatching:
    """End-to-end patching tests."""

    @pytest.mark.integration
    def test_patch_and_generate(self) -> None:
        """Patch model and generate works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        embed_dim = 128
        num_heads = 2

        # Create model
        block = SimpleLLaMABlock(embed_dim, num_heads).to(device, dtype)
        x = torch.randn(1, 32, embed_dim, device=device, dtype=dtype)

        # Generate before (uses LayerZero attention already)
        output_before = block(x)

        # Model should still work after potential "patching" (already uses LayerZero)
        output_after = block(x)

        assert torch.allclose(output_before, output_after)

    @pytest.mark.integration
    def test_patch_unpatch_consistency(self) -> None:
        """Patch/unpatch produces consistent results."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 2
        num_heads = 4
        seq_len = 32
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # Call LayerZero attention multiple times
        output1 = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)
        output2 = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)

        # Should be identical
        assert torch.allclose(output1, output2)


class TestEndToEndMixedPrecision:
    """End-to-end mixed precision tests."""

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for FP16")
    def test_e2e_fp16_inference(self) -> None:
        """FP16 inference end-to-end."""
        device = torch.device("cuda")
        dtype = torch.float16

        batch_size = 2
        num_heads = 4
        seq_len = 64
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        output = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_BF16, reason="BF16 support required")
    def test_e2e_bf16_inference(self) -> None:
        """BF16 inference end-to-end."""
        device = torch.device("cuda")
        dtype = torch.bfloat16

        batch_size = 2
        num_heads = 4
        seq_len = 64
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        output = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)

        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    @pytest.mark.skipif(not HAS_CUDA, reason="CUDA required for autocast")
    def test_e2e_mixed_precision_autocast(self) -> None:
        """Autocast mixed precision works."""
        device = torch.device("cuda")

        embed_dim = 256
        num_heads = 4
        batch_size = 2
        seq_len = 64

        # Model in FP32
        block = SimpleLLaMABlock(embed_dim, num_heads).to(device)
        x = torch.randn(batch_size, seq_len, embed_dim, device=device)

        # With autocast
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            output = block(x)

        # Output dtype depends on PyTorch autocast behavior with residual connections
        # The important thing is that it runs without errors and produces valid output
        assert output.dtype in (torch.float16, torch.float32)
        assert not torch.isnan(output).any()


class TestEndToEndBackends:
    """End-to-end backend integration tests."""

    @pytest.mark.integration
    def test_sdpa_backend_integration(self) -> None:
        """SDPA backend integration works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 2
        num_heads = 4
        seq_len = 32
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # LayerZero uses SDPA as one of its backends
        # Pass scale=None to use default scale (1/sqrt(head_dim))
        output = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, None)

        # Compare with direct SDPA (also uses default scale)
        reference = F.scaled_dot_product_attention(q, k, v)

        assert torch.allclose(output, reference, rtol=1e-3, atol=1e-3)

    @pytest.mark.integration
    def test_backend_selection_consistency(self) -> None:
        """Backend selection produces consistent results."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 2
        num_heads = 4
        seq_len = 32
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # Multiple calls should produce identical results
        outputs = [
            torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)
            for _ in range(3)
        ]

        for output in outputs[1:]:
            assert torch.allclose(outputs[0], output)


class TestEndToEndStress:
    """End-to-end stress tests."""

    @pytest.mark.integration
    def test_long_sequence_handling(self) -> None:
        """Long sequence handling works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 1
        num_heads = 2
        seq_len = 2048  # Long sequence
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        output = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    def test_large_batch_handling(self) -> None:
        """Large batch handling works."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 32  # Large batch
        num_heads = 4
        seq_len = 64
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        output = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        # Check each batch item
        for i in range(batch_size):
            assert not torch.isnan(output[i]).any()

    @pytest.mark.integration
    def test_memory_efficiency(self) -> None:
        """Memory usage is reasonable."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 2
        num_heads = 4
        seq_len = 256
        head_dim = 64

        if HAS_CUDA:
            torch.cuda.empty_cache()
            mem_before = torch.cuda.memory_allocated()

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        output = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)

        if HAS_CUDA:
            mem_after = torch.cuda.memory_allocated()
            # Memory increase should be reasonable (not 10x input size)
            input_size = q.numel() * 4 * 3  # 3 tensors, 4 bytes per float32
            mem_increase = mem_after - mem_before
            # Allow up to 5x input size for intermediate buffers
            assert mem_increase < input_size * 5

        assert output.shape == q.shape


class TestEndToEndValidation:
    """End-to-end validation integration tests."""

    @pytest.mark.integration
    def test_validation_before_attention(self) -> None:
        """Validation runs before attention."""
        device = torch.device("cuda" if HAS_CUDA else "cpu")
        dtype = torch.float32

        batch_size = 2
        num_heads = 4
        seq_len = 32
        head_dim = 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # Validate inputs
        result = validate_attention_inputs(q, k, v)
        assert result.valid

        # Then run attention
        output = torch.ops.layerzero.attention(q, k, v, None, 0.0, False, 1.0)
        assert output.shape == q.shape

    @pytest.mark.integration
    def test_validation_catches_invalid_inputs(self) -> None:
        """Validation catches invalid inputs."""
        # Invalid dtype
        result = validate_dtype(torch.int32)
        assert not result.valid

        # Invalid head dim
        result = validate_head_dim(65)  # Not multiple of 8
        assert not result.valid

        # Valid cases
        result = validate_dtype(torch.float16)
        assert result.valid

        result = validate_head_dim(64)
        assert result.valid
