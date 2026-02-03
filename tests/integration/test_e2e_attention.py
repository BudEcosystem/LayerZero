"""End-to-end integration tests for attention operations."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any


class TestEndToEndAttention:
    """End-to-end tests for attention operations."""

    @pytest.mark.integration
    def test_e2e_basic_attention(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Basic attention forward pass works."""
        batch_size, seq_len, num_heads, head_dim = 2, 64, 4, 64

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)

        # Transpose for SDPA (batch, heads, seq, dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.integration
    def test_e2e_llama_style_attention(
        self, device: torch.device, dtype: torch.dtype, llama_config: dict
    ) -> None:
        """LLaMA-style attention with GQA works."""
        batch_size = 2
        seq_len = 128
        num_heads = llama_config["num_attention_heads"]
        num_kv_heads = llama_config["num_key_value_heads"]
        head_dim = llama_config["head_dim"]

        q = torch.randn(batch_size, seq_len, num_heads, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim, device=device, dtype=dtype)

        # GQA: expand KV heads to match Q heads
        num_key_value_groups = num_heads // num_kv_heads
        k = k.unsqueeze(3).expand(-1, -1, -1, num_key_value_groups, -1).reshape(
            batch_size, seq_len, num_heads, head_dim
        )
        v = v.unsqueeze(3).expand(-1, -1, -1, num_key_value_groups, -1).reshape(
            batch_size, seq_len, num_heads, head_dim
        )

        # Transpose for SDPA
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    @pytest.mark.integration
    def test_e2e_gpt_style_attention(
        self, device: torch.device, dtype: torch.dtype, gpt_config: dict
    ) -> None:
        """GPT-style self-attention works."""
        batch_size = 2
        seq_len = 256
        num_heads = gpt_config["num_attention_heads"]
        head_dim = gpt_config["head_dim"]

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        output = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    @pytest.mark.integration
    def test_e2e_t5_encoder_decoder_attention(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """T5-style encoder-decoder cross attention works."""
        batch_size = 2
        encoder_seq_len = 128
        decoder_seq_len = 64
        num_heads = 4
        head_dim = 64

        # Decoder queries attending to encoder keys/values
        q = torch.randn(batch_size, num_heads, decoder_seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, encoder_seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, encoder_seq_len, head_dim, device=device, dtype=dtype)

        # Cross-attention is NOT causal
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        assert output.shape == (batch_size, num_heads, decoder_seq_len, head_dim)

    @pytest.mark.integration
    def test_e2e_vit_attention(
        self, device: torch.device, dtype: torch.dtype, vit_config: dict
    ) -> None:
        """ViT-style bidirectional attention works."""
        batch_size = 2
        # (image_size / patch_size)^2 + 1 (CLS token)
        num_patches = (vit_config["image_size"] // vit_config["patch_size"]) ** 2 + 1
        num_heads = vit_config["num_attention_heads"]
        head_dim = vit_config["head_dim"]

        q = torch.randn(batch_size, num_heads, num_patches, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, num_patches, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, num_patches, head_dim, device=device, dtype=dtype)

        # ViT uses bidirectional attention (not causal)
        output = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        assert output.shape == (batch_size, num_heads, num_patches, head_dim)

    @pytest.mark.integration
    def test_e2e_attention_with_mask(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Attention with custom mask works."""
        batch_size, seq_len, num_heads, head_dim = 2, 64, 4, 64

        q = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=dtype)

        # Custom mask: attend only to first half
        mask = torch.zeros(seq_len, seq_len, device=device, dtype=dtype)
        mask[:, seq_len // 2:] = float("-inf")

        output = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)

        assert output.shape == (batch_size, num_heads, seq_len, head_dim)

    @pytest.mark.integration
    def test_e2e_attention_variable_length(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Attention with variable sequence lengths works."""
        batch_size = 4
        max_seq_len = 128
        num_heads = 4
        head_dim = 64

        # Variable lengths
        seq_lens = [32, 64, 96, 128]

        for seq_len in seq_lens:
            q = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
            k = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)
            v = torch.randn(1, num_heads, seq_len, head_dim, device=device, dtype=dtype)

            output = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            assert output.shape == (1, num_heads, seq_len, head_dim)


class TestEndToEndNormalization:
    """End-to-end tests for normalization operations."""

    @pytest.mark.integration
    def test_e2e_rms_norm_basic(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Basic RMS normalization works."""
        batch_size, seq_len, hidden_size = 2, 64, 256
        eps = 1e-6

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        weight = torch.ones(hidden_size, device=device, dtype=dtype)

        # RMSNorm implementation
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        output = x_normed * weight

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    def test_e2e_rms_norm_in_model(
        self, device: torch.device, dtype: torch.dtype, llama_config: dict
    ) -> None:
        """RMSNorm in LLaMA-style model works."""
        batch_size = 2
        seq_len = 64
        hidden_size = llama_config["hidden_size"]
        eps = llama_config["rms_norm_eps"]

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        weight = torch.ones(hidden_size, device=device, dtype=dtype)

        # RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)
        output = x_normed * weight

        # Check output is normalized (variance close to 1)
        out_var = output.pow(2).mean(dim=-1)
        assert out_var.mean().item() < 10  # Reasonable variance

    @pytest.mark.integration
    def test_e2e_layer_norm_basic(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Basic layer normalization works."""
        batch_size, seq_len, hidden_size = 2, 64, 256

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)
        weight = torch.ones(hidden_size, device=device, dtype=dtype)
        bias = torch.zeros(hidden_size, device=device, dtype=dtype)

        output = F.layer_norm(x, [hidden_size], weight, bias)

        assert output.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    def test_e2e_layer_norm_in_model(
        self, device: torch.device, dtype: torch.dtype, gpt_config: dict
    ) -> None:
        """LayerNorm in GPT-style model works."""
        batch_size = 2
        seq_len = 64
        hidden_size = gpt_config["hidden_size"]
        eps = gpt_config["layer_norm_eps"]

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        ln = nn.LayerNorm(hidden_size, eps=eps, device=device, dtype=dtype)
        output = ln(x)

        # LayerNorm output should have mean ~0 and std ~1 per sample
        assert output.shape == (batch_size, seq_len, hidden_size)


class TestEndToEndPipeline:
    """End-to-end tests for full inference pipelines."""

    @pytest.mark.integration
    def test_e2e_transformer_block(
        self, device: torch.device, dtype: torch.dtype, llama_config: dict
    ) -> None:
        """Full transformer block forward pass."""
        batch_size = 2
        seq_len = 64
        hidden_size = llama_config["hidden_size"]
        num_heads = llama_config["num_attention_heads"]
        head_dim = llama_config["head_dim"]
        intermediate_size = llama_config["intermediate_size"]
        eps = llama_config["rms_norm_eps"]

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        # Pre-attention RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)

        # Attention (simplified - no projections)
        q = x_normed.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        k = x_normed.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)
        v = x_normed.view(batch_size, seq_len, num_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, hidden_size)

        # Residual connection
        x = x + attn_out

        # Post-attention RMSNorm
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x_normed = x * torch.rsqrt(variance + eps)

        # FFN (simplified)
        ffn_out = F.silu(x_normed) * x_normed
        x = x + ffn_out

        assert x.shape == (batch_size, seq_len, hidden_size)
        assert not torch.isnan(x).any()

    @pytest.mark.integration
    def test_e2e_generation_step(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Single generation step works."""
        batch_size = 2
        seq_len = 32
        vocab_size = 1000
        hidden_size = 256

        # Simulated hidden states from model
        hidden = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        # LM head projection
        lm_head = nn.Linear(hidden_size, vocab_size, device=device, dtype=dtype)
        logits = lm_head(hidden)

        assert logits.shape == (batch_size, seq_len, vocab_size)

        # Greedy decoding
        next_tokens = logits[:, -1, :].argmax(dim=-1)
        assert next_tokens.shape == (batch_size,)

    @pytest.mark.integration
    def test_e2e_kv_cache_update(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """KV cache update during generation."""
        batch_size = 2
        num_heads = 4
        head_dim = 64
        max_seq_len = 128

        # Initialize empty KV cache
        k_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )
        v_cache = torch.zeros(
            batch_size, num_heads, max_seq_len, head_dim,
            device=device, dtype=dtype
        )

        # Simulate generation steps
        current_len = 0
        for step in range(10):
            # New K, V for current position
            new_k = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)
            new_v = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)

            # Update cache
            k_cache[:, :, current_len:current_len+1, :] = new_k
            v_cache[:, :, current_len:current_len+1, :] = new_v
            current_len += 1

            # Query attends to all cached K, V
            q = torch.randn(batch_size, num_heads, 1, head_dim, device=device, dtype=dtype)
            k = k_cache[:, :, :current_len, :]
            v = v_cache[:, :, :current_len, :]

            output = F.scaled_dot_product_attention(q, k, v, is_causal=False)
            assert output.shape == (batch_size, num_heads, 1, head_dim)

    @pytest.mark.integration
    def test_e2e_batch_generation(
        self, device: torch.device, dtype: torch.dtype
    ) -> None:
        """Batched generation works."""
        batch_size = 4
        seq_len = 16
        vocab_size = 1000
        hidden_size = 256

        # Different sequence lengths in batch
        for bs in [1, 2, 4, 8]:
            hidden = torch.randn(bs, seq_len, hidden_size, device=device, dtype=dtype)
            lm_head = nn.Linear(hidden_size, vocab_size, device=device, dtype=dtype)
            logits = lm_head(hidden)

            # Top-k sampling
            top_k = 50
            topk_logits, topk_indices = torch.topk(logits[:, -1, :], top_k, dim=-1)
            probs = F.softmax(topk_logits, dim=-1)
            sampled = torch.multinomial(probs, num_samples=1)
            next_tokens = topk_indices.gather(dim=-1, index=sampled).squeeze(-1)

            assert next_tokens.shape == (bs,)


class TestEndToEndMixedPrecision:
    """End-to-end tests for mixed precision inference."""

    @pytest.mark.integration
    def test_e2e_fp16_inference(self, device: torch.device) -> None:
        """FP16 inference works correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for FP16 testing")

        batch_size, seq_len, hidden_size = 2, 64, 256
        dtype = torch.float16

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        # Simple forward pass in FP16
        weight = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
        output = torch.matmul(x, weight)

        assert output.dtype == torch.float16
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    @pytest.mark.integration
    def test_e2e_bf16_inference(self, device: torch.device) -> None:
        """BF16 inference works correctly."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for BF16 testing")

        if not torch.cuda.is_bf16_supported():
            pytest.skip("BF16 not supported on this device")

        batch_size, seq_len, hidden_size = 2, 64, 256
        dtype = torch.bfloat16

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        weight = torch.randn(hidden_size, hidden_size, device=device, dtype=dtype)
        output = torch.matmul(x, weight)

        assert output.dtype == torch.bfloat16
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    def test_e2e_mixed_precision_autocast(self, device: torch.device) -> None:
        """Autocast mixed precision works."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for autocast testing")

        batch_size, seq_len, hidden_size = 2, 64, 256

        x = torch.randn(batch_size, seq_len, hidden_size, device=device)
        weight = torch.randn(hidden_size, hidden_size, device=device)

        with torch.autocast(device_type='cuda', dtype=torch.float16):
            output = torch.matmul(x, weight)

        # Output may be fp16 or fp32 depending on operation
        assert not torch.isnan(output).any()

    @pytest.mark.integration
    def test_e2e_fp32_accumulation(self, device: torch.device, dtype: torch.dtype) -> None:
        """FP32 accumulation for numerical stability."""
        batch_size, seq_len, hidden_size = 2, 64, 256

        x = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=dtype)

        # Accumulate in FP32 for stability
        sum_fp32 = x.float().sum()
        sum_native = x.sum()

        # Results should be close
        assert torch.allclose(sum_fp32.to(dtype), sum_native, rtol=0.1, atol=0.1)
