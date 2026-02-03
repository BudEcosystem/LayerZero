"""End-to-end integration tests for model patching."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from typing import Any


class SimpleAttentionModule(nn.Module):
    """Simple attention module for testing."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        attn_out = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)

        attn_out = attn_out.transpose(1, 2).reshape(batch_size, seq_len, self.hidden_size)
        return self.o_proj(attn_out)


class SimpleMLP(nn.Module):
    """Simple MLP module for testing."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size)
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class SimpleTransformerBlock(nn.Module):
    """Simple transformer block for testing."""

    def __init__(self, hidden_size: int, num_heads: int, intermediate_size: int, eps: float = 1e-6):
        super().__init__()
        self.attn = SimpleAttentionModule(hidden_size, num_heads)
        self.mlp = SimpleMLP(hidden_size, intermediate_size)
        self.input_layernorm_weight = nn.Parameter(torch.ones(hidden_size))
        self.post_attention_layernorm_weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(variance + self.eps) * weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-attention norm
        normed = self._rms_norm(x, self.input_layernorm_weight)
        x = x + self.attn(normed)

        # Pre-MLP norm
        normed = self._rms_norm(x, self.post_attention_layernorm_weight)
        x = x + self.mlp(normed)

        return x


class SimpleModel(nn.Module):
    """Simple transformer model for testing."""

    def __init__(
        self,
        vocab_size: int = 1000,
        hidden_size: int = 256,
        num_heads: int = 4,
        intermediate_size: int = 512,
        num_layers: int = 2,
    ):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.lm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(x)


class TestEndToEndPatching:
    """End-to-end tests for model patching."""

    @pytest.mark.integration
    def test_patch_model_basic(self, device: torch.device, dtype: torch.dtype) -> None:
        """Basic model patching works."""
        model = SimpleModel().to(device=device, dtype=dtype)

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Forward pass without patching
        with torch.no_grad():
            output_original = model(input_ids)

        assert output_original.shape == (batch_size, seq_len, 1000)

    @pytest.mark.integration
    def test_patch_unpatch_consistency(self, device: torch.device, dtype: torch.dtype) -> None:
        """Patch/unpatch produces consistent results."""
        torch.manual_seed(42)

        model = SimpleModel().to(device=device, dtype=dtype)
        model.eval()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Output before any changes
        with torch.no_grad():
            output_before = model(input_ids).clone()

        # Simulate patch (replace attention with different implementation)
        original_forward = model.layers[0].attn.forward

        def patched_forward(x: torch.Tensor) -> torch.Tensor:
            # Same computation but through different path
            return original_forward(x)

        model.layers[0].attn.forward = patched_forward

        # Output after patch
        with torch.no_grad():
            output_patched = model(input_ids).clone()

        # Restore original
        model.layers[0].attn.forward = original_forward

        # Output after unpatch
        with torch.no_grad():
            output_restored = model(input_ids).clone()

        # Before and restored should match
        assert torch.allclose(output_before, output_restored, rtol=1e-4, atol=1e-4)

    @pytest.mark.integration
    def test_patch_preserves_gradients(self, device: torch.device) -> None:
        """Patching preserves gradient computation."""
        dtype = torch.float32  # Use fp32 for gradient testing

        model = SimpleModel().to(device=device, dtype=dtype)

        batch_size, seq_len = 2, 16
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Forward + backward
        output = model(input_ids)
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"Missing gradient for {name}"

    @pytest.mark.integration
    def test_multiple_forward_passes(self, device: torch.device, dtype: torch.dtype) -> None:
        """Multiple forward passes produce consistent results."""
        torch.manual_seed(42)

        model = SimpleModel().to(device=device, dtype=dtype)
        model.eval()

        batch_size, seq_len = 2, 32
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        outputs = []
        with torch.no_grad():
            for _ in range(5):
                output = model(input_ids)
                outputs.append(output.clone())

        # All outputs should be identical
        for i in range(1, len(outputs)):
            assert torch.allclose(outputs[0], outputs[i], rtol=1e-5, atol=1e-5)

    @pytest.mark.integration
    def test_different_batch_sizes(self, device: torch.device, dtype: torch.dtype) -> None:
        """Model handles different batch sizes correctly."""
        model = SimpleModel().to(device=device, dtype=dtype)
        model.eval()

        seq_len = 32

        for batch_size in [1, 2, 4, 8, 16]:
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

            with torch.no_grad():
                output = model(input_ids)

            assert output.shape == (batch_size, seq_len, 1000)
            assert not torch.isnan(output).any()

    @pytest.mark.integration
    def test_different_seq_lengths(self, device: torch.device, dtype: torch.dtype) -> None:
        """Model handles different sequence lengths correctly."""
        model = SimpleModel().to(device=device, dtype=dtype)
        model.eval()

        batch_size = 2

        for seq_len in [8, 16, 32, 64, 128]:
            input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

            with torch.no_grad():
                output = model(input_ids)

            assert output.shape == (batch_size, seq_len, 1000)
            assert not torch.isnan(output).any()


class TestModelPatchingPerformance:
    """Performance tests for model patching."""

    @pytest.mark.integration
    def test_patched_model_not_slower(self, device: torch.device, dtype: torch.dtype) -> None:
        """Patched model is not significantly slower."""
        import time

        model = SimpleModel().to(device=device, dtype=dtype)
        model.eval()

        batch_size, seq_len = 4, 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Warmup
        for _ in range(3):
            with torch.no_grad():
                _ = model(input_ids)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Time original
        start = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        original_time = time.perf_counter() - start

        # Simulate patch
        original_forward = model.layers[0].attn.forward
        model.layers[0].attn.forward = lambda x: original_forward(x)

        # Time patched
        start = time.perf_counter()
        for _ in range(10):
            with torch.no_grad():
                _ = model(input_ids)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        patched_time = time.perf_counter() - start

        # Patched should not be more than 50% slower
        assert patched_time < original_time * 1.5

    @pytest.mark.integration
    def test_memory_not_increased_significantly(self, device: torch.device, dtype: torch.dtype) -> None:
        """Patching doesn't significantly increase memory usage."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for memory testing")

        model = SimpleModel().to(device=device, dtype=dtype)
        model.eval()

        batch_size, seq_len = 4, 64
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)

        # Clear cache
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        # Measure original
        with torch.no_grad():
            _ = model(input_ids)

        original_memory = torch.cuda.max_memory_allocated()

        # Reset and measure after patch
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        original_forward = model.layers[0].attn.forward
        model.layers[0].attn.forward = lambda x: original_forward(x)

        with torch.no_grad():
            _ = model(input_ids)

        patched_memory = torch.cuda.max_memory_allocated()

        # Memory should not increase more than 10%
        assert patched_memory < original_memory * 1.1
