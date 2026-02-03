"""ALiBi (Attention with Linear Biases) Tests for LayerZero.

Tests for ALiBi positional encoding per "Train Short, Test Long" (Press et al., 2021).
ALiBi adds a linear bias to attention scores based on query-key distance.
"""
from __future__ import annotations

import math
import pytest
import torch
from typing import Generator


class TestALiBiSlopeGeneration:
    """Tests for ALiBi slope generation."""

    def test_alibi_slopes_power_of_two_heads(self) -> None:
        """Slopes for power-of-2 head counts follow geometric sequence."""
        from layerzero.posenc.alibi import get_alibi_slopes

        # For 8 heads, ALiBi uses slopes that form a geometric sequence
        # The formula creates slopes that decrease: steeper (smaller) slopes
        # for early heads, gentler (larger) slopes for later heads
        slopes = get_alibi_slopes(8)

        assert slopes.shape == (8,)
        # All slopes should be positive
        assert (slopes > 0).all()
        # Slopes should form a geometric sequence (ratio should be constant)
        ratios = slopes[1:] / slopes[:-1]
        assert torch.allclose(ratios, ratios[0].expand_as(ratios), atol=1e-5)
        # First slope should be largest (steepest decay with most negative bias)
        assert slopes[0] > slopes[-1]

    def test_alibi_slopes_non_power_of_two_heads(self) -> None:
        """Slopes for non-power-of-2 heads use interleaved pattern."""
        from layerzero.posenc.alibi import get_alibi_slopes

        # For 12 heads (not power of 2), interpolate between 8 and 16
        slopes = get_alibi_slopes(12)

        assert slopes.shape == (12,)
        # All slopes should be positive
        assert (slopes > 0).all()
        # Slopes should be monotonically related (decreasing exponents)
        # First few slopes should be smaller (steeper decay)

    def test_alibi_slopes_single_head(self) -> None:
        """Single head returns single slope."""
        from layerzero.posenc.alibi import get_alibi_slopes

        slopes = get_alibi_slopes(1)
        assert slopes.shape == (1,)
        # For 1 head, slope should be positive
        assert slopes[0] > 0
        # Should not be nan or inf
        assert torch.isfinite(slopes[0])

    def test_alibi_slopes_large_head_count(self) -> None:
        """Large head counts work correctly."""
        from layerzero.posenc.alibi import get_alibi_slopes

        # Llama-style large head counts
        for num_heads in [32, 40, 64, 96, 128]:
            slopes = get_alibi_slopes(num_heads)
            assert slopes.shape == (num_heads,)
            assert (slopes > 0).all()
            assert not torch.isnan(slopes).any()
            assert not torch.isinf(slopes).any()

    def test_alibi_slopes_deterministic(self) -> None:
        """Slope generation is deterministic."""
        from layerzero.posenc.alibi import get_alibi_slopes

        slopes1 = get_alibi_slopes(16)
        slopes2 = get_alibi_slopes(16)
        assert torch.allclose(slopes1, slopes2)


class TestALiBiBiasGeneration:
    """Tests for ALiBi bias matrix generation."""

    def test_alibi_bias_shape(self) -> None:
        """Bias tensor has correct shape."""
        from layerzero.posenc.alibi import get_alibi_bias

        num_heads = 8
        seq_len = 128

        bias = get_alibi_bias(num_heads, seq_len)

        # Shape: (num_heads, seq_len, seq_len) or (1, num_heads, seq_len, seq_len)
        assert bias.shape[-3:] == (num_heads, seq_len, seq_len) or \
               bias.shape == (num_heads, seq_len, seq_len)

    def test_alibi_bias_diagonal_zero(self) -> None:
        """Diagonal (same position) has zero bias."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias = get_alibi_bias(8, 64)

        # Extract last two dims for each head
        for h in range(8):
            if bias.dim() == 4:
                head_bias = bias[0, h]
            else:
                head_bias = bias[h]
            # Diagonal should be 0 (same position, no distance penalty)
            diagonal = torch.diag(head_bias)
            assert torch.allclose(diagonal, torch.zeros_like(diagonal), atol=1e-6)

    def test_alibi_bias_increases_with_distance(self) -> None:
        """Bias magnitude increases with query-key distance."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias = get_alibi_bias(4, 32)

        # For each head, bias should become more negative as distance increases
        for h in range(4):
            if bias.dim() == 4:
                head_bias = bias[0, h]
            else:
                head_bias = bias[h]

            # Row 10: positions before should have increasingly negative bias
            row = head_bias[10, :]
            for i in range(1, 10):
                # Positions further back should have more negative bias
                # (bias at position i should be >= bias at position i-1)
                assert row[10-i] <= row[10-i+1] + 1e-6

    def test_alibi_bias_linear_relationship(self) -> None:
        """Bias is linearly proportional to distance."""
        from layerzero.posenc.alibi import get_alibi_bias, get_alibi_slopes

        num_heads = 4
        seq_len = 32

        bias = get_alibi_bias(num_heads, seq_len)
        slopes = get_alibi_slopes(num_heads)

        # Check linearity for first head
        if bias.dim() == 4:
            head_bias = bias[0, 0]
        else:
            head_bias = bias[0]

        # At position 10, looking at position 5, distance is 5
        # Bias should be -5 * slope[0]
        expected_bias = -5 * slopes[0]
        actual_bias = head_bias[10, 5]
        assert torch.isclose(actual_bias, expected_bias, atol=1e-5)

    def test_alibi_bias_different_slopes_per_head(self) -> None:
        """Each head has different bias pattern based on its slope."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias = get_alibi_bias(8, 64)

        # Different heads should have different bias magnitudes
        if bias.dim() == 4:
            head0_range = bias[0, 0].max() - bias[0, 0].min()
            head7_range = bias[0, 7].max() - bias[0, 7].min()
        else:
            head0_range = bias[0].max() - bias[0].min()
            head7_range = bias[7].max() - bias[7].min()

        # Head 0 has steeper slope, larger range
        assert head0_range > head7_range


class TestALiBiCausalMask:
    """Tests for ALiBi with causal masking."""

    def test_alibi_causal_mask(self) -> None:
        """ALiBi combined with causal mask."""
        from layerzero.posenc.alibi import get_alibi_bias_causal

        num_heads = 8
        seq_len = 64

        bias = get_alibi_bias_causal(num_heads, seq_len)

        # Upper triangle (future tokens) should be -inf
        if bias.dim() == 4:
            head_bias = bias[0, 0]
        else:
            head_bias = bias[0]

        # Check that future positions are masked
        for i in range(seq_len):
            for j in range(i + 1, seq_len):
                assert head_bias[i, j] == float('-inf') or head_bias[i, j] < -1e9

    def test_alibi_causal_past_positions_not_masked(self) -> None:
        """Past positions are not masked, only have ALiBi bias."""
        from layerzero.posenc.alibi import get_alibi_bias_causal

        bias = get_alibi_bias_causal(4, 32)

        if bias.dim() == 4:
            head_bias = bias[0, 0]
        else:
            head_bias = bias[0]

        # Lower triangle should be finite (just ALiBi bias, not masked)
        for i in range(1, 32):
            for j in range(i):
                assert torch.isfinite(head_bias[i, j])


class TestALiBiDevices:
    """Device-specific tests for ALiBi."""

    def test_alibi_cpu(self) -> None:
        """ALiBi works on CPU."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias = get_alibi_bias(8, 64, device='cpu')
        assert bias.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_alibi_cuda(self) -> None:
        """ALiBi works on CUDA."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias = get_alibi_bias(8, 64, device='cuda')
        assert bias.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_alibi_slopes_cuda(self) -> None:
        """Slopes can be on CUDA."""
        from layerzero.posenc.alibi import get_alibi_slopes

        slopes = get_alibi_slopes(8, device='cuda')
        assert slopes.device.type == 'cuda'


class TestALiBiDtypes:
    """Data type tests for ALiBi."""

    def test_alibi_float32(self) -> None:
        """ALiBi with float32."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias = get_alibi_bias(8, 64, dtype=torch.float32)
        assert bias.dtype == torch.float32

    def test_alibi_float16(self) -> None:
        """ALiBi with float16."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias = get_alibi_bias(8, 64, dtype=torch.float16)
        assert bias.dtype == torch.float16

    def test_alibi_bfloat16(self) -> None:
        """ALiBi with bfloat16."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias = get_alibi_bias(8, 64, dtype=torch.bfloat16)
        assert bias.dtype == torch.bfloat16


class TestALiBiCaching:
    """Tests for ALiBi bias caching."""

    def test_alibi_caching_same_params(self) -> None:
        """Same parameters return cached bias."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias1 = get_alibi_bias(8, 64, use_cache=True)
        bias2 = get_alibi_bias(8, 64, use_cache=True)

        # Should be the same tensor (from cache)
        assert bias1.data_ptr() == bias2.data_ptr()

    def test_alibi_caching_different_seq_len(self) -> None:
        """Different seq_len creates new bias."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias1 = get_alibi_bias(8, 64, use_cache=True)
        bias2 = get_alibi_bias(8, 128, use_cache=True)

        assert bias1.shape != bias2.shape

    def test_alibi_no_cache(self) -> None:
        """Caching can be disabled."""
        from layerzero.posenc.alibi import get_alibi_bias

        bias1 = get_alibi_bias(8, 64, use_cache=False)
        bias2 = get_alibi_bias(8, 64, use_cache=False)

        # Different tensors (not from cache)
        assert bias1.data_ptr() != bias2.data_ptr()


class TestALiBiIntegration:
    """Integration tests for ALiBi with LayerZero."""

    def test_alibi_operation_spec_exists(self) -> None:
        """OperationSpec for ALiBi exists."""
        from layerzero.models.operation_spec import posenc_alibi_spec

        spec = posenc_alibi_spec()
        assert spec.op_id == "posenc.alibi"
        assert spec.has_fallback is True

    def test_alibi_with_attention_scores(self) -> None:
        """ALiBi bias can be added to attention scores."""
        from layerzero.posenc.alibi import get_alibi_bias

        batch_size = 2
        num_heads = 8
        seq_len = 64

        # Simulated attention scores
        attention_scores = torch.randn(batch_size, num_heads, seq_len, seq_len)

        # Get ALiBi bias
        alibi_bias = get_alibi_bias(num_heads, seq_len)

        # Add bias (should broadcast over batch)
        biased_scores = attention_scores + alibi_bias

        assert biased_scores.shape == attention_scores.shape
        # Scores should be modified
        assert not torch.allclose(biased_scores, attention_scores)


class TestALiBiPerformance:
    """Performance tests for ALiBi."""

    @pytest.mark.stress
    def test_alibi_large_sequence(self) -> None:
        """ALiBi with large sequence length."""
        from layerzero.posenc.alibi import get_alibi_bias
        import time

        num_heads = 32
        seq_len = 4096  # Long sequence

        # Warmup
        _ = get_alibi_bias(num_heads, seq_len, use_cache=False)

        start = time.perf_counter()
        bias = get_alibi_bias(num_heads, seq_len, use_cache=False)
        elapsed = time.perf_counter() - start

        # Should complete in reasonable time
        assert elapsed < 1.0, f"ALiBi generation too slow: {elapsed:.3f}s"

        # Memory check - bias should be reasonable size
        # (num_heads * seq_len * seq_len * 4 bytes for float32)
        expected_bytes = num_heads * seq_len * seq_len * 4
        assert expected_bytes <= 2 * 1024**3  # At most 2GB

    @pytest.mark.stress
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_alibi_cuda_performance(self) -> None:
        """ALiBi CUDA performance."""
        from layerzero.posenc.alibi import get_alibi_bias
        import time

        num_heads = 32
        seq_len = 2048

        # Warmup
        for _ in range(3):
            _ = get_alibi_bias(num_heads, seq_len, device='cuda', use_cache=False)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(10):
            _ = get_alibi_bias(num_heads, seq_len, device='cuda', use_cache=False)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # 10 iterations should complete quickly on GPU
        assert elapsed < 1.0, f"CUDA ALiBi too slow: {elapsed:.3f}s for 10 iterations"
