"""Top-P (Nucleus) Sampling Tests for LayerZero.

Tests for top-p sampling operation correctness and performance.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from typing import Generator


class TestTopPSamplingCorrectness:
    """Tests for top-p sampling correctness."""

    def test_topp_basic_selection(self) -> None:
        """Top-p selects from nucleus with cumulative prob >= p."""
        from layerzero.sampling.topp import topp_sample

        # Create logits where first 2 tokens have ~95% probability
        logits = torch.tensor([[10.0, 9.0, 0.0, 0.0, 0.0]])

        # With p=0.9, should mostly select from top 2 tokens
        samples = []
        for _ in range(100):
            sample = topp_sample(logits, p=0.9, temperature=1.0)
            samples.append(sample.item())

        # Most samples should be token 0 or 1
        top2_count = sum(1 for s in samples if s in [0, 1])
        assert top2_count >= 95  # Allow some due to probability

    def test_topp_temperature_scaling(self) -> None:
        """Temperature affects top-p distribution."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Low temperature samples
        low_temp_samples = []
        for _ in range(100):
            sample = topp_sample(logits, p=0.9, temperature=0.1)
            low_temp_samples.append(sample.item())

        # High temperature samples
        high_temp_samples = []
        for _ in range(100):
            sample = topp_sample(logits, p=0.9, temperature=2.0)
            high_temp_samples.append(sample.item())

        # Low temp should be more concentrated on highest
        low_temp_mode_count = low_temp_samples.count(4)
        high_temp_mode_count = high_temp_samples.count(4)
        assert low_temp_mode_count >= high_temp_mode_count

    def test_topp_deterministic_with_seed(self) -> None:
        """Deterministic sampling with generator seed."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        gen1 = torch.Generator().manual_seed(123)
        sample1 = topp_sample(logits, p=0.9, temperature=1.0, generator=gen1)

        gen2 = torch.Generator().manual_seed(123)
        sample2 = topp_sample(logits, p=0.9, temperature=1.0, generator=gen2)

        assert sample1.item() == sample2.item()

    def test_topp_batched(self) -> None:
        """Batched top-p sampling."""
        from layerzero.sampling.topp import topp_sample

        batch_size = 8
        vocab_size = 1000
        logits = torch.randn(batch_size, vocab_size)

        samples = topp_sample(logits, p=0.9, temperature=1.0)

        assert samples.shape == (batch_size, 1)
        assert samples.dtype == torch.long

    def test_topp_p_equals_one(self) -> None:
        """Top-p with p=1.0 is full distribution sampling."""
        from layerzero.sampling.topp import topp_sample

        vocab_size = 100
        logits = torch.randn(1, vocab_size)

        # All tokens should be possible
        samples = set()
        for _ in range(500):
            sample = topp_sample(logits, p=1.0, temperature=1.0)
            samples.add(sample.item())

        # Should see variety of tokens
        assert len(samples) > 10

    def test_topp_p_very_small(self) -> None:
        """Top-p with very small p is near-greedy."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.tensor([[0.1, 0.2, 0.3, 10.0, 0.4]])

        samples = []
        for _ in range(100):
            sample = topp_sample(logits, p=0.01, temperature=1.0)
            samples.append(sample.item())

        # Should almost always select token 3 (highest)
        assert samples.count(3) >= 95

    def test_topp_maintains_dtype(self) -> None:
        """Output is always int64/long."""
        from layerzero.sampling.topp import topp_sample

        for dtype in [torch.float16, torch.float32]:
            logits = torch.randn(2, 100, dtype=dtype)
            sample = topp_sample(logits, p=0.9, temperature=1.0)
            assert sample.dtype == torch.long


class TestTopPSamplingEdgeCases:
    """Edge case tests for top-p sampling."""

    def test_topp_uniform_distribution(self) -> None:
        """Handles uniform distribution."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.zeros(1, 100)

        # With p=0.5, should include ~50 tokens
        sample = topp_sample(logits, p=0.5, temperature=1.0)
        assert 0 <= sample.item() < 100

    def test_topp_negative_logits(self) -> None:
        """Handles negative logits."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.tensor([[-10.0, -5.0, -1.0, -0.1, -0.01]])

        sample = topp_sample(logits, p=0.9, temperature=1.0)
        assert 0 <= sample.item() < 5

    def test_topp_single_dominant_token(self) -> None:
        """Single dominant token gets selected."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.tensor([[0.0, 0.0, 100.0, 0.0, 0.0]])

        samples = []
        for _ in range(50):
            sample = topp_sample(logits, p=0.9, temperature=1.0)
            samples.append(sample.item())

        # Token 2 should dominate
        assert all(s == 2 for s in samples)

    def test_topp_two_equal_high_tokens(self) -> None:
        """Two equal high tokens share probability."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.tensor([[0.0, 10.0, 10.0, 0.0, 0.0]])

        samples = []
        for _ in range(100):
            sample = topp_sample(logits, p=0.9, temperature=1.0)
            samples.append(sample.item())

        # Should see both tokens 1 and 2
        assert 1 in samples
        assert 2 in samples
        # Most should be tokens 1 or 2
        top2_count = sum(1 for s in samples if s in [1, 2])
        assert top2_count >= 95

    def test_topp_p_zero(self) -> None:
        """P=0 should raise or return greedy."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.tensor([[0.1, 0.5, 0.2, 0.9, 0.3]])

        try:
            sample = topp_sample(logits, p=0.0, temperature=1.0)
            # If doesn't raise, should be greedy
            assert sample.item() == 3
        except ValueError:
            pass  # Also acceptable

    def test_topp_p_greater_than_one(self) -> None:
        """P > 1 should be clamped to 1."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.randn(1, 50)

        # Should not raise, clamp to p=1.0
        sample = topp_sample(logits, p=1.5, temperature=1.0)
        assert 0 <= sample.item() < 50


class TestTopPSamplingDevice:
    """Device-specific tests for top-p sampling."""

    def test_topp_cpu(self) -> None:
        """Top-p works on CPU."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.randn(4, 1000, device='cpu')
        sample = topp_sample(logits, p=0.9, temperature=1.0)

        assert sample.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_topp_cuda(self) -> None:
        """Top-p works on CUDA."""
        from layerzero.sampling.topp import topp_sample

        logits = torch.randn(4, 1000, device='cuda')
        sample = topp_sample(logits, p=0.9, temperature=1.0)

        assert sample.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_topp_output_same_device(self) -> None:
        """Output is on same device as input."""
        from layerzero.sampling.topp import topp_sample

        for device in ['cpu', 'cuda']:
            logits = torch.randn(2, 100, device=device)
            sample = topp_sample(logits, p=0.9, temperature=1.0)
            assert sample.device.type == device


class TestTopPSamplingPerformance:
    """Performance tests for top-p sampling."""

    @pytest.mark.stress
    def test_topp_large_vocab(self) -> None:
        """Top-p with large vocabulary."""
        from layerzero.sampling.topp import topp_sample
        import time

        vocab_size = 128000  # Llama-3 vocab size
        batch_size = 32
        logits = torch.randn(batch_size, vocab_size)

        # Warmup
        for _ in range(3):
            _ = topp_sample(logits, p=0.9, temperature=1.0)

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = topp_sample(logits, p=0.9, temperature=1.0)
        elapsed = time.perf_counter() - start

        # Top-p requires full sort, so allow more time (relaxed for CPU)
        assert elapsed < 30.0, f"Top-p too slow: {elapsed:.3f}s for 100 samples"

    @pytest.mark.stress
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_topp_cuda_performance(self) -> None:
        """Top-p CUDA performance."""
        from layerzero.sampling.topp import topp_sample
        import time

        vocab_size = 128000
        batch_size = 32
        logits = torch.randn(batch_size, vocab_size, device='cuda')

        # Warmup
        for _ in range(10):
            _ = topp_sample(logits, p=0.9, temperature=1.0)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(1000):
            _ = topp_sample(logits, p=0.9, temperature=1.0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Top-p requires full sort, allow reasonable time on CUDA
        assert elapsed < 10.0, f"CUDA top-p too slow: {elapsed:.3f}s for 1000 samples"


class TestTopPIntegration:
    """Integration tests for top-p with LayerZero."""

    def test_topp_operation_spec_exists(self) -> None:
        """OperationSpec for top-p exists."""
        from layerzero.models.operation_spec import sampling_topp_spec

        spec = sampling_topp_spec()
        assert spec.op_id == "sampling.topp"
        assert spec.has_fallback is True

    def test_topp_pytorch_op_registered(self) -> None:
        """PyTorch custom op is registered."""
        from layerzero.pytorch import ops  # noqa: F401

        # Check op exists
        assert hasattr(torch.ops.layerzero, 'sample_topp')

    def test_topp_via_pytorch_op(self) -> None:
        """Calling via torch.ops.layerzero.sample_topp."""
        from layerzero.pytorch import ops  # noqa: F401

        logits = torch.randn(2, 100)
        sample = torch.ops.layerzero.sample_topp(logits, p=0.9, temperature=1.0)

        assert sample.shape == (2, 1)
        assert sample.dtype == torch.long


class TestCombinedSampling:
    """Tests for combined top-k + top-p sampling."""

    def test_combined_topk_topp(self) -> None:
        """Combined top-k and top-p sampling."""
        from layerzero.sampling.combined import topk_topp_sample

        logits = torch.randn(4, 1000)

        # First apply top-k, then top-p within top-k
        sample = topk_topp_sample(logits, k=50, p=0.9, temperature=1.0)

        assert sample.shape == (4, 1)
        assert sample.dtype == torch.long

    def test_combined_respects_both_constraints(self) -> None:
        """Combined sampling respects both k and p."""
        from layerzero.sampling.combined import topk_topp_sample

        # Create logits where top-3 have 99% of probability
        logits = torch.tensor([[10.0, 9.5, 9.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        samples = []
        for _ in range(100):
            sample = topk_topp_sample(logits, k=5, p=0.9, temperature=1.0)
            samples.append(sample.item())

        # All samples should be from top-3 (intersection of top-5 and p=0.9 nucleus)
        assert all(s in [0, 1, 2] for s in samples)
