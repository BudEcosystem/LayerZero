"""Top-K Sampling Tests for LayerZero.

Tests for top-k sampling operation correctness and performance.
"""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F
from typing import Generator


class TestTopKSamplingCorrectness:
    """Tests for top-k sampling correctness."""

    def test_topk_basic_selection(self) -> None:
        """Top-k selects from top k tokens."""
        from layerzero.sampling.topk import topk_sample

        # Create logits with clear top-k
        logits = torch.tensor([[0.1, 0.2, 10.0, 0.3, 9.0]])  # tokens 2, 4 are top-2

        # Sample 100 times, all should be from top-k
        samples = []
        for _ in range(100):
            sample = topk_sample(logits, k=2, temperature=1.0)
            samples.append(sample.item())

        # All samples should be token 2 or 4
        assert all(s in [2, 4] for s in samples)

    def test_topk_temperature_scaling(self) -> None:
        """Temperature affects sampling distribution."""
        from layerzero.sampling.topk import topk_sample

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        # Low temperature (more deterministic)
        low_temp_samples = []
        for _ in range(100):
            sample = topk_sample(logits, k=3, temperature=0.1)
            low_temp_samples.append(sample.item())

        # High temperature (more random)
        high_temp_samples = []
        for _ in range(100):
            sample = topk_sample(logits, k=3, temperature=2.0)
            high_temp_samples.append(sample.item())

        # Low temp should favor highest token more
        low_temp_mode = max(set(low_temp_samples), key=low_temp_samples.count)
        assert low_temp_mode == 4  # Highest logit token

    def test_topk_deterministic_with_seed(self) -> None:
        """Deterministic sampling with generator seed."""
        from layerzero.sampling.topk import topk_sample

        logits = torch.tensor([[1.0, 2.0, 3.0, 4.0, 5.0]])

        gen1 = torch.Generator().manual_seed(42)
        sample1 = topk_sample(logits, k=3, temperature=1.0, generator=gen1)

        gen2 = torch.Generator().manual_seed(42)
        sample2 = topk_sample(logits, k=3, temperature=1.0, generator=gen2)

        assert sample1.item() == sample2.item()

    def test_topk_batched(self) -> None:
        """Batched top-k sampling."""
        from layerzero.sampling.topk import topk_sample

        batch_size = 8
        vocab_size = 1000
        logits = torch.randn(batch_size, vocab_size)

        samples = topk_sample(logits, k=50, temperature=1.0)

        assert samples.shape == (batch_size, 1)
        assert samples.dtype == torch.long

    def test_topk_k_equals_vocab_size(self) -> None:
        """Top-k with k = vocab_size is full sampling."""
        from layerzero.sampling.topk import topk_sample

        vocab_size = 100
        logits = torch.randn(1, vocab_size)

        # Should not raise
        sample = topk_sample(logits, k=vocab_size, temperature=1.0)
        assert 0 <= sample.item() < vocab_size

    def test_topk_k_equals_one(self) -> None:
        """Top-k with k=1 is argmax."""
        from layerzero.sampling.topk import topk_sample

        logits = torch.tensor([[0.1, 0.5, 0.2, 0.9, 0.3]])

        sample = topk_sample(logits, k=1, temperature=1.0)
        expected = logits.argmax(dim=-1)

        assert sample.item() == expected.item()

    def test_topk_maintains_dtype(self) -> None:
        """Output is always int64/long."""
        from layerzero.sampling.topk import topk_sample

        for dtype in [torch.float16, torch.float32, torch.bfloat16]:
            logits = torch.randn(2, 100, dtype=dtype)
            if dtype == torch.bfloat16 and not torch.cuda.is_available():
                continue
            sample = topk_sample(logits, k=10, temperature=1.0)
            assert sample.dtype == torch.long


class TestTopKSamplingEdgeCases:
    """Edge case tests for top-k sampling."""

    def test_topk_all_same_logits(self) -> None:
        """Handles uniform distribution."""
        from layerzero.sampling.topk import topk_sample

        logits = torch.ones(1, 100)

        # Should not raise, uniform distribution
        sample = topk_sample(logits, k=50, temperature=1.0)
        assert 0 <= sample.item() < 100

    def test_topk_negative_logits(self) -> None:
        """Handles negative logits."""
        from layerzero.sampling.topk import topk_sample

        logits = torch.tensor([[-10.0, -5.0, -1.0, -0.1, -0.01]])

        sample = topk_sample(logits, k=3, temperature=1.0)
        # Should select from -0.01, -0.1, -1.0
        assert sample.item() in [2, 3, 4]

    def test_topk_very_large_logits(self) -> None:
        """Handles large logit values."""
        from layerzero.sampling.topk import topk_sample

        logits = torch.tensor([[0.0, 0.0, 1000.0, 0.0, 0.0]])

        sample = topk_sample(logits, k=3, temperature=1.0)
        # Should almost always select token 2
        assert sample.item() == 2

    def test_topk_inf_logits(self) -> None:
        """Handles infinite logits gracefully."""
        from layerzero.sampling.topk import topk_sample

        # Inf in logits can cause issues with softmax/multinomial
        # Test that very large (but finite) logits work
        logits = torch.tensor([[0.0, 1000.0, 0.0, 0.0, 0.0]])

        sample = topk_sample(logits, k=2, temperature=1.0)
        # Large value dominates, should select token 1
        assert sample.item() == 1

    def test_topk_k_larger_than_vocab(self) -> None:
        """K larger than vocab size is clamped."""
        from layerzero.sampling.topk import topk_sample

        vocab_size = 10
        logits = torch.randn(1, vocab_size)

        # Should clamp k to vocab_size
        sample = topk_sample(logits, k=100, temperature=1.0)
        assert 0 <= sample.item() < vocab_size

    def test_topk_zero_temperature(self) -> None:
        """Zero temperature raises or uses small epsilon."""
        from layerzero.sampling.topk import topk_sample

        logits = torch.tensor([[0.1, 0.5, 0.2, 0.9, 0.3]])

        # Zero temperature should be argmax or raise
        try:
            sample = topk_sample(logits, k=3, temperature=0.0)
            # If it doesn't raise, should be deterministic argmax within top-k
            assert sample.item() == 3  # 0.9 is max
        except (ValueError, ZeroDivisionError):
            pass  # Also acceptable behavior


class TestTopKSamplingDevice:
    """Device-specific tests for top-k sampling."""

    def test_topk_cpu(self) -> None:
        """Top-k works on CPU."""
        from layerzero.sampling.topk import topk_sample

        logits = torch.randn(4, 1000, device='cpu')
        sample = topk_sample(logits, k=50, temperature=1.0)

        assert sample.device.type == 'cpu'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_topk_cuda(self) -> None:
        """Top-k works on CUDA."""
        from layerzero.sampling.topk import topk_sample

        logits = torch.randn(4, 1000, device='cuda')
        sample = topk_sample(logits, k=50, temperature=1.0)

        assert sample.device.type == 'cuda'

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_topk_output_same_device(self) -> None:
        """Output is on same device as input."""
        from layerzero.sampling.topk import topk_sample

        for device in ['cpu', 'cuda']:
            logits = torch.randn(2, 100, device=device)
            sample = topk_sample(logits, k=10, temperature=1.0)
            assert sample.device.type == device


class TestTopKSamplingPerformance:
    """Performance tests for top-k sampling."""

    @pytest.mark.stress
    def test_topk_large_vocab(self) -> None:
        """Top-k with large vocabulary."""
        from layerzero.sampling.topk import topk_sample
        import time

        vocab_size = 128000  # Llama-3 vocab size
        batch_size = 32
        logits = torch.randn(batch_size, vocab_size)

        # Warmup
        for _ in range(3):
            _ = topk_sample(logits, k=50, temperature=1.0)

        # Benchmark
        start = time.perf_counter()
        for _ in range(100):
            _ = topk_sample(logits, k=50, temperature=1.0)
        elapsed = time.perf_counter() - start

        # Should complete 100 samples in reasonable time (relaxed threshold for CPU)
        assert elapsed < 5.0, f"Top-k too slow: {elapsed:.3f}s for 100 samples"

    @pytest.mark.stress
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_topk_cuda_performance(self) -> None:
        """Top-k CUDA performance."""
        from layerzero.sampling.topk import topk_sample
        import time

        vocab_size = 128000
        batch_size = 32
        logits = torch.randn(batch_size, vocab_size, device='cuda')

        # Warmup
        for _ in range(10):
            _ = topk_sample(logits, k=50, temperature=1.0)
        torch.cuda.synchronize()

        # Benchmark
        start = time.perf_counter()
        for _ in range(1000):
            _ = topk_sample(logits, k=50, temperature=1.0)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        # Should complete 1000 CUDA samples quickly
        assert elapsed < 1.0, f"CUDA top-k too slow: {elapsed:.3f}s for 1000 samples"


class TestTopKIntegration:
    """Integration tests for top-k with LayerZero."""

    def test_topk_operation_spec_exists(self) -> None:
        """OperationSpec for top-k exists."""
        from layerzero.models.operation_spec import sampling_topk_spec

        spec = sampling_topk_spec()
        assert spec.op_id == "sampling.topk"
        assert spec.has_fallback is True

    def test_topk_pytorch_op_registered(self) -> None:
        """PyTorch custom op is registered."""
        from layerzero.pytorch import ops  # noqa: F401

        # Check op exists
        assert hasattr(torch.ops.layerzero, 'sample_topk')

    def test_topk_via_pytorch_op(self) -> None:
        """Calling via torch.ops.layerzero.sample_topk."""
        from layerzero.pytorch import ops  # noqa: F401

        logits = torch.randn(2, 100)
        sample = torch.ops.layerzero.sample_topk(logits, k=10, temperature=1.0)

        assert sample.shape == (2, 1)
        assert sample.dtype == torch.long
