"""Tests for speculative decoding verification."""
from __future__ import annotations

import pytest
import torch
from typing import Any

from layerzero.device import GPUGeneration


class TestVerificationConfig:
    """Tests for verification configuration."""

    def test_config_default_values(self) -> None:
        """Config has sensible defaults."""
        from layerzero.speculative.verification import VerificationConfig

        config = VerificationConfig()

        assert config.sampling_temperature == 1.0
        assert config.top_p == 1.0
        assert config.use_rejection_sampling is True

    def test_config_greedy_sampling(self) -> None:
        """Greedy sampling configuration."""
        from layerzero.speculative.verification import VerificationConfig

        config = VerificationConfig(
            sampling_temperature=0.0,
            top_p=1.0,
        )

        assert config.sampling_temperature == 0.0
        assert config.is_greedy()


class TestVerificationKernel:
    """Tests for verification kernel."""

    def test_select_verification_kernel(self, hopper_device: GPUGeneration) -> None:
        """Select verification kernel."""
        from layerzero.speculative.verification import (
            VerificationKernelSelector,
            VerificationConfig,
        )

        config = VerificationConfig()
        selector = VerificationKernelSelector(config)

        kernel_id = selector.select_kernel(
            batch_size=4,
            speculation_length=5,
            gpu_generation=hopper_device,
        )

        assert kernel_id is not None

    def test_verification_batch_format(self) -> None:
        """Verification batch format."""
        from layerzero.speculative.verification import prepare_verification_batch

        # Draft logits: (batch, speculation_length, vocab_size)
        draft_logits = torch.randn(4, 5, 32000)

        # Target logits: (batch, speculation_length + 1, vocab_size)
        target_logits = torch.randn(4, 6, 32000)

        batch = prepare_verification_batch(
            draft_logits=draft_logits,
            target_logits=target_logits,
        )

        assert "draft_logits" in batch
        assert "target_logits" in batch
        assert batch["batch_size"] == 4
        assert batch["speculation_length"] == 5


class TestRejectionSampling:
    """Tests for rejection sampling verification."""

    def test_rejection_sampling_basic(self) -> None:
        """Basic rejection sampling works."""
        from layerzero.speculative.verification import (
            rejection_sample,
            VerificationConfig,
        )

        torch.manual_seed(42)

        config = VerificationConfig(use_rejection_sampling=True)

        # Draft probabilities
        draft_probs = torch.softmax(torch.randn(4, 5, 100), dim=-1)

        # Target probabilities (ground truth)
        target_probs = torch.softmax(torch.randn(4, 5, 100), dim=-1)

        # Draft tokens sampled from draft probs
        draft_tokens = torch.multinomial(
            draft_probs.view(-1, 100), num_samples=1
        ).view(4, 5)

        result = rejection_sample(
            draft_tokens=draft_tokens,
            draft_probs=draft_probs,
            target_probs=target_probs,
            config=config,
        )

        assert "accepted_mask" in result
        assert "num_accepted" in result
        assert result["accepted_mask"].shape == (4, 5)

    def test_greedy_verification(self) -> None:
        """Greedy verification (no rejection sampling)."""
        from layerzero.speculative.verification import (
            greedy_verify,
            VerificationConfig,
        )

        config = VerificationConfig(sampling_temperature=0.0)

        # Draft tokens
        draft_tokens = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])

        # Target logits
        target_logits = torch.randn(2, 6, 100)
        # Make target match draft for first 3 tokens
        for b in range(2):
            for i in range(3):
                target_logits[b, i, draft_tokens[b, i]] = 100.0

        result = greedy_verify(
            draft_tokens=draft_tokens,
            target_logits=target_logits,
        )

        assert "accepted_mask" in result
        # First 3 should be accepted
        assert result["accepted_mask"][:, :3].all()


class TestVerificationOutput:
    """Tests for verification output handling."""

    def test_compute_accepted_tokens(self) -> None:
        """Compute number of accepted tokens per sequence."""
        from layerzero.speculative.verification import compute_accepted_tokens

        # Accepted mask: (batch_size, speculation_length)
        accepted_mask = torch.tensor([
            [True, True, True, False, False],  # 3 accepted
            [True, True, True, True, True],     # 5 accepted
            [True, False, False, False, False], # 1 accepted
        ])

        num_accepted = compute_accepted_tokens(accepted_mask)

        assert num_accepted.tolist() == [3, 5, 1]

    def test_get_first_rejection_index(self) -> None:
        """Get index of first rejected token."""
        from layerzero.speculative.verification import get_first_rejection_index

        accepted_mask = torch.tensor([
            [True, True, False, True, True],   # First rejection at 2
            [True, True, True, True, True],    # No rejection -> 5
            [False, True, True, True, True],   # First rejection at 0
        ])

        indices = get_first_rejection_index(accepted_mask)

        assert indices.tolist() == [2, 5, 0]

    def test_select_continuation_tokens(self) -> None:
        """Select tokens to continue generation."""
        from layerzero.speculative.verification import select_continuation_tokens

        # Draft tokens
        draft_tokens = torch.tensor([
            [10, 20, 30, 40, 50],
            [11, 21, 31, 41, 51],
        ])

        # Target logits for bonus token
        target_logits = torch.randn(2, 6, 100)
        target_logits[0, 3, 99] = 100.0  # Force token 99 at position 3
        target_logits[1, 5, 88] = 100.0  # Force token 88 at position 5

        accepted_mask = torch.tensor([
            [True, True, True, False, False],  # Accept first 3
            [True, True, True, True, True],     # Accept all 5
        ])

        continuation = select_continuation_tokens(
            draft_tokens=draft_tokens,
            target_logits=target_logits,
            accepted_mask=accepted_mask,
        )

        assert continuation["num_new_tokens"].tolist() == [4, 6]


class TestVerificationMetrics:
    """Tests for verification metrics tracking."""

    def test_acceptance_rate_tracking(self) -> None:
        """Track acceptance rate over time."""
        from layerzero.speculative.verification import VerificationMetrics

        metrics = VerificationMetrics()

        # Simulate multiple verifications
        metrics.record_verification(
            speculation_length=5,
            num_accepted=3,
        )
        metrics.record_verification(
            speculation_length=5,
            num_accepted=5,
        )
        metrics.record_verification(
            speculation_length=5,
            num_accepted=4,
        )

        avg_rate = metrics.get_average_acceptance_rate()

        # (3 + 5 + 4) / (5 + 5 + 5) = 12/15 = 0.8
        assert abs(avg_rate - 0.8) < 0.01

    def test_effective_speedup_estimate(self) -> None:
        """Estimate effective speedup from acceptance rate."""
        from layerzero.speculative.verification import estimate_effective_speedup

        speedup = estimate_effective_speedup(
            acceptance_rate=0.8,
            draft_latency_ms=5.0,
            target_latency_ms=50.0,
            speculation_length=5,
        )

        # Higher acceptance rate = higher speedup
        assert speedup > 1.0


class TestVerificationThreadSafety:
    """Tests for thread-safe verification."""

    def test_concurrent_verification(self) -> None:
        """Handle concurrent verifications safely."""
        import threading
        from layerzero.speculative.verification import VerificationMetrics

        metrics = VerificationMetrics()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    metrics.record_verification(
                        speculation_length=5,
                        num_accepted=3,
                    )
                    _ = metrics.get_average_acceptance_rate()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert metrics.total_verifications == 400
