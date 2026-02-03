"""Speculative decoding verification.

This module provides:
- VerificationConfig for verification settings
- Rejection sampling implementation
- Greedy verification
- Acceptance rate tracking
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

import torch

from layerzero.device import GPUGeneration

logger = logging.getLogger(__name__)


@dataclass
class VerificationConfig:
    """Configuration for verification.

    Attributes:
        sampling_temperature: Sampling temperature (0 = greedy).
        top_p: Top-p sampling parameter.
        use_rejection_sampling: Use rejection sampling.
    """

    sampling_temperature: float = 1.0
    top_p: float = 1.0
    use_rejection_sampling: bool = True

    def is_greedy(self) -> bool:
        """Check if greedy sampling."""
        return self.sampling_temperature == 0.0


class VerificationKernelSelector:
    """Selects appropriate verification kernel."""

    def __init__(self, config: VerificationConfig) -> None:
        """Initialize selector.

        Args:
            config: Verification configuration.
        """
        self._config = config
        self._lock = RLock()

    def select_kernel(
        self,
        batch_size: int,
        speculation_length: int,
        gpu_generation: GPUGeneration,
    ) -> str:
        """Select verification kernel.

        Args:
            batch_size: Batch size.
            speculation_length: Speculation length.
            gpu_generation: GPU generation.

        Returns:
            Kernel ID.
        """
        with self._lock:
            if self._config.is_greedy():
                kernel_id = "greedy_verify"
            elif self._config.use_rejection_sampling:
                kernel_id = "rejection_sample_verify"
            else:
                kernel_id = "sampling_verify"

            logger.debug(
                "Selected verification kernel: %s for batch=%d, spec=%d",
                kernel_id,
                batch_size,
                speculation_length,
            )

            return kernel_id


class VerificationMetrics:
    """Tracks verification metrics."""

    def __init__(self) -> None:
        """Initialize metrics tracker."""
        self._lock = RLock()
        self._total_verifications = 0
        self._total_speculated = 0
        self._total_accepted = 0

    @property
    def total_verifications(self) -> int:
        """Total number of verifications."""
        with self._lock:
            return self._total_verifications

    def record_verification(
        self,
        speculation_length: int,
        num_accepted: int,
    ) -> None:
        """Record a verification result.

        Args:
            speculation_length: Number of speculated tokens.
            num_accepted: Number of accepted tokens.
        """
        with self._lock:
            self._total_verifications += 1
            self._total_speculated += speculation_length
            self._total_accepted += num_accepted

            logger.debug(
                "Verification: %d/%d accepted (%.1f%%)",
                num_accepted,
                speculation_length,
                100 * num_accepted / max(1, speculation_length),
            )

    def get_average_acceptance_rate(self) -> float:
        """Get average acceptance rate.

        Returns:
            Average acceptance rate (0.0 to 1.0).
        """
        with self._lock:
            if self._total_speculated == 0:
                return 0.0
            return self._total_accepted / self._total_speculated


# ============================================================================
# Batch Preparation
# ============================================================================


def prepare_verification_batch(
    draft_logits: torch.Tensor,
    target_logits: torch.Tensor,
) -> dict[str, Any]:
    """Prepare verification batch.

    Args:
        draft_logits: Draft model logits (batch, speculation_length, vocab).
        target_logits: Target model logits (batch, speculation_length + 1, vocab).

    Returns:
        Dictionary with prepared batch data.
    """
    batch_size, speculation_length, vocab_size = draft_logits.shape

    return {
        "draft_logits": draft_logits,
        "target_logits": target_logits,
        "batch_size": batch_size,
        "speculation_length": speculation_length,
        "vocab_size": vocab_size,
    }


# ============================================================================
# Rejection Sampling
# ============================================================================


def rejection_sample(
    draft_tokens: torch.Tensor,
    draft_probs: torch.Tensor,
    target_probs: torch.Tensor,
    config: VerificationConfig,
) -> dict[str, Any]:
    """Perform rejection sampling verification.

    Args:
        draft_tokens: Draft tokens (batch, speculation_length).
        draft_probs: Draft probabilities (batch, speculation_length, vocab).
        target_probs: Target probabilities (batch, speculation_length, vocab).
        config: Verification configuration.

    Returns:
        Dictionary with accepted_mask and num_accepted.
    """
    batch_size, speculation_length = draft_tokens.shape

    # Gather probabilities for draft tokens
    draft_token_probs = draft_probs.gather(
        dim=-1, index=draft_tokens.unsqueeze(-1)
    ).squeeze(-1)

    target_token_probs = target_probs.gather(
        dim=-1, index=draft_tokens.unsqueeze(-1)
    ).squeeze(-1)

    # Acceptance probability: min(1, p_target / p_draft)
    acceptance_probs = torch.minimum(
        torch.ones_like(target_token_probs),
        target_token_probs / (draft_token_probs + 1e-10),
    )

    # Sample acceptance
    random_vals = torch.rand_like(acceptance_probs)
    accepted_mask = random_vals < acceptance_probs

    # Find first rejection per sequence
    first_rejection = get_first_rejection_index(accepted_mask)

    # Mask out positions after first rejection
    position_indices = torch.arange(
        speculation_length, device=accepted_mask.device
    ).unsqueeze(0).expand(batch_size, -1)

    final_accepted_mask = (position_indices < first_rejection.unsqueeze(1))

    num_accepted = final_accepted_mask.sum(dim=1)

    return {
        "accepted_mask": final_accepted_mask,
        "num_accepted": num_accepted.sum().item(),
        "per_seq_accepted": num_accepted,
    }


def greedy_verify(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
) -> dict[str, Any]:
    """Perform greedy verification.

    Args:
        draft_tokens: Draft tokens (batch, speculation_length).
        target_logits: Target logits (batch, speculation_length + 1, vocab).

    Returns:
        Dictionary with accepted_mask.
    """
    batch_size, speculation_length = draft_tokens.shape

    # Get argmax from target logits
    target_tokens = target_logits[:, :speculation_length, :].argmax(dim=-1)

    # Check match
    accepted_mask = draft_tokens == target_tokens

    return {
        "accepted_mask": accepted_mask,
        "target_tokens": target_tokens,
    }


# ============================================================================
# Output Processing
# ============================================================================


def compute_accepted_tokens(accepted_mask: torch.Tensor) -> torch.Tensor:
    """Compute number of accepted tokens per sequence.

    Args:
        accepted_mask: Boolean mask (batch, speculation_length).

    Returns:
        Tensor of accepted counts per sequence.
    """
    # Find first rejection position
    first_rejection = get_first_rejection_index(accepted_mask)
    return first_rejection


def get_first_rejection_index(accepted_mask: torch.Tensor) -> torch.Tensor:
    """Get index of first rejected token per sequence.

    Args:
        accepted_mask: Boolean mask (batch, speculation_length).

    Returns:
        Tensor of first rejection indices (speculation_length if all accepted).
    """
    batch_size, speculation_length = accepted_mask.shape

    # Find first False in each row
    # Use cumsum trick: first position where cumsum != position + 1
    cumsum = accepted_mask.int().cumsum(dim=1)
    expected = torch.arange(
        1, speculation_length + 1,
        device=accepted_mask.device
    ).unsqueeze(0).expand(batch_size, -1)

    # Where cumsum differs from expected, we have a rejection
    mismatch = cumsum != expected

    # Find first mismatch
    first_mismatch_mask = mismatch & (
        torch.cat([
            torch.ones(batch_size, 1, device=mismatch.device, dtype=torch.bool),
            ~mismatch[:, :-1]
        ], dim=1)
    )

    # Get indices
    indices = torch.arange(
        speculation_length, device=accepted_mask.device
    ).unsqueeze(0).expand(batch_size, -1)

    # Where no mismatch, use speculation_length
    result = torch.where(
        mismatch.any(dim=1, keepdim=True).expand(batch_size, speculation_length),
        torch.where(
            first_mismatch_mask,
            indices,
            torch.full_like(indices, speculation_length)
        ),
        torch.full_like(indices, speculation_length)
    ).min(dim=1).values

    return result


def select_continuation_tokens(
    draft_tokens: torch.Tensor,
    target_logits: torch.Tensor,
    accepted_mask: torch.Tensor,
) -> dict[str, Any]:
    """Select tokens for continuing generation.

    Args:
        draft_tokens: Draft tokens (batch, speculation_length).
        target_logits: Target logits (batch, speculation_length + 1, vocab).
        accepted_mask: Accepted mask (batch, speculation_length).

    Returns:
        Dictionary with continuation tokens.
    """
    batch_size, speculation_length = draft_tokens.shape

    # Get number of accepted per sequence
    num_accepted = compute_accepted_tokens(accepted_mask)

    # Get bonus token from target at position num_accepted
    bonus_tokens = []
    for b in range(batch_size):
        pos = num_accepted[b].item()
        bonus_token = target_logits[b, pos, :].argmax().item()
        bonus_tokens.append(bonus_token)

    bonus_tokens_tensor = torch.tensor(bonus_tokens, device=draft_tokens.device)

    return {
        "num_new_tokens": num_accepted + 1,  # Accepted + bonus
        "bonus_tokens": bonus_tokens_tensor,
        "num_accepted": num_accepted,
    }


# ============================================================================
# Performance Estimation
# ============================================================================


def estimate_effective_speedup(
    acceptance_rate: float,
    draft_latency_ms: float,
    target_latency_ms: float,
    speculation_length: int,
) -> float:
    """Estimate effective speedup from speculative decoding.

    Args:
        acceptance_rate: Average acceptance rate (0 to 1).
        draft_latency_ms: Draft model latency in ms.
        target_latency_ms: Target model latency in ms.
        speculation_length: Number of speculated tokens.

    Returns:
        Estimated speedup factor.
    """
    # Expected tokens per speculation round
    expected_accepted = acceptance_rate * speculation_length

    # Time for speculation round: draft time + verification time
    # Assuming verification is one target forward pass
    speculation_time = (
        draft_latency_ms * speculation_length + target_latency_ms
    )

    # Baseline time: target latency per token
    baseline_time = target_latency_ms * (expected_accepted + 1)

    if speculation_time == 0:
        return 1.0

    speedup = baseline_time / speculation_time

    return max(1.0, speedup)  # At least 1x (no slowdown)
