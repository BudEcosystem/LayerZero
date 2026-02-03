"""
Top-K Sampling Implementation

Selects tokens from the top-k highest probability tokens.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

from layerzero.sampling.temperature import apply_temperature


def topk_sample(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Sample from top-k tokens.

    Filters logits to only include the top-k highest values,
    then samples from the resulting distribution.

    Args:
        logits: Input logits tensor of shape (batch_size, vocab_size) or
                (batch_size, seq_len, vocab_size).
        k: Number of top tokens to consider. Clamped to vocab_size if larger.
        temperature: Temperature for scaling logits before softmax.
                    Values < 1.0 make sampling more deterministic.
        generator: Optional PyTorch random generator for reproducibility.

    Returns:
        Sampled token indices tensor of shape (batch_size, 1) or
        (batch_size, seq_len, 1).

    Note:
        - If k=1, this is equivalent to argmax (greedy decoding).
        - If k >= vocab_size, this samples from the full distribution.
        - NaN in logits will propagate to output probabilities.
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    # Handle temperature
    if temperature <= 0:
        # Zero/negative temperature: greedy within top-k
        # Just use very small temperature
        temperature = 1e-10

    # Get vocab size and clamp k
    vocab_size = logits.shape[-1]
    k = min(k, vocab_size)

    # Apply temperature scaling
    scaled_logits = apply_temperature(logits, temperature)

    # Get top-k values and indices
    topk_values, topk_indices = torch.topk(scaled_logits, k, dim=-1)

    # Convert to probabilities
    probs = F.softmax(topk_values, dim=-1)

    # Sample from the top-k distribution
    # multinomial requires 2D input, handle batched case
    original_shape = probs.shape[:-1]  # All dims except last
    probs_flat = probs.reshape(-1, k)

    # Sample one token per row
    if generator is not None and probs_flat.device.type == 'cpu':
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1, generator=generator)
    else:
        # CUDA multinomial doesn't support generator in older PyTorch
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1)

    # Reshape back
    sampled_idx = sampled_idx_flat.reshape(*original_shape, 1)

    # Map back to original vocabulary indices
    # Gather from topk_indices using sampled positions
    result = topk_indices.gather(-1, sampled_idx)

    return result


def topk_sample_with_probs(
    logits: torch.Tensor,
    k: int,
    temperature: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from top-k tokens and return probabilities.

    Same as topk_sample but also returns the probability of the sampled token.

    Args:
        logits: Input logits tensor.
        k: Number of top tokens.
        temperature: Temperature scaling.
        generator: Random generator.

    Returns:
        Tuple of (sampled_tokens, token_probabilities).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if temperature <= 0:
        temperature = 1e-10

    vocab_size = logits.shape[-1]
    k = min(k, vocab_size)

    scaled_logits = apply_temperature(logits, temperature)
    topk_values, topk_indices = torch.topk(scaled_logits, k, dim=-1)
    probs = F.softmax(topk_values, dim=-1)

    original_shape = probs.shape[:-1]
    probs_flat = probs.reshape(-1, k)

    if generator is not None and probs_flat.device.type == 'cpu':
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1, generator=generator)
    else:
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1)

    sampled_idx = sampled_idx_flat.reshape(*original_shape, 1)

    # Get token indices
    tokens = topk_indices.gather(-1, sampled_idx)

    # Get probabilities of sampled tokens
    token_probs = probs.gather(-1, sampled_idx)

    return tokens, token_probs
