"""
Top-P (Nucleus) Sampling Implementation

Selects tokens from the smallest set whose cumulative probability >= p.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

from layerzero.sampling.temperature import apply_temperature


def topp_sample(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Sample from nucleus (top-p) distribution.

    Filters to the smallest set of tokens whose cumulative probability
    is >= p, then samples from that set.

    Args:
        logits: Input logits tensor of shape (batch_size, vocab_size) or
                (batch_size, seq_len, vocab_size).
        p: Cumulative probability threshold (0.0, 1.0].
           Lower values are more deterministic.
        temperature: Temperature for scaling logits before softmax.
        generator: Optional PyTorch random generator for reproducibility.
        min_tokens_to_keep: Minimum number of tokens to keep regardless of p.

    Returns:
        Sampled token indices tensor of shape (batch_size, 1) or
        (batch_size, seq_len, 1).

    Note:
        - If p is very small, this approaches greedy decoding.
        - If p=1.0, this samples from the full distribution.
        - The algorithm keeps all tokens until cumulative prob exceeds p.
    """
    if not 0.0 < p <= 1.0:
        # Clamp p to valid range
        p = max(min(p, 1.0), 1e-10)

    if temperature <= 0:
        temperature = 1e-10

    # Apply temperature scaling
    scaled_logits = apply_temperature(logits, temperature)

    # Sort logits in descending order
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)

    # Compute cumulative probabilities
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask for tokens to remove (cumulative prob > p)
    # We shift by 1 to include the token that crosses the threshold
    sorted_indices_to_remove = cumulative_probs > p

    # Keep at least min_tokens_to_keep
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    # Shift mask right to keep token that crosses threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Mask out removed tokens with -inf
    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))

    # Convert to probabilities (masked tokens become 0)
    probs = F.softmax(sorted_logits, dim=-1)

    # Sample from filtered distribution
    original_shape = probs.shape[:-1]
    vocab_size = probs.shape[-1]
    probs_flat = probs.reshape(-1, vocab_size)

    if generator is not None and probs_flat.device.type == 'cpu':
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1, generator=generator)
    else:
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1)

    sampled_idx = sampled_idx_flat.reshape(*original_shape, 1)

    # Map back to original vocabulary indices
    result = sorted_indices.gather(-1, sampled_idx)

    return result


def topp_sample_with_probs(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
    min_tokens_to_keep: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from nucleus distribution and return probabilities.

    Same as topp_sample but also returns the probability of the sampled token.

    Args:
        logits: Input logits tensor.
        p: Cumulative probability threshold.
        temperature: Temperature scaling.
        generator: Random generator.
        min_tokens_to_keep: Minimum tokens to keep.

    Returns:
        Tuple of (sampled_tokens, token_probabilities).
    """
    if not 0.0 < p <= 1.0:
        p = max(min(p, 1.0), 1e-10)

    if temperature <= 0:
        temperature = 1e-10

    scaled_logits = apply_temperature(logits, temperature)
    sorted_logits, sorted_indices = torch.sort(scaled_logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    if min_tokens_to_keep > 1:
        sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    sorted_logits = sorted_logits.masked_fill(sorted_indices_to_remove, float('-inf'))
    probs = F.softmax(sorted_logits, dim=-1)

    original_shape = probs.shape[:-1]
    vocab_size = probs.shape[-1]
    probs_flat = probs.reshape(-1, vocab_size)

    if generator is not None and probs_flat.device.type == 'cpu':
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1, generator=generator)
    else:
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1)

    sampled_idx = sampled_idx_flat.reshape(*original_shape, 1)

    # Get token indices
    tokens = sorted_indices.gather(-1, sampled_idx)

    # Get probabilities of sampled tokens
    token_probs = probs.gather(-1, sampled_idx)

    return tokens, token_probs


def compute_nucleus_size(
    logits: torch.Tensor,
    p: float,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute the nucleus size for given p threshold.

    Returns the number of tokens in the nucleus (top-p set) for each
    position in the input.

    Args:
        logits: Input logits tensor.
        p: Cumulative probability threshold.
        temperature: Temperature scaling.

    Returns:
        Tensor of nucleus sizes with shape matching input batch dims.
    """
    if not 0.0 < p <= 1.0:
        p = max(min(p, 1.0), 1e-10)

    if temperature <= 0:
        temperature = 1e-10

    scaled_logits = apply_temperature(logits, temperature)
    sorted_logits, _ = torch.sort(scaled_logits, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_logits, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Count tokens where cumulative prob <= p (plus 1 for the crossing token)
    nucleus_sizes = (cumulative_probs <= p).sum(dim=-1) + 1

    # Clamp to vocab size
    vocab_size = logits.shape[-1]
    nucleus_sizes = nucleus_sizes.clamp(max=vocab_size)

    return nucleus_sizes
