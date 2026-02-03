"""
Combined Top-K + Top-P Sampling

Applies both top-k and top-p filtering for more controlled sampling.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Optional

from layerzero.sampling.temperature import apply_temperature


def topk_topp_sample(
    logits: torch.Tensor,
    k: int,
    p: float,
    temperature: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
    min_tokens_to_keep: int = 1,
) -> torch.Tensor:
    """Combined top-k and top-p (nucleus) sampling.

    First applies top-k filtering, then applies top-p within the top-k set.
    This provides fine-grained control over the sampling distribution.

    Args:
        logits: Input logits tensor of shape (batch_size, vocab_size).
        k: Number of top tokens for top-k filtering.
        p: Cumulative probability threshold for top-p filtering.
        temperature: Temperature for scaling logits.
        generator: Optional random generator.
        min_tokens_to_keep: Minimum tokens to keep.

    Returns:
        Sampled token indices tensor.

    Note:
        The effective vocabulary is intersection of top-k and top-p sets.
        - k=50, p=0.9: Sample from nucleus within top-50 tokens
        - If top-k contains less than nucleus, top-k dominates
        - If nucleus is smaller than k, top-p dominates
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if not 0.0 < p <= 1.0:
        p = max(min(p, 1.0), 1e-10)

    if temperature <= 0:
        temperature = 1e-10

    vocab_size = logits.shape[-1]
    k = min(k, vocab_size)

    # Apply temperature
    scaled_logits = apply_temperature(logits, temperature)

    # Step 1: Top-K filtering
    # Get top-k values and indices
    topk_values, topk_indices = torch.topk(scaled_logits, k, dim=-1)

    # Step 2: Top-P filtering within top-k
    # Sort the top-k values
    sorted_values, sorted_idx_in_topk = torch.sort(topk_values, descending=True, dim=-1)

    # Compute cumulative probabilities within top-k
    sorted_probs = F.softmax(sorted_values, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Create mask for tokens to remove
    sorted_indices_to_remove = cumulative_probs > p

    # Keep at least min_tokens_to_keep
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False

    # Shift mask to include the token that crosses threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    # Apply mask
    sorted_values = sorted_values.masked_fill(sorted_indices_to_remove, float('-inf'))

    # Convert to probabilities
    probs = F.softmax(sorted_values, dim=-1)

    # Sample
    original_shape = probs.shape[:-1]
    probs_flat = probs.reshape(-1, k)

    if generator is not None and probs_flat.device.type == 'cpu':
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1, generator=generator)
    else:
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1)

    sampled_idx = sampled_idx_flat.reshape(*original_shape, 1)

    # Map back: sampled_idx -> sorted_idx_in_topk -> topk_indices -> vocab
    idx_in_topk = sorted_idx_in_topk.gather(-1, sampled_idx)
    result = topk_indices.gather(-1, idx_in_topk)

    return result


def topk_topp_sample_with_probs(
    logits: torch.Tensor,
    k: int,
    p: float,
    temperature: float = 1.0,
    *,
    generator: Optional[torch.Generator] = None,
    min_tokens_to_keep: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Combined sampling with probability output.

    Same as topk_topp_sample but returns token probabilities.

    Args:
        logits: Input logits tensor.
        k: Top-k parameter.
        p: Top-p parameter.
        temperature: Temperature scaling.
        generator: Random generator.
        min_tokens_to_keep: Minimum tokens.

    Returns:
        Tuple of (sampled_tokens, token_probabilities).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")

    if not 0.0 < p <= 1.0:
        p = max(min(p, 1.0), 1e-10)

    if temperature <= 0:
        temperature = 1e-10

    vocab_size = logits.shape[-1]
    k = min(k, vocab_size)

    scaled_logits = apply_temperature(logits, temperature)
    topk_values, topk_indices = torch.topk(scaled_logits, k, dim=-1)
    sorted_values, sorted_idx_in_topk = torch.sort(topk_values, descending=True, dim=-1)
    sorted_probs = F.softmax(sorted_values, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., :min_tokens_to_keep] = False
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    sorted_values = sorted_values.masked_fill(sorted_indices_to_remove, float('-inf'))
    probs = F.softmax(sorted_values, dim=-1)

    original_shape = probs.shape[:-1]
    probs_flat = probs.reshape(-1, k)

    if generator is not None and probs_flat.device.type == 'cpu':
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1, generator=generator)
    else:
        sampled_idx_flat = torch.multinomial(probs_flat, num_samples=1)

    sampled_idx = sampled_idx_flat.reshape(*original_shape, 1)

    idx_in_topk = sorted_idx_in_topk.gather(-1, sampled_idx)
    tokens = topk_indices.gather(-1, idx_in_topk)
    token_probs = probs.gather(-1, sampled_idx)

    return tokens, token_probs
