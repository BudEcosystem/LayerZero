"""ALiBi (Attention with Linear Biases) Positional Encoding.

Implementation of ALiBi per "Train Short, Test Long: Attention with Linear Biases
Enables Input Length Extrapolation" (Press et al., ICLR 2022).

ALiBi adds a linear bias to attention scores based on the distance between
query and key positions, allowing models to generalize to longer sequences
than seen during training.

Key properties:
- No learned parameters (bias is computed from geometric slopes)
- Linear relationship between distance and bias
- Different slopes per attention head for multi-resolution
- Excellent length extrapolation capability

Reference: https://arxiv.org/abs/2108.12409
"""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Optional, Dict, Tuple, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass

# Module-level cache for ALiBi biases
_alibi_bias_cache: Dict[Tuple[int, int, str, str], torch.Tensor] = {}


def get_alibi_slopes(
    num_heads: int,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Generate ALiBi slopes for each attention head.

    For a power-of-2 number of heads n, slopes are:
        2^(-8/n), 2^(-16/n), ..., 2^(-8)

    For non-power-of-2 heads, uses interleaving pattern from original paper.

    Args:
        num_heads: Number of attention heads.
        device: Device to place slopes tensor on.
        dtype: Data type for slopes (default: float32).

    Returns:
        Tensor of shape (num_heads,) with ALiBi slopes.

    Example:
        >>> slopes = get_alibi_slopes(8)
        >>> slopes.shape
        torch.Size([8])
        >>> slopes[0]  # First head has steepest slope
        tensor(0.5000)
    """
    if dtype is None:
        dtype = torch.float32

    def _get_slopes_power_of_2(n: int) -> list[float]:
        """Generate slopes for power-of-2 head counts."""
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio ** i) for i in range(n)]

    # Check if num_heads is power of 2
    if num_heads > 0 and (num_heads & (num_heads - 1)) == 0:
        # Power of 2 - use direct formula
        slopes = _get_slopes_power_of_2(num_heads)
    else:
        # Non-power of 2 - interleave slopes from nearest power of 2
        closest_power_of_2 = 2 ** math.ceil(math.log2(num_heads))
        slopes_full = _get_slopes_power_of_2(closest_power_of_2)

        # Interleave to get num_heads slopes
        # Take every other slope from the full list, twice if needed
        slopes = []
        for i in range(num_heads):
            if i < closest_power_of_2 // 2:
                # First half: even indices from full slopes
                slopes.append(slopes_full[i * 2])
            else:
                # Second half: odd indices from full slopes
                idx = (i - closest_power_of_2 // 2) * 2 + 1
                if idx < len(slopes_full):
                    slopes.append(slopes_full[idx])
                else:
                    # Fallback for edge cases
                    slopes.append(slopes_full[i % len(slopes_full)])

    slopes_tensor = torch.tensor(slopes, dtype=dtype)

    if device is not None:
        slopes_tensor = slopes_tensor.to(device)

    return slopes_tensor


def get_alibi_bias(
    num_heads: int,
    seq_len: int,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Generate ALiBi attention bias matrix.

    Creates a bias tensor where bias[h, i, j] = -slopes[h] * |i - j|
    This penalizes attention between distant positions, with different
    penalty rates per head.

    Args:
        num_heads: Number of attention heads.
        seq_len: Sequence length for query and key.
        device: Device to place bias tensor on.
        dtype: Data type for bias (default: float32).
        use_cache: Whether to cache and reuse bias tensors.

    Returns:
        Tensor of shape (1, num_heads, seq_len, seq_len) with ALiBi biases.
        The leading dimension of 1 allows broadcasting over batch.

    Example:
        >>> bias = get_alibi_bias(8, 64)
        >>> bias.shape
        torch.Size([1, 8, 64, 64])
        >>> bias[0, 0, 10, 5]  # Bias for head 0, query at 10, key at 5
        tensor(-2.5000)  # = -0.5 * 5 (distance is 5)
    """
    if dtype is None:
        dtype = torch.float32

    device_str = str(device) if device is not None else "cpu"
    dtype_str = str(dtype)

    # Check cache
    cache_key = (num_heads, seq_len, device_str, dtype_str)
    if use_cache and cache_key in _alibi_bias_cache:
        cached = _alibi_bias_cache[cache_key]
        # Verify cached tensor is still valid
        if cached.device.type == device_str.split(":")[0]:
            return cached

    # Get slopes for each head
    slopes = get_alibi_slopes(num_heads, device=device, dtype=dtype)

    # Create position indices
    # positions: (seq_len,) from 0 to seq_len-1
    positions = torch.arange(seq_len, device=slopes.device, dtype=dtype)

    # Create distance matrix: distance[i, j] = i - j
    # Negative distances mean key is after query (future)
    # Positive distances mean key is before query (past)
    # For bias, we use |i - j| multiplied by negative slope
    query_positions = positions.unsqueeze(1)  # (seq_len, 1)
    key_positions = positions.unsqueeze(0)    # (1, seq_len)
    distances = query_positions - key_positions  # (seq_len, seq_len)

    # ALiBi bias: -slope * distance (use negative to penalize distant tokens)
    # For causal attention (i >= j), distances are non-negative
    # For full attention, we use absolute distance
    # Original ALiBi uses signed distance for causal (only past matters)
    # Here we provide the raw distance-based bias

    # Reshape slopes for broadcasting: (num_heads, 1, 1)
    slopes_reshaped = slopes.view(num_heads, 1, 1)

    # Compute bias: (num_heads, seq_len, seq_len)
    # Negative slope * positive distance = negative bias (penalty)
    bias = -slopes_reshaped * distances.abs().unsqueeze(0)

    # Add batch dimension for broadcasting: (1, num_heads, seq_len, seq_len)
    bias = bias.unsqueeze(0)

    # Cache if enabled
    if use_cache:
        _alibi_bias_cache[cache_key] = bias

    return bias


def get_alibi_bias_causal(
    num_heads: int,
    seq_len: int,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    use_cache: bool = True,
) -> torch.Tensor:
    """Generate ALiBi bias with causal masking.

    Combines ALiBi positional bias with causal mask that prevents
    attending to future positions.

    Args:
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        device: Device to place tensor on.
        dtype: Data type (default: float32).
        use_cache: Whether to cache bias tensors.

    Returns:
        Tensor of shape (1, num_heads, seq_len, seq_len) where:
        - Lower triangle: ALiBi bias values
        - Upper triangle: -inf (masked)

    Example:
        >>> bias = get_alibi_bias_causal(8, 64)
        >>> bias[0, 0, 5, 10]  # Query 5, key 10 (future)
        tensor(-inf)
        >>> bias[0, 0, 10, 5]  # Query 10, key 5 (past)
        tensor(-2.5000)  # ALiBi bias
    """
    # Get base ALiBi bias
    bias = get_alibi_bias(
        num_heads, seq_len, device=device, dtype=dtype, use_cache=False
    )

    # Create causal mask
    # mask[i, j] = True if j > i (future position, should be masked)
    causal_mask = torch.triu(
        torch.ones(seq_len, seq_len, device=bias.device, dtype=torch.bool),
        diagonal=1
    )

    # Apply mask: set future positions to -inf
    bias = bias.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

    return bias


def build_alibi_tensor(
    num_heads: int,
    max_seq_len: int,
    device: Optional[str | torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Build ALiBi tensor for use with scaled_dot_product_attention.

    This is a convenience function that builds the ALiBi bias in the format
    expected by torch.nn.functional.scaled_dot_product_attention.

    Args:
        num_heads: Number of attention heads.
        max_seq_len: Maximum sequence length to support.
        device: Device to place tensor on.
        dtype: Data type for tensor.

    Returns:
        ALiBi bias tensor ready for use with SDPA.

    Example:
        >>> alibi = build_alibi_tensor(8, 2048)
        >>> # Use with scaled_dot_product_attention
        >>> output = F.scaled_dot_product_attention(
        ...     query, key, value, attn_mask=alibi
        ... )
    """
    return get_alibi_bias_causal(
        num_heads, max_seq_len, device=device, dtype=dtype, use_cache=True
    )


def clear_alibi_cache() -> None:
    """Clear the ALiBi bias cache.

    Useful for freeing memory when ALiBi biases are no longer needed,
    or when switching between different model configurations.
    """
    global _alibi_bias_cache
    _alibi_bias_cache.clear()


# Pre-compute common configurations for fast startup
def _warmup_common_configs() -> None:
    """Pre-compute ALiBi biases for common configurations.

    Called lazily on first use to warm up the cache with typical
    LLM configurations.
    """
    common_configs = [
        (8, 512),
        (8, 1024),
        (8, 2048),
        (32, 512),
        (32, 1024),
        (32, 2048),
    ]
    for num_heads, seq_len in common_configs:
        try:
            get_alibi_bias(num_heads, seq_len, use_cache=True)
        except Exception:
            pass  # Ignore warmup failures
