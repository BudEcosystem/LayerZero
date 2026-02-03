"""
Tensor fixtures for LayerZero tests.

Provides reusable tensor creation utilities.
"""
from __future__ import annotations

import torch


def create_sample_tensors(
    batch_size: int = 2,
    num_heads: int = 4,
    seq_len: int = 16,
    head_dim: int = 32,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Create sample Q, K, V tensors for attention tests.

    Args:
        batch_size: Batch size.
        num_heads: Number of attention heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        dtype: Tensor dtype.
        device: Target device.

    Returns:
        Dictionary with 'query', 'key', 'value' tensors.
    """
    shape = (batch_size, num_heads, seq_len, head_dim)

    return {
        "query": torch.randn(shape, dtype=dtype, device=device),
        "key": torch.randn(shape, dtype=dtype, device=device),
        "value": torch.randn(shape, dtype=dtype, device=device),
    }


def create_attention_mask(
    batch_size: int = 2,
    seq_len: int = 16,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    mask_ratio: float = 0.0,
) -> torch.Tensor:
    """Create attention mask tensor.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        dtype: Tensor dtype.
        device: Target device.
        mask_ratio: Ratio of positions to mask (0.0 = no masking).

    Returns:
        Attention mask tensor [batch_size, seq_len].
    """
    if mask_ratio == 0.0:
        # No masking - all ones
        return torch.ones(batch_size, seq_len, dtype=dtype, device=device)

    # Random masking
    mask = torch.rand(batch_size, seq_len, device=device) > mask_ratio
    return mask.to(dtype)


def create_causal_mask(
    seq_len: int = 16,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create causal (lower triangular) attention mask.

    Args:
        seq_len: Sequence length.
        dtype: Tensor dtype.
        device: Target device.

    Returns:
        Causal mask tensor [seq_len, seq_len].
    """
    # Create lower triangular mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))

    # Convert to attention mask format (0 for attend, -inf for mask)
    if dtype == torch.bool:
        return mask.bool()

    return mask.masked_fill(mask == 0, float("-inf")).to(dtype)


def create_kv_cache(
    batch_size: int = 2,
    num_heads: int = 4,
    max_len: int = 128,
    head_dim: int = 32,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create KV cache tensors.

    Args:
        batch_size: Batch size.
        num_heads: Number of attention heads.
        max_len: Maximum sequence length.
        head_dim: Head dimension.
        dtype: Tensor dtype.
        device: Target device.

    Returns:
        Tuple of (key_cache, value_cache) tensors.
    """
    shape = (batch_size, num_heads, max_len, head_dim)

    key_cache = torch.zeros(shape, dtype=dtype, device=device)
    value_cache = torch.zeros(shape, dtype=dtype, device=device)

    return key_cache, value_cache


def create_position_ids(
    batch_size: int = 2,
    seq_len: int = 16,
    device: str | torch.device = "cpu",
    start_pos: int = 0,
) -> torch.Tensor:
    """Create position ID tensor.

    Args:
        batch_size: Batch size.
        seq_len: Sequence length.
        device: Target device.
        start_pos: Starting position.

    Returns:
        Position IDs tensor [batch_size, seq_len].
    """
    positions = torch.arange(start_pos, start_pos + seq_len, device=device)
    return positions.unsqueeze(0).expand(batch_size, -1)


def create_gqa_tensors(
    batch_size: int = 2,
    num_q_heads: int = 8,
    num_kv_heads: int = 2,
    seq_len: int = 16,
    head_dim: int = 32,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Create tensors for GQA (Grouped Query Attention) tests.

    Args:
        batch_size: Batch size.
        num_q_heads: Number of query heads.
        num_kv_heads: Number of key/value heads.
        seq_len: Sequence length.
        head_dim: Head dimension.
        dtype: Tensor dtype.
        device: Target device.

    Returns:
        Dictionary with 'query', 'key', 'value' tensors.
    """
    return {
        "query": torch.randn(
            batch_size, num_q_heads, seq_len, head_dim,
            dtype=dtype, device=device
        ),
        "key": torch.randn(
            batch_size, num_kv_heads, seq_len, head_dim,
            dtype=dtype, device=device
        ),
        "value": torch.randn(
            batch_size, num_kv_heads, seq_len, head_dim,
            dtype=dtype, device=device
        ),
    }


def create_sliding_window_mask(
    seq_len: int = 16,
    window_size: int = 4,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Create sliding window attention mask.

    Args:
        seq_len: Sequence length.
        window_size: Sliding window size.
        dtype: Tensor dtype.
        device: Target device.

    Returns:
        Sliding window mask tensor [seq_len, seq_len].
    """
    # Create position indices
    row_idx = torch.arange(seq_len, device=device).unsqueeze(1)
    col_idx = torch.arange(seq_len, device=device).unsqueeze(0)

    # Window: attend to positions within window_size distance
    # Also apply causal constraint (can't attend to future)
    in_window = (row_idx - col_idx).abs() <= window_size
    causal = row_idx >= col_idx

    mask = in_window & causal

    if dtype == torch.bool:
        return mask

    # Convert to attention mask format
    return mask.float().masked_fill(~mask, float("-inf")).to(dtype)
