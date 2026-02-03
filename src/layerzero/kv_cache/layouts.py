"""KV cache layout definitions and conversion utilities.

This module provides:
- KVCacheLayout enum (NHD, HND, BNHD)
- Layout conversion functions
- Layout validation utilities
"""
from __future__ import annotations

import logging
from enum import Enum, unique
from typing import Any

import torch

logger = logging.getLogger(__name__)


@unique
class KVCacheLayout(str, Enum):
    """KV cache tensor layouts.

    NHD: (batch, seq_len, num_heads, head_dim) - HuggingFace default
    HND: (batch, num_heads, seq_len, head_dim) - FlashAttention default
    BNHD: (num_blocks, block_size, num_heads, head_dim) - Paged attention
    """

    NHD = "nhd"
    HND = "hnd"
    BNHD = "bnhd"


# Layout dimension orderings
LAYOUT_DIM_ORDER: dict[KVCacheLayout, tuple[str, ...]] = {
    KVCacheLayout.NHD: ("batch", "seq", "heads", "dim"),
    KVCacheLayout.HND: ("batch", "heads", "seq", "dim"),
    KVCacheLayout.BNHD: ("blocks", "blocksize", "heads", "dim"),
}


def get_layout_dim_order(layout: KVCacheLayout) -> tuple[str, ...]:
    """Get dimension order for a layout.

    Args:
        layout: KV cache layout.

    Returns:
        Tuple of dimension names.
    """
    return LAYOUT_DIM_ORDER[layout]


def get_seq_dim(layout: KVCacheLayout) -> int:
    """Get sequence dimension index for a layout.

    Args:
        layout: KV cache layout.

    Returns:
        Sequence dimension index.
    """
    dim_order = LAYOUT_DIM_ORDER[layout]
    return dim_order.index("seq") if "seq" in dim_order else -1


def get_head_dim(layout: KVCacheLayout) -> int:
    """Get head dimension index for a layout.

    Args:
        layout: KV cache layout.

    Returns:
        Head dimension index.
    """
    dim_order = LAYOUT_DIM_ORDER[layout]
    return dim_order.index("heads")


def convert_layout(
    tensor: torch.Tensor,
    from_layout: KVCacheLayout,
    to_layout: KVCacheLayout,
) -> torch.Tensor:
    """Convert tensor between layouts.

    Args:
        tensor: Input tensor.
        from_layout: Source layout.
        to_layout: Target layout.

    Returns:
        Tensor in target layout.
    """
    if from_layout == to_layout:
        # Return view (no copy)
        return tensor

    # NHD <-> HND conversion
    if from_layout == KVCacheLayout.NHD and to_layout == KVCacheLayout.HND:
        # (batch, seq, heads, dim) -> (batch, heads, seq, dim)
        return tensor.transpose(1, 2)

    if from_layout == KVCacheLayout.HND and to_layout == KVCacheLayout.NHD:
        # (batch, heads, seq, dim) -> (batch, seq, heads, dim)
        return tensor.transpose(1, 2)

    # Other conversions would go here
    raise NotImplementedError(
        f"Layout conversion from {from_layout} to {to_layout} not implemented"
    )


def validate_layout_shape(
    tensor: torch.Tensor,
    layout: KVCacheLayout,
    num_heads: int,
    head_dim: int,
) -> bool:
    """Validate tensor shape matches expected layout.

    Args:
        tensor: Tensor to validate.
        layout: Expected layout.
        num_heads: Expected number of heads.
        head_dim: Expected head dimension.

    Returns:
        True if shape is valid.
    """
    if tensor.ndim != 4:
        return False

    if layout == KVCacheLayout.NHD:
        # (batch, seq, heads, dim)
        _, _, h, d = tensor.shape
        return h == num_heads and d == head_dim

    elif layout == KVCacheLayout.HND:
        # (batch, heads, seq, dim)
        _, h, _, d = tensor.shape
        return h == num_heads and d == head_dim

    elif layout == KVCacheLayout.BNHD:
        # (blocks, blocksize, heads, dim)
        _, _, h, d = tensor.shape
        return h == num_heads and d == head_dim

    return False


def layout_from_string(s: str) -> KVCacheLayout:
    """Parse layout from string.

    Args:
        s: Layout string (case-insensitive).

    Returns:
        KVCacheLayout enum value.

    Raises:
        ValueError: If layout string is unknown.
    """
    s_lower = s.lower()
    for layout in KVCacheLayout:
        if layout.value == s_lower:
            return layout

    valid = [l.value for l in KVCacheLayout]
    raise ValueError(f"Unknown layout '{s}'. Valid layouts: {valid}")
