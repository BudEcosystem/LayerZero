"""
LayerZero FlashAttention Layout Conversion

Functions for converting between tensor layouts.

FlashAttention uses BSHD (Batch, Seq, Heads, Dim) layout.
PyTorch SDPA uses BHSD (Batch, Heads, Seq, Dim) layout.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.enums import Layout

if TYPE_CHECKING:
    pass


def bhsd_to_bshd(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from BHSD to BSHD layout.

    BHSD: (Batch, Heads, Seq, Dim) - PyTorch standard
    BSHD: (Batch, Seq, Heads, Dim) - FlashAttention format

    Args:
        tensor: Input tensor in BHSD layout (batch, heads, seq, dim).

    Returns:
        Tensor in BSHD layout (batch, seq, heads, dim).
    """
    # Transpose dims 1 and 2
    return tensor.transpose(1, 2)


def bshd_to_bhsd(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from BSHD to BHSD layout.

    BSHD: (Batch, Seq, Heads, Dim) - FlashAttention format
    BHSD: (Batch, Heads, Seq, Dim) - PyTorch standard

    Args:
        tensor: Input tensor in BSHD layout (batch, seq, heads, dim).

    Returns:
        Tensor in BHSD layout (batch, heads, seq, dim).
    """
    # Transpose dims 1 and 2
    return tensor.transpose(1, 2)


def convert_layout(
    tensor: torch.Tensor,
    from_layout: Layout,
    to_layout: Layout,
) -> torch.Tensor:
    """Convert tensor between layouts.

    Supports:
    - BHSD <-> BSHD

    Args:
        tensor: Input tensor.
        from_layout: Current layout of tensor.
        to_layout: Desired output layout.

    Returns:
        Tensor in target layout.

    Raises:
        ValueError: If conversion is not supported.
    """
    # No conversion needed
    if from_layout == to_layout:
        return tensor

    # BHSD -> BSHD
    if from_layout == Layout.BHSD and to_layout == Layout.BSHD:
        return bhsd_to_bshd(tensor)

    # BSHD -> BHSD
    if from_layout == Layout.BSHD and to_layout == Layout.BHSD:
        return bshd_to_bhsd(tensor)

    raise ValueError(
        f"Layout conversion from {from_layout.value} to {to_layout.value} "
        "is not supported."
    )
