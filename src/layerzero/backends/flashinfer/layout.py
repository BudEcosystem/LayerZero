"""
LayerZero FlashInfer Layout Conversion

Functions for converting between tensor layouts for FlashInfer.

FlashInfer uses ragged layouts (no batch dimension):
- NHD: (num_tokens, num_heads, head_dim)
- HND: (num_heads, num_tokens, head_dim)

Standard layouts:
- BSHD: (batch, seq, heads, dim) - PyTorch SDPA
- BHSD: (batch, heads, seq, dim) - FlashAttention
"""
from __future__ import annotations

from typing import Any, TYPE_CHECKING

import torch

from layerzero.enums import Layout

if TYPE_CHECKING:
    pass


def bshd_to_nhd(
    tensor: torch.Tensor,
    seq_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert tensor from BSHD to NHD layout.

    BSHD: (batch, seq, heads, dim) - batched, padded
    NHD: (num_tokens, heads, dim) - ragged, no batch dimension

    Args:
        tensor: Input tensor in BSHD layout.
        seq_lens: Optional actual sequence lengths per batch item.
            If None, assumes full sequences (no padding).

    Returns:
        Tuple of (tensor in NHD layout, sequence lengths tensor).
    """
    batch, seq, heads, dim = tensor.shape
    device = tensor.device

    if seq_lens is None:
        # All sequences are full length
        seq_lens = torch.full((batch,), seq, device=device, dtype=torch.int32)

    # Check if all sequences are same length (common case)
    if seq_lens.min() == seq_lens.max() == seq:
        # Simple reshape: (B, S, H, D) -> (B*S, H, D)
        result = tensor.reshape(batch * seq, heads, dim)
        return result, seq_lens

    # Variable length sequences - need to extract valid tokens
    total_tokens = seq_lens.sum().item()
    result = torch.empty(total_tokens, heads, dim, device=device, dtype=tensor.dtype)

    offset = 0
    for b in range(batch):
        seq_len = seq_lens[b].item()
        result[offset:offset + seq_len] = tensor[b, :seq_len]
        offset += seq_len

    return result, seq_lens


def bhsd_to_nhd(
    tensor: torch.Tensor,
    seq_lens: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert tensor from BHSD to NHD layout.

    BHSD: (batch, heads, seq, dim) - FlashAttention layout
    NHD: (num_tokens, heads, dim) - ragged layout

    Args:
        tensor: Input tensor in BHSD layout.
        seq_lens: Optional actual sequence lengths per batch item.

    Returns:
        Tuple of (tensor in NHD layout, sequence lengths tensor).
    """
    batch, heads, seq, dim = tensor.shape
    device = tensor.device

    if seq_lens is None:
        seq_lens = torch.full((batch,), seq, device=device, dtype=torch.int32)

    # Transpose to BSHD first, then convert
    # BHSD -> BSHD: swap dims 1 and 2
    tensor_bshd = tensor.transpose(1, 2).contiguous()

    return bshd_to_nhd(tensor_bshd, seq_lens)


def nhd_to_bshd(
    tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Convert tensor from NHD to BSHD layout.

    NHD: (num_tokens, heads, dim) - ragged layout
    BSHD: (batch, seq, heads, dim) - batched, padded

    Args:
        tensor: Input tensor in NHD layout.
        seq_lens: Sequence lengths per batch item.
        max_seq_len: Maximum sequence length for output. If None, uses max(seq_lens).

    Returns:
        Tensor in BSHD layout (zero-padded to max_seq_len).
    """
    num_tokens, heads, dim = tensor.shape
    batch = seq_lens.shape[0]
    device = tensor.device

    if max_seq_len is None:
        max_seq_len = seq_lens.max().item()

    # Check if all sequences are same length
    if seq_lens.min() == seq_lens.max():
        seq = seq_lens[0].item()
        return tensor.reshape(batch, seq, heads, dim)

    # Variable length - need to pad
    result = torch.zeros(batch, max_seq_len, heads, dim, device=device, dtype=tensor.dtype)

    offset = 0
    for b in range(batch):
        seq_len = seq_lens[b].item()
        result[b, :seq_len] = tensor[offset:offset + seq_len]
        offset += seq_len

    return result


def nhd_to_bhsd(
    tensor: torch.Tensor,
    seq_lens: torch.Tensor,
    max_seq_len: int | None = None,
) -> torch.Tensor:
    """Convert tensor from NHD to BHSD layout.

    NHD: (num_tokens, heads, dim) - ragged layout
    BHSD: (batch, heads, seq, dim) - FlashAttention layout

    Args:
        tensor: Input tensor in NHD layout.
        seq_lens: Sequence lengths per batch item.
        max_seq_len: Maximum sequence length for output.

    Returns:
        Tensor in BHSD layout (zero-padded to max_seq_len).
    """
    # Convert to BSHD first, then transpose
    bshd = nhd_to_bshd(tensor, seq_lens, max_seq_len)
    # BSHD -> BHSD: swap dims 1 and 2
    return bshd.transpose(1, 2).contiguous()


def hnd_to_nhd(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from HND to NHD layout.

    HND: (heads, num_tokens, dim)
    NHD: (num_tokens, heads, dim)

    Args:
        tensor: Input tensor in HND layout.

    Returns:
        Tensor in NHD layout.
    """
    # Transpose dims 0 and 1
    return tensor.transpose(0, 1).contiguous()


def nhd_to_hnd(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor from NHD to HND layout.

    NHD: (num_tokens, heads, dim)
    HND: (heads, num_tokens, dim)

    Args:
        tensor: Input tensor in NHD layout.

    Returns:
        Tensor in HND layout.
    """
    # Transpose dims 0 and 1
    return tensor.transpose(0, 1).contiguous()


def convert_layout_for_flashinfer(
    tensor: torch.Tensor,
    from_layout: Layout,
    to_layout: Layout,
    seq_lens: torch.Tensor | None = None,
    max_seq_len: int | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Convert tensor between layouts for FlashInfer.

    Handles conversions between batched (BSHD, BHSD) and
    ragged (NHD, HND) layouts.

    Args:
        tensor: Input tensor.
        from_layout: Current layout.
        to_layout: Target layout.
        seq_lens: Sequence lengths (required for NHD -> batched conversions).
        max_seq_len: Maximum sequence length for output (batched layouts).

    Returns:
        Tuple of (converted tensor, metadata dict with seq_lens etc.)

    Raises:
        ValueError: If conversion is not supported.
    """
    metadata: dict[str, Any] = {}

    # No-op if same layout
    if from_layout == to_layout:
        return tensor, metadata

    # BSHD -> NHD
    if from_layout == Layout.BSHD and to_layout == Layout.NHD:
        result, output_seq_lens = bshd_to_nhd(tensor, seq_lens)
        metadata["seq_lens"] = output_seq_lens
        return result, metadata

    # BHSD -> NHD
    if from_layout == Layout.BHSD and to_layout == Layout.NHD:
        result, output_seq_lens = bhsd_to_nhd(tensor, seq_lens)
        metadata["seq_lens"] = output_seq_lens
        return result, metadata

    # NHD -> BSHD
    if from_layout == Layout.NHD and to_layout == Layout.BSHD:
        if seq_lens is None:
            raise ValueError("seq_lens required for NHD to BSHD conversion")
        result = nhd_to_bshd(tensor, seq_lens, max_seq_len)
        return result, metadata

    # NHD -> BHSD
    if from_layout == Layout.NHD and to_layout == Layout.BHSD:
        if seq_lens is None:
            raise ValueError("seq_lens required for NHD to BHSD conversion")
        result = nhd_to_bhsd(tensor, seq_lens, max_seq_len)
        return result, metadata

    # HND <-> NHD
    if from_layout == Layout.HND and to_layout == Layout.NHD:
        return hnd_to_nhd(tensor), metadata

    if from_layout == Layout.NHD and to_layout == Layout.HND:
        return nhd_to_hnd(tensor), metadata

    # Unsupported conversions
    raise ValueError(
        f"Layout conversion from {from_layout.value} to {to_layout.value} "
        "is not supported. seq_lens may be required for ragged->batched conversions."
    )
