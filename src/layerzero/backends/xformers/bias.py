"""
LayerZero xFormers Attention Bias Handling

Functions for validating and expanding attention bias tensors
to meet xFormers requirements.

CRITICAL: xFormers has strict attn_bias requirements:
1. Must be on same device as query tensor
2. Cannot broadcast batch dimension (must match exactly)
3. Cannot broadcast head dimension (must match exactly)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.reasons import (
    ATTN_BIAS_BROADCAST_BATCH,
    ATTN_BIAS_BROADCAST_HEAD,
    ATTN_BIAS_DEVICE_MISMATCH,
    Reason,
    make_reason,
)

if TYPE_CHECKING:
    pass


def check_bias_device(
    bias: torch.Tensor,
    query: torch.Tensor,
) -> list[Reason]:
    """Check if attention bias is on same device as query.

    Args:
        bias: Attention bias tensor (batch, heads, seq_q, seq_k).
        query: Query tensor (batch, seq_q, heads, dim) in BSHD layout.

    Returns:
        Empty list if valid, else list with ATTN_BIAS_DEVICE_MISMATCH reason.
    """
    reasons: list[Reason] = []

    if bias.device != query.device:
        reasons.append(make_reason(
            ATTN_BIAS_DEVICE_MISMATCH,
            f"attn_bias on {bias.device} but query on {query.device}"
        ))

    return reasons


def check_bias_broadcast(
    bias: torch.Tensor,
    batch: int,
    heads: int,
) -> list[Reason]:
    """Check if attention bias requires broadcast (not allowed by xFormers).

    xFormers memory_efficient_attention does NOT support implicit
    broadcasting of batch or head dimensions in attn_bias.

    Args:
        bias: Attention bias tensor (batch, heads, seq_q, seq_k).
        batch: Expected batch size.
        heads: Expected number of heads.

    Returns:
        List of broadcast-related constraint failures.
    """
    reasons: list[Reason] = []

    # Check batch dimension (dim 0)
    if bias.shape[0] == 1 and batch > 1:
        reasons.append(make_reason(
            ATTN_BIAS_BROADCAST_BATCH,
            f"attn_bias batch dim is 1 but expected {batch}; "
            "xFormers does not support batch broadcast"
        ))

    # Check head dimension (dim 1)
    if bias.shape[1] == 1 and heads > 1:
        reasons.append(make_reason(
            ATTN_BIAS_BROADCAST_HEAD,
            f"attn_bias head dim is 1 but expected {heads}; "
            "xFormers does not support head broadcast"
        ))

    return reasons


def validate_attn_bias(
    bias: torch.Tensor | None,
    query: torch.Tensor,
    batch: int,
    heads: int,
) -> list[Reason]:
    """Validate attention bias for xFormers compatibility.

    Combines all validation checks for attn_bias. If bias is None,
    returns empty list (no validation needed).

    Args:
        bias: Attention bias tensor or None.
        query: Query tensor in BSHD layout.
        batch: Expected batch size.
        heads: Expected number of heads.

    Returns:
        List of all validation failures (empty if all valid or bias is None).
    """
    if bias is None:
        return []

    reasons: list[Reason] = []

    # Device check
    reasons.extend(check_bias_device(bias, query))

    # Broadcast check
    reasons.extend(check_bias_broadcast(bias, batch, heads))

    return reasons


def expand_attn_bias(
    bias: torch.Tensor,
    batch: int,
    heads: int,
) -> torch.Tensor:
    """Explicitly expand attention bias to avoid broadcast.

    xFormers does not support implicit broadcasting, so we must
    explicitly expand the bias tensor to match expected dimensions.

    This function uses torch.expand() which shares memory (no copy)
    when possible, only allocating new memory when necessary.

    Args:
        bias: Attention bias tensor (B, H, seq_q, seq_k).
               B and H can be 1 for broadcast dimensions.
        batch: Target batch size.
        heads: Target number of heads.

    Returns:
        Expanded attention bias tensor (batch, heads, seq_q, seq_k).
        If no expansion needed, returns the input tensor unchanged.
    """
    b, h, seq_q, seq_k = bias.shape

    # Check if expansion is needed
    needs_batch_expand = b == 1 and batch > 1
    needs_head_expand = h == 1 and heads > 1

    if not needs_batch_expand and not needs_head_expand:
        # No expansion needed, return as-is
        return bias

    # Build target shape
    target_shape = (
        batch if needs_batch_expand else b,
        heads if needs_head_expand else h,
        seq_q,
        seq_k,
    )

    # Use expand() for memory efficiency (shares underlying storage)
    # Then contiguous() to ensure proper memory layout for xFormers
    expanded = bias.expand(target_shape).contiguous()

    return expanded
