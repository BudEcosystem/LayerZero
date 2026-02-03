"""
Meta Kernels for Tracing and Export

Provides meta kernels that compute output shapes without
executing actual computation. Used for:
- torch.export tracing
- torch.compile shape inference
- FX graph construction
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    pass


def attention_meta(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Meta kernel for attention op.

    Computes output shape without executing attention.

    Args:
        query: Query tensor [B, H, L, D]
        key: Key tensor [B, H, S, D]
        value: Value tensor [B, H, S, D]
        attn_mask: Optional attention mask
        dropout_p: Dropout probability
        is_causal: Whether to use causal masking
        scale: Optional scale factor

    Returns:
        Empty tensor with correct output shape and dtype.
    """
    # Output shape matches query shape: [B, H, L, D]
    return query.new_empty(query.shape)


def rms_norm_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Meta kernel for RMS normalization.

    Args:
        input: Input tensor
        weight: Weight tensor
        eps: Small constant for numerical stability

    Returns:
        Empty tensor with same shape as input.
    """
    return input.new_empty(input.shape)


def layer_norm_meta(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Meta kernel for layer normalization.

    Args:
        input: Input tensor
        weight: Weight tensor
        bias: Optional bias tensor
        eps: Small constant for numerical stability

    Returns:
        Empty tensor with same shape as input.
    """
    return input.new_empty(input.shape)


def rope_meta(
    input: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Meta kernel for RoPE (Rotary Position Embedding).

    Args:
        input: Input tensor
        cos: Cosine rotation tensor
        sin: Sine rotation tensor

    Returns:
        Empty tensor with same shape as input.
    """
    return input.new_empty(input.shape)


def swiglu_meta(
    gate: torch.Tensor,
    up: torch.Tensor,
) -> torch.Tensor:
    """Meta kernel for SwiGLU activation.

    Args:
        gate: Gate tensor
        up: Up projection tensor

    Returns:
        Empty tensor with same shape as gate.
    """
    return gate.new_empty(gate.shape)


def cross_entropy_meta(
    input: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    """Meta kernel for cross-entropy loss.

    Args:
        input: Input logits
        target: Target labels
        weight: Optional class weights
        ignore_index: Index to ignore
        label_smoothing: Label smoothing factor

    Returns:
        Scalar tensor (loss).
    """
    return input.new_empty(())
