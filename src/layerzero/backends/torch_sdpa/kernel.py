"""
LayerZero SDPA Kernel Wrapper

Low-level wrapper for torch.nn.functional.scaled_dot_product_attention.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    pass


@dataclass(frozen=True, slots=True)
class SDPAConfig:
    """Configuration for SDPA forward pass.

    Attributes:
        is_causal: Use causal attention mask.
        dropout_p: Dropout probability.
        scale: Attention scale (None = 1/sqrt(head_dim)).
        enable_gqa: Enable grouped query attention.
    """

    is_causal: bool = False
    dropout_p: float = 0.0
    scale: float | None = None
    enable_gqa: bool = False


def sdpa_forward(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    dropout_p: float = 0.0,
    scale: float | None = None,
    enable_gqa: bool = False,
    training: bool = False,
    backend_hint: str | None = None,
    config: SDPAConfig | None = None,
) -> torch.Tensor:
    """Execute scaled dot product attention.

    Wraps torch.nn.functional.scaled_dot_product_attention with
    additional validation and backend selection.

    Args:
        query: Query tensor (batch, heads, seq_q, head_dim).
        key: Key tensor (batch, heads_kv, seq_k, head_dim).
        value: Value tensor (batch, heads_kv, seq_k, head_dim).
        attn_mask: Optional attention mask.
        is_causal: Use causal attention masking.
        dropout_p: Dropout probability.
        scale: Attention scale (default: 1/sqrt(head_dim)).
        enable_gqa: Enable grouped query attention.
        training: Whether in training mode (affects dropout).
        backend_hint: Hint for backend selection ("flash", "efficient", "cudnn", "math").
        config: SDPAConfig to use (overrides individual args).

    Returns:
        Attention output tensor (batch, heads, seq_q, head_dim).

    Raises:
        ValueError: If attn_mask and is_causal are both specified.
        RuntimeError: On shape mismatch between key and value.
    """
    # Apply config if provided
    if config is not None:
        is_causal = config.is_causal
        dropout_p = config.dropout_p
        scale = config.scale
        enable_gqa = config.enable_gqa

    # Validate mask + causal constraint
    if attn_mask is not None and is_causal:
        raise ValueError(
            "Cannot use both attn_mask and is_causal=True. "
            "Use is_causal=False with a causal mask, or is_causal=True without a mask."
        )

    # Validate key/value shape match
    if key.shape[2] != value.shape[2]:
        raise RuntimeError(
            f"Key and value sequence lengths must match: "
            f"key.shape[2]={key.shape[2]}, value.shape[2]={value.shape[2]}"
        )

    # Apply dropout only during training
    effective_dropout = dropout_p if training else 0.0

    # Select backend if hint provided
    if backend_hint is not None:
        return _sdpa_with_backend(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=effective_dropout,
            scale=scale,
            enable_gqa=enable_gqa,
            backend_hint=backend_hint,
        )

    # Default: let PyTorch auto-select
    return F.scaled_dot_product_attention(
        query=query,
        key=key,
        value=value,
        attn_mask=attn_mask,
        is_causal=is_causal,
        dropout_p=effective_dropout,
        scale=scale,
        enable_gqa=enable_gqa,
    )


def _sdpa_with_backend(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None,
    is_causal: bool,
    dropout_p: float,
    scale: float | None,
    enable_gqa: bool,
    backend_hint: str,
) -> torch.Tensor:
    """Execute SDPA with specific backend hint.

    Uses torch.nn.attention.sdpa_kernel context manager to
    select the desired backend.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attn_mask: Attention mask.
        is_causal: Causal masking flag.
        dropout_p: Dropout probability.
        scale: Attention scale.
        enable_gqa: GQA flag.
        backend_hint: Backend to use ("flash", "efficient", "cudnn", "math").

    Returns:
        Attention output tensor.
    """
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
    except ImportError:
        # Older PyTorch version without sdpa_kernel
        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    # Map hint to backend
    backend_map = {
        "flash": SDPBackend.FLASH_ATTENTION,
        "efficient": SDPBackend.EFFICIENT_ATTENTION,
        "cudnn": SDPBackend.CUDNN_ATTENTION,
        "math": SDPBackend.MATH,
    }

    backend = backend_map.get(backend_hint.lower())
    if backend is None:
        # Unknown hint, fall back to auto
        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    # Use context manager to select backend
    with sdpa_kernel([backend]):
        return F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=scale,
            enable_gqa=enable_gqa,
        )
