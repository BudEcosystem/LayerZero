"""LayerZero Positional Encoding Operations.

This module provides positional encoding implementations:
- ALiBi (Attention with Linear Biases)
- RoPE (Rotary Positional Embeddings) - planned
- Sinusoidal embeddings - planned
"""
from __future__ import annotations

from layerzero.posenc.alibi import (
    get_alibi_slopes,
    get_alibi_bias,
    get_alibi_bias_causal,
    build_alibi_tensor,
)

__all__ = [
    "get_alibi_slopes",
    "get_alibi_bias",
    "get_alibi_bias_causal",
    "build_alibi_tensor",
]
