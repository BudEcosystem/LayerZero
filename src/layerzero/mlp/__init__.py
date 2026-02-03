"""LayerZero MLP (Feed-Forward Network) Operations.

This module provides MLP layer implementations:
- Gated activations (SwiGLU, GeGLU, ReGLU)
- Fused gate-up-activation-down patterns
- Linear/GEMM operations
"""
from __future__ import annotations

from layerzero.mlp.fused import (
    swiglu,
    geglu,
    reglu,
    fused_mlp,
)
from layerzero.mlp.linear import linear

__all__ = [
    "swiglu",
    "geglu",
    "reglu",
    "fused_mlp",
    "linear",
]
