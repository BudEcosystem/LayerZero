"""Linear/GEMM Operations for LayerZero.

Provides linear (matrix multiplication) operations with unified interface
for different backends and optimizations.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F


def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Linear transformation: y = x @ weight.T + bias.

    General-purpose linear operation (GEMM) with optional bias.
    This function provides a unified interface that can be optimized
    with different backends (cuBLAS, CUTLASS, etc.).

    Args:
        x: Input tensor, shape (..., in_features).
        weight: Weight matrix, shape (out_features, in_features).
        bias: Optional bias vector, shape (out_features,).

    Returns:
        Output tensor, shape (..., out_features).

    Example:
        >>> x = torch.randn(batch, seq_len, hidden_dim)
        >>> weight = torch.randn(out_dim, hidden_dim)
        >>> output = linear(x, weight)  # (batch, seq_len, out_dim)
    """
    return F.linear(x, weight, bias)


def linear_fused_bias_gelu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused linear + bias + GELU activation.

    Computes: gelu(x @ weight.T + bias)

    This pattern is common in transformer FFN layers and can be
    optimized with fused kernels on GPU.

    Args:
        x: Input tensor, shape (..., in_features).
        weight: Weight matrix, shape (out_features, in_features).
        bias: Optional bias vector, shape (out_features,).

    Returns:
        Output tensor with GELU activation applied.
    """
    out = F.linear(x, weight, bias)
    return F.gelu(out)


def linear_fused_bias_relu(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused linear + bias + ReLU activation.

    Computes: relu(x @ weight.T + bias)

    Args:
        x: Input tensor.
        weight: Weight matrix.
        bias: Optional bias.

    Returns:
        Output tensor with ReLU activation applied.
    """
    out = F.linear(x, weight, bias)
    return F.relu(out)


def batched_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Batched linear operation.

    For cases where weight matrices differ per batch element.

    Args:
        x: Input tensor, shape (batch, ..., in_features).
        weight: Weight matrices, shape (batch, out_features, in_features).
        bias: Optional bias, shape (batch, out_features) or (out_features,).

    Returns:
        Output tensor, shape (batch, ..., out_features).
    """
    # Use einsum for batched matmul: b...i,boi->b...o
    out = torch.einsum('b...i,boi->b...o', x, weight)

    if bias is not None:
        out = out + bias.unsqueeze(-2) if bias.dim() == 2 else out + bias

    return out


def column_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    world_size: int = 1,
    rank: int = 0,
) -> torch.Tensor:
    """Column-parallel linear for tensor parallelism.

    Weight is partitioned along output dimension across devices.
    Each device computes a slice of the output features.

    Args:
        x: Input tensor, same on all ranks.
        weight: Local weight slice, shape (local_out_features, in_features).
        bias: Optional local bias slice.
        world_size: Number of parallel processes.
        rank: Current process rank.

    Returns:
        Local output slice, shape (..., local_out_features).

    Note:
        Results must be gathered across ranks for full output.
    """
    # Standard linear on local slice
    return F.linear(x, weight, bias)


def row_parallel_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    world_size: int = 1,
    rank: int = 0,
) -> torch.Tensor:
    """Row-parallel linear for tensor parallelism.

    Weight is partitioned along input dimension across devices.
    Each device has a slice of input features.

    Args:
        x: Local input slice, shape (..., local_in_features).
        weight: Local weight slice, shape (out_features, local_in_features).
        bias: Optional bias (only added on rank 0).
        world_size: Number of parallel processes.
        rank: Current process rank.

    Returns:
        Partial output, shape (..., out_features).

    Note:
        Results must be all-reduced across ranks for final output.
    """
    out = F.linear(x, weight)

    # Bias only added on one rank (will be summed in all-reduce)
    if bias is not None and rank == 0:
        out = out + bias

    return out
