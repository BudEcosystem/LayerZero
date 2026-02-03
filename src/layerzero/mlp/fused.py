"""Fused MLP Operations for LayerZero.

Implements gated activation functions and fused MLP patterns used in
modern transformer architectures (LLaMA, PaLM, Falcon, etc.).

Gated Linear Units (GLUs):
- SwiGLU: silu(gate) * up - LLaMA, Mistral
- GeGLU: gelu(gate) * up - GPT-J variants
- ReGLU: relu(gate) * up - GLU paper original

Reference:
- GLU Variants Improve Transformer (Shazeer, 2020)
- LLaMA: Open and Efficient Foundation Language Models (Touvron et al., 2023)
"""
from __future__ import annotations

from typing import Optional, Literal

import torch
import torch.nn.functional as F


def swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """SwiGLU activation: silu(gate) * up.

    Swish-Gated Linear Unit activation used in LLaMA, Mistral, and
    other modern LLMs. Combines Swish (SiLU) activation with gating.

    Args:
        gate: Gate tensor after projection, shape (..., intermediate_dim).
        up: Up-projection tensor, shape (..., intermediate_dim).
        inplace: Whether to perform operation inplace on gate tensor.

    Returns:
        Activated tensor, shape (..., intermediate_dim).

    Example:
        >>> gate = x @ gate_proj.t()  # (batch, seq, intermediate)
        >>> up = x @ up_proj.t()      # (batch, seq, intermediate)
        >>> hidden = swiglu(gate, up)  # (batch, seq, intermediate)
    """
    if inplace:
        gate = F.silu(gate, inplace=True)
        gate.mul_(up)
        return gate
    else:
        return F.silu(gate) * up


def geglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """GeGLU activation: gelu(gate) * up.

    GELU-Gated Linear Unit activation. Combines GELU activation with gating.

    Args:
        gate: Gate tensor after projection.
        up: Up-projection tensor.
        inplace: Whether to perform operation inplace (not supported for GELU).

    Returns:
        Activated tensor.

    Example:
        >>> hidden = geglu(gate, up)
    """
    # GELU doesn't support inplace in PyTorch
    return F.gelu(gate) * up


def reglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    inplace: bool = False,
) -> torch.Tensor:
    """ReGLU activation: relu(gate) * up.

    ReLU-Gated Linear Unit activation from the original GLU paper.

    Args:
        gate: Gate tensor after projection.
        up: Up-projection tensor.
        inplace: Whether to perform operation inplace on gate tensor.

    Returns:
        Activated tensor.

    Example:
        >>> hidden = reglu(gate, up)
    """
    if inplace:
        gate = F.relu(gate, inplace=True)
        gate.mul_(up)
        return gate
    else:
        return F.relu(gate) * up


def fused_mlp(
    x: torch.Tensor,
    gate_proj: Optional[torch.Tensor],
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    activation: Literal["swiglu", "geglu", "reglu", "gelu", "relu", "silu"] = "swiglu",
    gate_bias: Optional[torch.Tensor] = None,
    up_bias: Optional[torch.Tensor] = None,
    down_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused MLP forward pass.

    Computes the complete MLP/FFN forward pass:
    1. Gate projection (optional): gate = x @ gate_proj.T + gate_bias
    2. Up projection: up = x @ up_proj.T + up_bias
    3. Activation: hidden = activation(gate, up) or activation(up)
    4. Down projection: output = hidden @ down_proj.T + down_bias

    For gated activations (SwiGLU, GeGLU, ReGLU), gate_proj must be provided.
    For non-gated activations (GELU, ReLU, SiLU), gate_proj should be None.

    Args:
        x: Input tensor, shape (batch, seq_len, hidden_dim).
        gate_proj: Gate projection weight, shape (intermediate_dim, hidden_dim).
                   Required for gated activations, None for non-gated.
        up_proj: Up projection weight, shape (intermediate_dim, hidden_dim).
        down_proj: Down projection weight, shape (hidden_dim, intermediate_dim).
        activation: Activation function name.
        gate_bias: Optional gate projection bias.
        up_bias: Optional up projection bias.
        down_bias: Optional down projection bias.

    Returns:
        Output tensor, shape (batch, seq_len, hidden_dim).

    Example:
        >>> # LLaMA-style SwiGLU MLP
        >>> output = fused_mlp(
        ...     x, gate_proj, up_proj, down_proj,
        ...     activation="swiglu"
        ... )
    """
    # Determine if using gated activation
    gated_activations = {"swiglu", "geglu", "reglu"}
    is_gated = activation in gated_activations

    if is_gated:
        if gate_proj is None:
            raise ValueError(
                f"gate_proj required for gated activation '{activation}'"
            )

        # Compute gate and up projections
        gate = F.linear(x, gate_proj, gate_bias)
        up = F.linear(x, up_proj, up_bias)

        # Apply gated activation
        if activation == "swiglu":
            hidden = swiglu(gate, up)
        elif activation == "geglu":
            hidden = geglu(gate, up)
        elif activation == "reglu":
            hidden = reglu(gate, up)
        else:
            raise ValueError(f"Unknown gated activation: {activation}")
    else:
        # Non-gated: single up projection + activation
        up = F.linear(x, up_proj, up_bias)

        if activation == "gelu":
            hidden = F.gelu(up)
        elif activation == "relu":
            hidden = F.relu(up)
        elif activation == "silu":
            hidden = F.silu(up)
        else:
            raise ValueError(f"Unknown activation: {activation}")

    # Down projection
    output = F.linear(hidden, down_proj, down_bias)

    return output


def fused_mlp_chunked(
    x: torch.Tensor,
    gate_proj: Optional[torch.Tensor],
    up_proj: torch.Tensor,
    down_proj: torch.Tensor,
    activation: str = "swiglu",
    chunk_size: int = 1024,
    **kwargs,
) -> torch.Tensor:
    """Memory-efficient fused MLP with chunking.

    For very long sequences, processes in chunks to reduce peak memory.

    Args:
        x: Input tensor, shape (batch, seq_len, hidden_dim).
        gate_proj: Gate projection weight.
        up_proj: Up projection weight.
        down_proj: Down projection weight.
        activation: Activation function name.
        chunk_size: Maximum sequence length per chunk.
        **kwargs: Additional arguments for fused_mlp.

    Returns:
        Output tensor, shape (batch, seq_len, hidden_dim).
    """
    batch_size, seq_len, hidden_dim = x.shape

    if seq_len <= chunk_size:
        # No chunking needed
        return fused_mlp(x, gate_proj, up_proj, down_proj, activation, **kwargs)

    # Process in chunks
    outputs = []
    for i in range(0, seq_len, chunk_size):
        chunk = x[:, i:i + chunk_size, :]
        out = fused_mlp(chunk, gate_proj, up_proj, down_proj, activation, **kwargs)
        outputs.append(out)

    return torch.cat(outputs, dim=1)
