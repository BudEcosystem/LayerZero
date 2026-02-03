"""
Reference implementations for correctness testing.

Provides PyTorch reference implementations for comparing against
optimized LayerZero kernels.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


# Tolerance mapping by dtype
# FP16 needs slightly looser tolerances due to SDPA implementation differences
DTYPE_TOLERANCES: dict[torch.dtype, tuple[float, float]] = {
    torch.float32: (1e-4, 1e-5),
    torch.float64: (1e-5, 1e-6),
    torch.float16: (2e-3, 2e-3),
    torch.bfloat16: (1e-2, 1e-2),
}


def get_tolerance(dtype: torch.dtype) -> tuple[float, float]:
    """Get rtol and atol for a given dtype.

    Args:
        dtype: PyTorch dtype.

    Returns:
        Tuple of (rtol, atol).
    """
    return DTYPE_TOLERANCES.get(dtype, (1e-4, 1e-5))


def assert_close(
    actual: torch.Tensor,
    expected: torch.Tensor,
    dtype: torch.dtype | None = None,
    rtol: float | None = None,
    atol: float | None = None,
) -> None:
    """Assert two tensors are close within dtype-specific tolerances.

    Args:
        actual: Actual tensor output.
        expected: Expected tensor output.
        dtype: Dtype for tolerance lookup (default: actual.dtype).
        rtol: Override relative tolerance.
        atol: Override absolute tolerance.

    Raises:
        AssertionError: If tensors are not close.
    """
    if dtype is None:
        dtype = actual.dtype

    if rtol is None or atol is None:
        default_rtol, default_atol = get_tolerance(dtype)
        rtol = rtol if rtol is not None else default_rtol
        atol = atol if atol is not None else default_atol

    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """PyTorch reference attention implementation.

    Implements scaled dot-product attention using standard PyTorch operations.

    Args:
        query: Query tensor [B, H, L, D].
        key: Key tensor [B, H, S, D].
        value: Value tensor [B, H, S, D].
        attn_mask: Optional attention mask.
        dropout_p: Dropout probability.
        is_causal: Whether to apply causal masking.
        scale: Optional scale factor (default: 1/sqrt(D)).

    Returns:
        Attention output tensor [B, H, L, D].
    """
    # Get dimensions
    batch_size, num_heads, seq_len_q, head_dim = query.shape
    _, _, seq_len_k, _ = key.shape

    # Compute scale
    if scale is None:
        scale = head_dim ** -0.5

    # Compute attention scores
    attn_weights = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask if requested
    if is_causal:
        causal_mask = torch.triu(
            torch.ones(seq_len_q, seq_len_k, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        attn_weights = attn_weights.masked_fill(causal_mask, float("-inf"))

    # Apply attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_weights = attn_weights.masked_fill(~attn_mask, float("-inf"))
        else:
            attn_weights = attn_weights + attn_mask

    # Softmax
    attn_weights = F.softmax(attn_weights, dim=-1)

    # Handle NaN from all-masked positions
    attn_weights = torch.nan_to_num(attn_weights, nan=0.0)

    # Dropout
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    # Apply to values
    output = torch.matmul(attn_weights, value)

    return output


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Reference RMS normalization implementation.

    Args:
        x: Input tensor [..., hidden_size].
        weight: Weight tensor [hidden_size].
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor.
    """
    # Compute RMS
    variance = x.pow(2).mean(-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)

    # Apply weight
    return x_normed * weight


def reference_layer_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Reference layer normalization implementation.

    Args:
        x: Input tensor [..., hidden_size].
        weight: Weight tensor [hidden_size].
        bias: Optional bias tensor [hidden_size].
        eps: Epsilon for numerical stability.

    Returns:
        Normalized tensor.
    """
    return F.layer_norm(x, weight.shape, weight, bias, eps)


def reference_rotary_embedding(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Reference rotary embedding (RoPE) implementation.

    Args:
        x: Input tensor [B, H, L, D].
        cos: Cosine tensor for positions [1, 1, L, D].
        sin: Sine tensor for positions [1, 1, L, D].

    Returns:
        Tensor with rotary embeddings applied.
    """
    # Split into even/odd components
    x1 = x[..., ::2]
    x2 = x[..., 1::2]

    # Reshape cos/sin for broadcasting
    cos = cos[..., ::2]
    sin = sin[..., ::2]

    # Apply rotation
    out = torch.stack([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos,
    ], dim=-1)

    return out.flatten(-2)


def reference_swiglu(
    x: torch.Tensor,
    w_gate: torch.Tensor,
    w_up: torch.Tensor,
    w_down: torch.Tensor,
) -> torch.Tensor:
    """Reference SwiGLU implementation.

    Args:
        x: Input tensor [..., hidden_size].
        w_gate: Gate projection weight.
        w_up: Up projection weight.
        w_down: Down projection weight.

    Returns:
        Output tensor.
    """
    gate = F.silu(F.linear(x, w_gate))
    up = F.linear(x, w_up)
    return F.linear(gate * up, w_down)


def reference_cross_entropy(
    logits: torch.Tensor,
    labels: torch.Tensor,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Reference cross entropy loss implementation.

    Args:
        logits: Logits tensor [B, L, V] or [B*L, V].
        labels: Label tensor [B, L] or [B*L].
        ignore_index: Index to ignore in loss.

    Returns:
        Loss tensor.
    """
    # Reshape if needed
    if logits.dim() == 3:
        logits = logits.view(-1, logits.size(-1))
        labels = labels.view(-1)

    return F.cross_entropy(logits, labels, ignore_index=ignore_index)
