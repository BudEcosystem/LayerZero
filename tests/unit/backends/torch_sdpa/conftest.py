"""Shared fixtures for Torch SDPA tests."""
from __future__ import annotations

import pytest
import torch


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def _get_device() -> torch.device:
    """Get appropriate device for testing."""
    return torch.device("cuda" if _has_cuda() else "cpu")


@pytest.fixture
def device() -> torch.device:
    """Get test device (CUDA if available, else CPU)."""
    return _get_device()


@pytest.fixture
def query_tensor(device: torch.device) -> torch.Tensor:
    """Create query tensor for attention tests.

    Shape: (batch=2, num_heads=8, seq_len=16, head_dim=64)
    """
    return torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)


@pytest.fixture
def key_tensor(device: torch.device) -> torch.Tensor:
    """Create key tensor for attention tests.

    Shape: (batch=2, num_heads=8, seq_len=16, head_dim=64)
    """
    return torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)


@pytest.fixture
def value_tensor(device: torch.device) -> torch.Tensor:
    """Create value tensor for attention tests.

    Shape: (batch=2, num_heads=8, seq_len=16, head_dim=64)
    """
    return torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)


@pytest.fixture
def qkv_fp32(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q, K, V tensors in fp32."""
    q = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float32)
    k = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float32)
    v = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float32)
    return q, k, v


@pytest.fixture
def qkv_bf16(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q, K, V tensors in bf16."""
    q = torch.randn(2, 8, 16, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 8, 16, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(2, 8, 16, 64, device=device, dtype=torch.bfloat16)
    return q, k, v


@pytest.fixture
def bool_mask(device: torch.device) -> torch.Tensor:
    """Create boolean attention mask.

    Shape: (seq_len=16, seq_len=16)
    True = attend, False = mask out
    """
    mask = torch.ones(16, 16, dtype=torch.bool, device=device)
    # Create causal-like pattern
    mask = torch.tril(mask)
    return mask


@pytest.fixture
def float_mask(device: torch.device) -> torch.Tensor:
    """Create float attention mask (additive).

    Shape: (seq_len=16, seq_len=16)
    0 = attend, -inf = mask out
    """
    mask = torch.zeros(16, 16, device=device, dtype=torch.float16)
    # Create causal pattern with -inf
    mask = torch.triu(torch.full_like(mask, float("-inf")), diagonal=1)
    return mask


@pytest.fixture
def gqa_key_tensor(device: torch.device) -> torch.Tensor:
    """Create key tensor for GQA (fewer KV heads).

    Query has 8 heads, KV has 2 heads (4x GQA ratio).
    Shape: (batch=2, num_kv_heads=2, seq_len=16, head_dim=64)
    """
    return torch.randn(2, 2, 16, 64, device=device, dtype=torch.float16)


@pytest.fixture
def gqa_value_tensor(device: torch.device) -> torch.Tensor:
    """Create value tensor for GQA (fewer KV heads).

    Query has 8 heads, KV has 2 heads (4x GQA ratio).
    Shape: (batch=2, num_kv_heads=2, seq_len=16, head_dim=64)
    """
    return torch.randn(2, 2, 16, 64, device=device, dtype=torch.float16)


@pytest.fixture
def noncontiguous_query(device: torch.device) -> torch.Tensor:
    """Create non-contiguous query tensor.

    Made non-contiguous via transpose.
    """
    # Original shape: (2, 16, 8, 64) -> transpose to (2, 8, 16, 64)
    t = torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)
    return t.transpose(1, 2)  # Now non-contiguous


@pytest.fixture
def head_dim_84_qkv(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q, K, V with non-power-of-2 head_dim=84."""
    q = torch.randn(2, 8, 16, 84, device=device, dtype=torch.float16)
    k = torch.randn(2, 8, 16, 84, device=device, dtype=torch.float16)
    v = torch.randn(2, 8, 16, 84, device=device, dtype=torch.float16)
    return q, k, v


def reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    is_causal: bool = False,
    scale: float | None = None,
) -> torch.Tensor:
    """Naive reference attention implementation.

    Args:
        query: (batch, heads, seq_q, head_dim)
        key: (batch, heads, seq_k, head_dim)
        value: (batch, heads, seq_k, head_dim)
        attn_mask: Optional mask
        is_causal: Use causal mask
        scale: Attention scale (default: 1/sqrt(head_dim))

    Returns:
        Attention output (batch, heads, seq_q, head_dim)
    """
    # Use float32 for numerical stability in reference
    dtype = query.dtype
    query = query.float()
    key = key.float()
    value = value.float()

    head_dim = query.size(-1)
    scale = scale if scale is not None else (1.0 / (head_dim ** 0.5))

    # Compute attention scores
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    # Apply causal mask
    if is_causal:
        seq_q = query.size(-2)
        seq_k = key.size(-2)
        causal_mask = torch.triu(
            torch.full((seq_q, seq_k), float("-inf"), device=query.device),
            diagonal=1,
        )
        scores = scores + causal_mask

    # Apply attention mask
    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            scores = scores.masked_fill(~attn_mask, float("-inf"))
        else:
            scores = scores + attn_mask.float()

    # Softmax and output
    attn_weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, value)

    return output.to(dtype)
