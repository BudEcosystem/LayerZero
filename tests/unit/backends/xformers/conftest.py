"""Pytest fixtures for xFormers tests."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Get test device (prefer CUDA if available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def xformers_available() -> bool:
    """Check if xFormers is available."""
    try:
        import xformers
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_xformers(xformers_available: bool) -> None:
    """Skip test if xFormers is not available."""
    if not xformers_available:
        pytest.skip("xFormers not installed")


@pytest.fixture
def skip_if_no_cuda() -> None:
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def sample_qkv_bshd(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample Q, K, V tensors in BSHD layout.

    Shape: (batch=2, seq=128, heads=8, dim=64)
    """
    batch, seq, heads, dim = 2, 128, 8, 64
    dtype = torch.float16

    q = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)

    return q, k, v


@pytest.fixture
def sample_attn_bias_full(device: torch.device) -> torch.Tensor:
    """Create fully expanded attention bias.

    Shape: (batch=2, heads=8, seq_q=128, seq_k=128)
    """
    batch, heads, seq_q, seq_k = 2, 8, 128, 128
    dtype = torch.float16

    # Create additive bias (zeros = no bias effect)
    bias = torch.zeros(batch, heads, seq_q, seq_k, device=device, dtype=dtype)
    return bias


@pytest.fixture
def sample_attn_bias_broadcast_batch(device: torch.device) -> torch.Tensor:
    """Create attention bias with broadcast batch dim.

    Shape: (1, heads=8, seq_q=128, seq_k=128)
    This will fail xFormers validation.
    """
    heads, seq_q, seq_k = 8, 128, 128
    dtype = torch.float16

    bias = torch.zeros(1, heads, seq_q, seq_k, device=device, dtype=dtype)
    return bias


@pytest.fixture
def sample_attn_bias_broadcast_head(device: torch.device) -> torch.Tensor:
    """Create attention bias with broadcast head dim.

    Shape: (batch=2, 1, seq_q=128, seq_k=128)
    This will fail xFormers validation.
    """
    batch, seq_q, seq_k = 2, 128, 128
    dtype = torch.float16

    bias = torch.zeros(batch, 1, seq_q, seq_k, device=device, dtype=dtype)
    return bias


@pytest.fixture
def sample_gqa_tensors_5d(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample tensors for GQA testing with 5D format.

    Q shape: (batch=2, seq=128, heads_q=8, dim=64) - standard 4D
    K/V shape: (batch=2, seq=128, groups=4, heads_kv=2, dim=64) - 5D GQA format
    """
    batch, seq, heads_q, groups, heads_kv, dim = 2, 128, 8, 4, 2, 64
    dtype = torch.float16

    q = torch.randn(batch, seq, heads_q, dim, device=device, dtype=dtype)
    # 5D format: (B, S, G, Hkv, D) where G = Hq // Hkv
    k = torch.randn(batch, seq, groups, heads_kv, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, groups, heads_kv, dim, device=device, dtype=dtype)

    return q, k, v
