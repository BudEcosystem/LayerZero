"""Pytest fixtures for FlashInfer tests."""
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
def flashinfer_available() -> bool:
    """Check if FlashInfer is available."""
    try:
        import flashinfer
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_flashinfer(flashinfer_available: bool) -> None:
    """Skip test if FlashInfer is not available."""
    if not flashinfer_available:
        pytest.skip("FlashInfer not installed")


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
def sample_qkv_nhd(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample Q, K, V tensors in NHD layout.

    Shape: (num_tokens=256, heads=8, dim=64)
    """
    num_tokens, heads, dim = 256, 8, 64
    dtype = torch.float16

    q = torch.randn(num_tokens, heads, dim, device=device, dtype=dtype)
    k = torch.randn(num_tokens, heads, dim, device=device, dtype=dtype)
    v = torch.randn(num_tokens, heads, dim, device=device, dtype=dtype)

    return q, k, v


@pytest.fixture
def sample_gqa_tensors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample tensors for GQA testing.

    Q shape: (batch=2, seq=128, heads=8, dim=64)
    K/V shape: (batch=2, seq=128, kv_heads=2, dim=64)
    """
    batch, seq, q_heads, kv_heads, dim = 2, 128, 8, 2, 64
    dtype = torch.float16

    q = torch.randn(batch, seq, q_heads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, kv_heads, dim, device=device, dtype=dtype)
    v = torch.randn(batch, seq, kv_heads, dim, device=device, dtype=dtype)

    return q, k, v


@pytest.fixture
def sample_paged_kv_cache(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample paged KV cache.

    KV cache shape: (num_blocks=16, 2, block_size=16, heads=8, dim=64)
    Block table shape: (batch=2, max_blocks=8)
    Seq lens: (batch=2,)
    """
    num_blocks, block_size, heads, dim = 16, 16, 8, 64
    batch, max_blocks = 2, 8
    dtype = torch.float16

    kv_cache = torch.randn(
        num_blocks, 2, block_size, heads, dim,
        device=device, dtype=dtype
    )
    block_table = torch.randint(
        0, num_blocks, (batch, max_blocks),
        device=device, dtype=torch.int32
    )
    seq_lens = torch.tensor([100, 80], device=device, dtype=torch.int32)

    return kv_cache, block_table, seq_lens
