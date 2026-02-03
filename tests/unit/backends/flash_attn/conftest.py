"""Shared fixtures for FlashAttention tests."""
from __future__ import annotations

import pytest
import torch


def _has_cuda() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def _get_device() -> torch.device:
    """Get appropriate device for testing."""
    return torch.device("cuda" if _has_cuda() else "cpu")


def _flash_attn_available() -> bool:
    """Check if flash_attn is installed."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


@pytest.fixture
def device() -> torch.device:
    """Get test device (CUDA if available, else CPU)."""
    return _get_device()


@pytest.fixture
def query_bshd(device: torch.device) -> torch.Tensor:
    """Create query tensor in BSHD layout.

    Shape: (batch=2, seq_len=16, num_heads=8, head_dim=64)
    """
    return torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)


@pytest.fixture
def key_bshd(device: torch.device) -> torch.Tensor:
    """Create key tensor in BSHD layout.

    Shape: (batch=2, seq_len=16, num_heads=8, head_dim=64)
    """
    return torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)


@pytest.fixture
def value_bshd(device: torch.device) -> torch.Tensor:
    """Create value tensor in BSHD layout.

    Shape: (batch=2, seq_len=16, num_heads=8, head_dim=64)
    """
    return torch.randn(2, 16, 8, 64, device=device, dtype=torch.float16)


@pytest.fixture
def query_bhsd(device: torch.device) -> torch.Tensor:
    """Create query tensor in BHSD layout (PyTorch standard).

    Shape: (batch=2, num_heads=8, seq_len=16, head_dim=64)
    """
    return torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)


@pytest.fixture
def key_bhsd(device: torch.device) -> torch.Tensor:
    """Create key tensor in BHSD layout.

    Shape: (batch=2, num_heads=8, seq_len=16, head_dim=64)
    """
    return torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)


@pytest.fixture
def value_bhsd(device: torch.device) -> torch.Tensor:
    """Create value tensor in BHSD layout.

    Shape: (batch=2, num_heads=8, seq_len=16, head_dim=64)
    """
    return torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)


@pytest.fixture
def qkv_bf16_bshd(device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create Q, K, V tensors in bf16 BSHD layout."""
    q = torch.randn(2, 16, 8, 64, device=device, dtype=torch.bfloat16)
    k = torch.randn(2, 16, 8, 64, device=device, dtype=torch.bfloat16)
    v = torch.randn(2, 16, 8, 64, device=device, dtype=torch.bfloat16)
    return q, k, v


@pytest.fixture
def gqa_kv_bshd(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create K, V for GQA (2 KV heads, 8 Q heads).

    Shape: (batch=2, seq_len=16, num_kv_heads=2, head_dim=64)
    """
    k = torch.randn(2, 16, 2, 64, device=device, dtype=torch.float16)
    v = torch.randn(2, 16, 2, 64, device=device, dtype=torch.float16)
    return k, v
