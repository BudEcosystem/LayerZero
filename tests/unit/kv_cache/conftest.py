"""Fixtures for KV cache tests."""
from __future__ import annotations

import pytest
import torch

from layerzero.device import GPUGeneration


@pytest.fixture
def hopper_device() -> GPUGeneration:
    """Hopper GPU generation fixture."""
    return GPUGeneration.HOPPER


@pytest.fixture
def blackwell_device() -> GPUGeneration:
    """Blackwell GPU generation fixture."""
    return GPUGeneration.BLACKWELL


@pytest.fixture
def ampere_device() -> GPUGeneration:
    """Ampere GPU generation fixture."""
    return GPUGeneration.AMPERE


@pytest.fixture
def sample_kv_tensor_nhd() -> torch.Tensor:
    """Sample KV cache tensor in NHD layout (num_blocks, num_heads, head_dim)."""
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64
    return torch.randn(batch_size, seq_len, num_heads, head_dim)


@pytest.fixture
def sample_kv_tensor_hnd() -> torch.Tensor:
    """Sample KV cache tensor in HND layout (num_heads, num_blocks, head_dim)."""
    batch_size = 2
    seq_len = 128
    num_heads = 8
    head_dim = 64
    return torch.randn(batch_size, num_heads, seq_len, head_dim)


@pytest.fixture
def sample_block_table() -> torch.Tensor:
    """Sample block table for paged attention."""
    num_seqs = 4
    max_blocks = 32
    return torch.randint(0, 1000, (num_seqs, max_blocks), dtype=torch.int32)


@pytest.fixture
def page_size() -> int:
    """Default page size."""
    return 16
