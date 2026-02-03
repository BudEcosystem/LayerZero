"""Pytest fixtures for CPU backend tests."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def cpu_device() -> torch.device:
    """Get CPU device."""
    return torch.device("cpu")


@pytest.fixture
def sample_cpu_matrices(
    cpu_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample matrices for CPU testing.

    Shape: (batch=4, M=256, K=512) and (batch=4, K=512, N=256)
    """
    batch, m, k, n = 4, 256, 512, 256
    dtype = torch.float32

    a = torch.randn(batch, m, k, device=cpu_device, dtype=dtype)
    b = torch.randn(batch, k, n, device=cpu_device, dtype=dtype)

    return a, b


@pytest.fixture
def sample_cpu_attention_inputs(
    cpu_device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create sample attention inputs for CPU testing.

    Shape: (batch=2, heads=8, seq_len=128, head_dim=64)
    """
    batch, heads, seq_len, head_dim = 2, 8, 128, 64
    dtype = torch.float32

    query = torch.randn(batch, heads, seq_len, head_dim, device=cpu_device, dtype=dtype)
    key = torch.randn(batch, heads, seq_len, head_dim, device=cpu_device, dtype=dtype)
    value = torch.randn(batch, heads, seq_len, head_dim, device=cpu_device, dtype=dtype)

    return query, key, value


@pytest.fixture
def sample_cpu_layernorm_input(
    cpu_device: torch.device,
) -> torch.Tensor:
    """Create sample input for LayerNorm testing.

    Shape: (batch=4, seq_len=128, hidden_dim=512)
    """
    batch, seq_len, hidden_dim = 4, 128, 512
    dtype = torch.float32

    return torch.randn(batch, seq_len, hidden_dim, device=cpu_device, dtype=dtype)
