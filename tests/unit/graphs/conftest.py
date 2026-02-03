"""Pytest fixtures for CUDA graph safety validation tests."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch
from typing import Any

import torch


@pytest.fixture
def mock_cuda_available():
    """Mock CUDA as available."""
    with patch("torch.cuda.is_available", return_value=True):
        yield


@pytest.fixture
def mock_cuda_unavailable():
    """Mock CUDA as unavailable."""
    with patch("torch.cuda.is_available", return_value=False):
        yield


@pytest.fixture
def mock_memory_stats():
    """Mock CUDA memory statistics."""
    allocated = [1024 * 1024]  # 1MB starting
    reserved = [2 * 1024 * 1024]  # 2MB starting
    max_allocated = [1024 * 1024]

    def memory_allocated(device=0):
        return allocated[0]

    def memory_reserved(device=0):
        return reserved[0]

    def max_memory_allocated(device=0):
        return max_allocated[0]

    def set_allocated(value: int):
        allocated[0] = value

    def set_reserved(value: int):
        reserved[0] = value

    def set_max_allocated(value: int):
        max_allocated[0] = value

    with patch("torch.cuda.memory_allocated", side_effect=memory_allocated), \
         patch("torch.cuda.memory_reserved", side_effect=memory_reserved), \
         patch("torch.cuda.max_memory_allocated", side_effect=max_memory_allocated), \
         patch("torch.cuda.is_available", return_value=True):
        yield {
            "set_allocated": set_allocated,
            "set_reserved": set_reserved,
            "set_max_allocated": set_max_allocated,
            "get_allocated": lambda: allocated[0],
            "get_reserved": lambda: reserved[0],
        }


@pytest.fixture
def mock_kernel_spec():
    """Create a mock KernelSpec for testing."""
    spec = MagicMock()
    spec.kernel_id = "test_kernel"
    spec.operation = "attention"
    spec.is_cuda_graph_safe = None
    return spec


@pytest.fixture
def mock_kernel_spec_safe():
    """Create a mock KernelSpec marked as graph-safe."""
    spec = MagicMock()
    spec.kernel_id = "safe_kernel"
    spec.operation = "attention"
    spec.is_cuda_graph_safe = True
    return spec


@pytest.fixture
def mock_kernel_spec_unsafe():
    """Create a mock KernelSpec marked as graph-unsafe."""
    spec = MagicMock()
    spec.kernel_id = "unsafe_kernel"
    spec.operation = "dynamic_shape_op"
    spec.is_cuda_graph_safe = False
    return spec


@pytest.fixture
def sample_func():
    """Sample function for capture testing."""
    def func(x: torch.Tensor) -> torch.Tensor:
        return x * 2 + 1

    return func


@pytest.fixture
def sample_tensor():
    """Sample CPU tensor for testing."""
    return torch.randn(4, 4)
