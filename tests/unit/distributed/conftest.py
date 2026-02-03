"""Pytest fixtures for distributed tests."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_dist_available():
    """Mock torch.distributed as available."""
    with patch('torch.distributed.is_available', return_value=True):
        with patch('torch.distributed.is_initialized', return_value=True):
            yield


@pytest.fixture
def mock_dist_unavailable():
    """Mock torch.distributed as unavailable."""
    with patch('torch.distributed.is_available', return_value=False):
        yield


@pytest.fixture
def mock_rank_0():
    """Mock as rank 0."""
    with patch('torch.distributed.is_available', return_value=True):
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_rank', return_value=0):
                with patch('torch.distributed.get_world_size', return_value=4):
                    yield


@pytest.fixture
def mock_rank_1():
    """Mock as rank 1 (non-root)."""
    with patch('torch.distributed.is_available', return_value=True):
        with patch('torch.distributed.is_initialized', return_value=True):
            with patch('torch.distributed.get_rank', return_value=1):
                with patch('torch.distributed.get_world_size', return_value=4):
                    yield


@pytest.fixture
def mock_kernel():
    """Create mock kernel for testing."""
    kernel = MagicMock()
    kernel.kernel_id = "test_kernel"
    kernel.operation = "attention"
    kernel.tp_invariant = False
    return kernel


@pytest.fixture
def mock_tp_kernel():
    """Create mock TP-invariant kernel."""
    kernel = MagicMock()
    kernel.kernel_id = "tp_kernel"
    kernel.operation = "attention"
    kernel.tp_invariant = True
    return kernel


@pytest.fixture
def mock_selection_context():
    """Create mock selection context."""
    ctx = MagicMock()
    ctx.tp_size = 4
    ctx.tp_rank = 0
    ctx.is_training = False
    return ctx
