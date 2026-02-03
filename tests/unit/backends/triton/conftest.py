"""Pytest fixtures for Triton custom kernel tests."""
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
def triton_available() -> bool:
    """Check if Triton is available."""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_triton(triton_available: bool) -> None:
    """Skip test if Triton is not available."""
    if not triton_available:
        pytest.skip("Triton not installed")


@pytest.fixture
def skip_if_no_cuda() -> None:
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def sample_vectors(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample vectors for testing.

    Shape: (n_elements=1024,)
    """
    n_elements = 1024
    dtype = torch.float16

    x = torch.randn(n_elements, device=device, dtype=dtype)
    y = torch.randn(n_elements, device=device, dtype=dtype)

    return x, y


@pytest.fixture
def sample_matrices(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample matrices for testing.

    Shape: (batch=2, rows=128, cols=256)
    """
    batch, rows, cols = 2, 128, 256
    dtype = torch.float16

    x = torch.randn(batch, rows, cols, device=device, dtype=dtype)
    y = torch.randn(batch, rows, cols, device=device, dtype=dtype)

    return x, y
