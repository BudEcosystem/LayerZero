"""
PyTest Configuration for LayerZero Tests

Provides fixtures, markers, and test setup.
"""
import sys
from pathlib import Path

import pytest
import torch

# Add src directory to path for imports
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

# Add tests directory to path for fixtures
tests_path = Path(__file__).parent
sys.path.insert(0, str(tests_path))


def pytest_configure(config):
    """Configure custom markers."""
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "multigpu: mark test as requiring multiple GPUs")
    config.addinivalue_line("markers", "stress: mark test as stress test")
    config.addinivalue_line("markers", "slow: mark test as slow running")
    config.addinivalue_line("markers", "correctness: mark test as correctness test")
    config.addinivalue_line("markers", "distributed: mark test as distributed/multi-rank test")


@pytest.fixture(scope="session")
def has_cuda():
    """Check if CUDA is available."""
    try:
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture(scope="session")
def cuda_device():
    """Get CUDA device if available, otherwise skip."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    pytest.skip("CUDA not available")


@pytest.fixture
def device() -> torch.device:
    """Get the best available device (prefers CUDA)."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


@pytest.fixture
def sample_reason():
    """Create a sample Reason for testing."""
    from layerzero.reasons import Reason, ReasonCategory
    return Reason(
        code="SAMPLE_CODE",
        message="Sample message for testing",
        category=ReasonCategory.HARDWARE
    )


@pytest.fixture
def sample_tensors(device: torch.device) -> dict[str, torch.Tensor]:
    """Create sample Q, K, V tensors for attention tests."""
    batch_size = 2
    num_heads = 4
    seq_len = 16
    head_dim = 32

    return {
        "query": torch.randn(batch_size, num_heads, seq_len, head_dim, device=device),
        "key": torch.randn(batch_size, num_heads, seq_len, head_dim, device=device),
        "value": torch.randn(batch_size, num_heads, seq_len, head_dim, device=device),
    }


@pytest.fixture
def sample_attention_mask(device: torch.device) -> torch.Tensor:
    """Create sample attention mask."""
    seq_len = 16
    # Causal mask
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.masked_fill(mask == 0, float("-inf"))


@pytest.fixture
def random_seed() -> int:
    """Provide reproducible random seed."""
    seed = 42
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed
