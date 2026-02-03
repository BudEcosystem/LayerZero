"""Fixtures for integration tests."""
from __future__ import annotations

import pytest
import torch


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "integration: mark test as integration test")
    config.addinivalue_line("markers", "gpu: mark test as requiring GPU")
    config.addinivalue_line("markers", "slow: mark test as slow")


@pytest.fixture
def device() -> torch.device:
    """Get available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def dtype() -> torch.dtype:
    """Default dtype for testing."""
    return torch.float16 if torch.cuda.is_available() else torch.float32


@pytest.fixture
def small_batch_size() -> int:
    """Small batch size for testing."""
    return 2


@pytest.fixture
def medium_batch_size() -> int:
    """Medium batch size for testing."""
    return 8


@pytest.fixture
def small_seq_len() -> int:
    """Small sequence length."""
    return 64


@pytest.fixture
def medium_seq_len() -> int:
    """Medium sequence length."""
    return 512


@pytest.fixture
def large_seq_len() -> int:
    """Large sequence length."""
    return 2048


@pytest.fixture
def llama_config() -> dict:
    """Small LLaMA-like config for testing."""
    return {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "head_dim": 64,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "rms_norm_eps": 1e-6,
    }


@pytest.fixture
def gpt_config() -> dict:
    """Small GPT-like config for testing."""
    return {
        "hidden_size": 256,
        "num_attention_heads": 4,
        "head_dim": 64,
        "intermediate_size": 512,
        "num_hidden_layers": 2,
        "vocab_size": 1000,
        "layer_norm_eps": 1e-5,
    }


@pytest.fixture
def vit_config() -> dict:
    """Small ViT-like config for testing."""
    return {
        "hidden_size": 192,
        "num_attention_heads": 3,
        "head_dim": 64,
        "intermediate_size": 384,
        "num_hidden_layers": 2,
        "image_size": 224,
        "patch_size": 16,
    }


@pytest.fixture
def sample_qkv() -> dict[str, torch.Tensor]:
    """Sample Q, K, V tensors for attention tests (CPU)."""
    batch_size = 2
    seq_len = 64
    num_heads = 4
    head_dim = 64

    return {
        "query": torch.randn(batch_size, seq_len, num_heads, head_dim),
        "key": torch.randn(batch_size, seq_len, num_heads, head_dim),
        "value": torch.randn(batch_size, seq_len, num_heads, head_dim),
    }


@pytest.fixture
def sample_qkv_cuda() -> dict[str, torch.Tensor]:
    """Sample Q, K, V tensors for attention tests (CUDA)."""
    batch_size = 2
    seq_len = 64
    num_heads = 4
    head_dim = 64

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return {
        "query": torch.randn(batch_size, seq_len, num_heads, head_dim, device=device),
        "key": torch.randn(batch_size, seq_len, num_heads, head_dim, device=device),
        "value": torch.randn(batch_size, seq_len, num_heads, head_dim, device=device),
    }


@pytest.fixture
def sample_norm_input() -> dict[str, torch.Tensor]:
    """Sample input for normalization tests."""
    batch_size = 2
    seq_len = 64
    hidden_size = 256

    return {
        "input": torch.randn(batch_size, seq_len, hidden_size),
        "weight": torch.ones(hidden_size),
        "bias": torch.zeros(hidden_size),
    }
