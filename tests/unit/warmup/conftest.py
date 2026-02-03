"""Fixtures for warmup tests."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Generator

import pytest
import torch


@pytest.fixture
def temp_cache_dir() -> Generator[Path, None, None]:
    """Create temporary cache directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_shape_signatures() -> list[dict]:
    """Sample shape signatures for testing."""
    return [
        {
            "operation": "attention.causal",
            "dtype": torch.float16,
            "batch_size_bucket": 1,
            "seq_len_bucket": 1024,
            "head_dim": 128,
            "num_heads": 32,
            "num_kv_heads": 8,
        },
        {
            "operation": "attention.causal",
            "dtype": torch.float16,
            "batch_size_bucket": 8,
            "seq_len_bucket": 2048,
            "head_dim": 128,
            "num_heads": 32,
            "num_kv_heads": 8,
        },
        {
            "operation": "attention.causal",
            "dtype": torch.bfloat16,
            "batch_size_bucket": 1,
            "seq_len_bucket": 4096,
            "head_dim": 128,
            "num_heads": 32,
            "num_kv_heads": 8,
        },
    ]


@pytest.fixture
def sample_model_config() -> dict:
    """Sample model config for testing manifest generation."""
    return {
        "model_type": "llama",
        "hidden_size": 4096,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "max_position_embeddings": 8192,
        "vocab_size": 32000,
        "dtype": "float16",
    }
