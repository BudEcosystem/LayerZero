"""Fixtures for speculative decoding tests."""
from __future__ import annotations

import pytest

from layerzero.device import GPUGeneration


@pytest.fixture
def draft_model_config() -> dict:
    """Small draft model configuration."""
    return {
        "model_name": "draft-llama-68m",
        "num_layers": 4,
        "num_heads": 4,
        "num_kv_heads": 4,
        "head_dim": 64,
        "hidden_size": 256,
        "vocab_size": 32000,
    }


@pytest.fixture
def target_model_config() -> dict:
    """Large target model configuration."""
    return {
        "model_name": "llama-70b",
        "num_layers": 80,
        "num_heads": 64,
        "num_kv_heads": 8,
        "head_dim": 128,
        "hidden_size": 8192,
        "vocab_size": 32000,
    }


@pytest.fixture
def hopper_device() -> GPUGeneration:
    """Hopper GPU generation."""
    return GPUGeneration.HOPPER


@pytest.fixture
def blackwell_device() -> GPUGeneration:
    """Blackwell GPU generation."""
    return GPUGeneration.BLACKWELL
