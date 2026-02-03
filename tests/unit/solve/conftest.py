"""Pytest fixtures for solve tests."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from typing import Any


@pytest.fixture
def mock_hardware_context() -> MagicMock:
    """Create mock hardware context."""
    ctx = MagicMock()
    ctx.platform = "cuda"
    ctx.sm_version = 80
    ctx.compute_capability = (8, 0)
    ctx.device_name = "NVIDIA A100"
    ctx.total_memory = 40 * 1024 * 1024 * 1024  # 40GB
    return ctx


@pytest.fixture
def mock_kernel() -> MagicMock:
    """Create mock kernel specification."""
    kernel = MagicMock()
    kernel.kernel_id = "flash_attn_v2"
    kernel.operation = "attention"
    kernel.backend = "flashinfer"
    kernel.priority = 100
    return kernel


@pytest.fixture
def sample_shape_buckets() -> list[dict[str, Any]]:
    """Create sample shape buckets for testing."""
    return [
        {
            "batch_size": [1, 2, 4, 8],
            "seq_len": [128, 256, 512, 1024, 2048],
            "num_heads": [8, 16, 32],
            "head_dim": [64, 128],
        },
        {
            "batch_size": [16, 32, 64],
            "seq_len": [4096, 8192],
            "num_heads": [32, 64],
            "head_dim": [128],
        },
    ]


@pytest.fixture
def sample_dispatch_rules() -> list[dict[str, Any]]:
    """Create sample dispatch rules."""
    return [
        {
            "conditions": {
                "batch_size": {"min": 1, "max": 8},
                "seq_len": {"min": 1, "max": 2048},
            },
            "kernel_id": "flash_attn_v2",
            "priority": 100,
        },
        {
            "conditions": {
                "batch_size": {"min": 1, "max": 64},
                "seq_len": {"min": 2049, "max": 8192},
            },
            "kernel_id": "flash_attn_v2_long",
            "priority": 90,
        },
        {
            "conditions": {},  # Default fallback
            "kernel_id": "torch_sdpa",
            "priority": 50,
        },
    ]
