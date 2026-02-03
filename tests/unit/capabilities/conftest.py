"""Pytest fixtures for capabilities tests."""
from __future__ import annotations

import pytest
from typing import Any


@pytest.fixture
def valid_capabilities_v1() -> dict[str, Any]:
    """Create valid capabilities descriptor v1."""
    return {
        "schema_version": "1.0",
        "kernel_id": "flash_attn_v2",
        "operation": "attention",
        "backend": "flashinfer",
        "constraints": {
            "head_dim": {"min": 32, "max": 256, "valid": [32, 64, 128, 256]},
            "batch_size": {"min": 1, "max": 128},
            "seq_len": {"min": 1, "max": 32768},
            "num_heads": {"min": 1, "max": 128},
        },
        "dtypes": ["float16", "bfloat16"],
        "platforms": ["cuda"],
        "min_sm_version": 80,
    }


@pytest.fixture
def valid_capabilities_v2() -> dict[str, Any]:
    """Create valid capabilities descriptor v2 (future version)."""
    return {
        "schema_version": "2.0",
        "kernel_id": "future_kernel",
        "operation": "attention",
        "backend": "future_backend",
        "constraints": {},
        "new_v2_field": "some_value",
    }


@pytest.fixture
def minimal_valid_descriptor() -> dict[str, Any]:
    """Create minimal valid descriptor."""
    return {
        "schema_version": "1.0",
        "kernel_id": "test_kernel",
        "operation": "attention",
    }


@pytest.fixture
def invalid_missing_required() -> dict[str, Any]:
    """Create descriptor missing required field."""
    return {
        "schema_version": "1.0",
        # Missing kernel_id
        "operation": "attention",
    }


@pytest.fixture
def invalid_constraint() -> dict[str, Any]:
    """Create descriptor with invalid constraint."""
    return {
        "schema_version": "1.0",
        "kernel_id": "test_kernel",
        "operation": "attention",
        "constraints": {
            "head_dim": {"min": 256, "max": 32},  # min > max is invalid
        },
    }
