"""Pytest fixtures for isolation tests."""
from __future__ import annotations

import pytest
from typing import Any
from unittest.mock import MagicMock


@pytest.fixture
def mock_backend_config() -> dict[str, Any]:
    """Create mock backend configuration."""
    return {
        "backend_id": "test_backend",
        "module": "layerzero.backends.test",
        "requires_isolation": False,
        "abi_version": "1.0",
    }


@pytest.fixture
def mock_isolated_backend_config() -> dict[str, Any]:
    """Create mock isolated backend configuration."""
    return {
        "backend_id": "isolated_backend",
        "module": "layerzero.backends.isolated",
        "requires_isolation": True,
        "abi_version": "2.0",
    }


@pytest.fixture
def mock_request() -> dict[str, Any]:
    """Create mock backend request."""
    return {
        "operation": "attention",
        "inputs": {
            "query": "tensor_ref_1",
            "key": "tensor_ref_2",
            "value": "tensor_ref_3",
        },
        "config": {
            "is_causal": True,
            "scale": None,
        },
    }


@pytest.fixture
def mock_response() -> dict[str, Any]:
    """Create mock backend response."""
    return {
        "status": "success",
        "output": "tensor_ref_4",
        "metadata": {
            "kernel_time_ms": 1.5,
            "memory_used_mb": 128.0,
        },
    }
