"""Pytest fixtures for planner tests."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock
from typing import Any


@pytest.fixture
def attention_op() -> dict[str, Any]:
    """Create attention operation specification."""
    return {
        "op_type": "attention",
        "input_layout": "BSHD",
        "output_layout": "BSHD",
        "input_dtype": "float16",
        "output_dtype": "float16",
        "config": {
            "is_causal": True,
            "scale": None,
        },
    }


@pytest.fixture
def layernorm_op() -> dict[str, Any]:
    """Create layernorm operation specification."""
    return {
        "op_type": "layernorm",
        "input_layout": "BSHD",
        "output_layout": "BSHD",
        "input_dtype": "float16",
        "output_dtype": "float16",
        "config": {
            "normalized_shape": [512],
            "eps": 1e-5,
        },
    }


@pytest.fixture
def mlp_op() -> dict[str, Any]:
    """Create MLP operation specification."""
    return {
        "op_type": "mlp",
        "input_layout": "BSH",
        "output_layout": "BSH",
        "input_dtype": "float16",
        "output_dtype": "float16",
        "config": {
            "hidden_dim": 2048,
            "activation": "gelu",
        },
    }


@pytest.fixture
def transformer_block(attention_op, layernorm_op, mlp_op) -> list[dict[str, Any]]:
    """Create transformer block (attention + norm + MLP)."""
    return [attention_op, layernorm_op, mlp_op]


@pytest.fixture
def mock_kernel_registry() -> MagicMock:
    """Create mock kernel registry."""
    registry = MagicMock()

    # Define available kernels for each operation
    registry.get_kernels.side_effect = lambda op_type: {
        "attention": [
            MagicMock(kernel_id="flash_attn", input_layout="BSHD", output_layout="BSHD", latency_ms=1.0),
            MagicMock(kernel_id="sdpa", input_layout="BSHD", output_layout="BSHD", latency_ms=2.0),
        ],
        "layernorm": [
            MagicMock(kernel_id="triton_ln", input_layout="BSHD", output_layout="BSHD", latency_ms=0.2),
        ],
        "mlp": [
            MagicMock(kernel_id="fused_mlp", input_layout="BSH", output_layout="BSH", latency_ms=1.5),
        ],
    }.get(op_type, [])

    return registry
