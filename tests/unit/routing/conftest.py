"""Pytest fixtures for routing tests."""
from __future__ import annotations

import pytest
from typing import Any

from layerzero.device import GPUGeneration


@pytest.fixture
def fa_kernels() -> list[dict[str, Any]]:
    """Create FlashAttention kernel configurations."""
    return [
        {
            "id": "fa2",
            "supported_generations": frozenset([
                GPUGeneration.AMPERE,
                GPUGeneration.ADA_LOVELACE,
            ]),
            "native_generation": GPUGeneration.AMPERE,
            "priority": 80,
        },
        {
            "id": "fa3",
            "supported_generations": frozenset([
                GPUGeneration.AMPERE,
                GPUGeneration.ADA_LOVELACE,
                GPUGeneration.HOPPER,
            ]),
            "native_generation": GPUGeneration.HOPPER,
            "priority": 90,
        },
        {
            "id": "fa4",
            "supported_generations": frozenset([
                GPUGeneration.BLACKWELL,
            ]),
            "native_generation": GPUGeneration.BLACKWELL,
            "priority": 100,
        },
    ]


@pytest.fixture
def universal_kernel() -> dict[str, Any]:
    """Create universal kernel (empty supported_generations)."""
    return {
        "id": "sdpa_universal",
        "supported_generations": frozenset(),
        "native_generation": None,
        "priority": 50,
    }


@pytest.fixture
def mixed_kernels(fa_kernels, universal_kernel) -> list[dict[str, Any]]:
    """Create mixed kernel list."""
    return fa_kernels + [universal_kernel]
