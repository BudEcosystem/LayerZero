"""Pytest fixtures for quantization tests."""
from __future__ import annotations

import pytest
from typing import Any

from layerzero.device import GPUGeneration


@pytest.fixture
def ampere_device() -> dict[str, Any]:
    """Create Ampere device spec."""
    return {
        "gpu_generation": GPUGeneration.AMPERE,
        "sm_version": (8, 0),
        "supports_int8": True,
        "supports_fp8": False,
        "supports_fp4": False,
    }


@pytest.fixture
def hopper_device() -> dict[str, Any]:
    """Create Hopper device spec."""
    return {
        "gpu_generation": GPUGeneration.HOPPER,
        "sm_version": (9, 0),
        "supports_int8": True,
        "supports_fp8": True,
        "supports_fp4": False,
    }


@pytest.fixture
def blackwell_device() -> dict[str, Any]:
    """Create Blackwell device spec."""
    return {
        "gpu_generation": GPUGeneration.BLACKWELL,
        "sm_version": (10, 0),
        "supports_int8": True,
        "supports_fp8": True,
        "supports_fp4": True,
    }
