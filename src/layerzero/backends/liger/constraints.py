"""
LayerZero Liger Constraint Checking

Functions for validating Liger-specific constraints.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.enums import Platform
from layerzero.reasons import (
    BACKEND_VERSION_MISMATCH,
    DTYPE_UNSUPPORTED,
    PLATFORM_MISMATCH,
    Reason,
    make_reason,
)

if TYPE_CHECKING:
    pass


# Liger constants
LIGER_MIN_TRITON_VERSION: tuple[int, int, int] = (2, 0, 0)

# Supported dtypes (varies by kernel, but these are common)
LIGER_SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset([
    torch.float16,
    torch.bfloat16,
    torch.float32,
])

# Supported platforms (Triton works on CUDA and ROCm)
LIGER_SUPPORTED_PLATFORMS: frozenset[Platform] = frozenset([
    Platform.CUDA,
    Platform.ROCM,
])


def check_triton_version(triton_version: tuple[int, int, int]) -> list[Reason]:
    """Check if Triton version meets Liger requirements.

    Liger requires Triton >= 2.0.0.

    Args:
        triton_version: Triton version as (major, minor, patch) tuple.

    Returns:
        Empty list if valid, else list with BACKEND_VERSION_MISMATCH reason.
    """
    reasons: list[Reason] = []

    if triton_version < LIGER_MIN_TRITON_VERSION:
        reasons.append(make_reason(
            BACKEND_VERSION_MISMATCH,
            f"Liger requires Triton >= {LIGER_MIN_TRITON_VERSION}, "
            f"got {triton_version}"
        ))

    return reasons


def check_dtype(dtype: torch.dtype) -> list[Reason]:
    """Check if dtype is supported by Liger.

    Liger supports float16, bfloat16, and float32.

    Args:
        dtype: Tensor dtype to check.

    Returns:
        Empty list if valid, else list with DTYPE_UNSUPPORTED reason.
    """
    reasons: list[Reason] = []

    if dtype not in LIGER_SUPPORTED_DTYPES:
        reasons.append(make_reason(
            DTYPE_UNSUPPORTED,
            f"Liger does not support dtype {dtype}, "
            f"supported: {[str(d) for d in LIGER_SUPPORTED_DTYPES]}"
        ))

    return reasons


def check_platform(platform: Platform) -> list[Reason]:
    """Check if platform is supported by Liger.

    Liger uses Triton which supports CUDA and ROCm.
    CPU is not supported.

    Args:
        platform: Hardware platform to check.

    Returns:
        Empty list if valid, else list with PLATFORM_MISMATCH reason.
    """
    reasons: list[Reason] = []

    if platform not in LIGER_SUPPORTED_PLATFORMS:
        reasons.append(make_reason(
            PLATFORM_MISMATCH,
            f"Liger requires GPU (CUDA or ROCm), got {platform.value}"
        ))

    return reasons


def check_liger_constraints(
    triton_version: tuple[int, int, int] | None = None,
    dtype: torch.dtype | None = None,
    platform: Platform | None = None,
) -> list[Reason]:
    """Check all Liger constraints at once.

    Convenience function to validate multiple constraints. Only checks
    constraints for which values are provided (non-None).

    Args:
        triton_version: Triton version as (major, minor, patch) tuple.
        dtype: Tensor dtype.
        platform: Hardware platform.

    Returns:
        List of all constraint failure reasons (empty if all valid).
    """
    reasons: list[Reason] = []

    if triton_version is not None:
        reasons.extend(check_triton_version(triton_version))

    if dtype is not None:
        reasons.extend(check_dtype(dtype))

    if platform is not None:
        reasons.extend(check_platform(platform))

    return reasons
