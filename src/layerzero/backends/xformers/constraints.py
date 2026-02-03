"""
LayerZero xFormers Constraint Checking

Functions for validating xFormers-specific constraints.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.reasons import (
    DTYPE_UNSUPPORTED,
    HEAD_DIM_ALIGNMENT,
    HEAD_DIM_TOO_LARGE,
    HEAD_DIM_TOO_SMALL,
    SM_TOO_OLD,
    STRIDE_LAST_DIM,
    Reason,
    make_reason,
)

if TYPE_CHECKING:
    pass


# xFormers constants
XFORMERS_MIN_SM: tuple[int, int] = (7, 0)  # Volta
XFORMERS_MIN_HEAD_DIM: int = 8
XFORMERS_MAX_HEAD_DIM: int = 256
XFORMERS_HEAD_DIM_MULTIPLE: int = 8  # For optimal memory efficiency

# Supported dtypes
XFORMERS_SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset([
    torch.float16,
    torch.bfloat16,
])


def check_sm_version(sm_version: tuple[int, int]) -> list[Reason]:
    """Check if SM version meets xFormers requirements.

    xFormers requires SM 7.0+ (Volta) for memory efficient attention.

    Args:
        sm_version: GPU SM version as (major, minor) tuple.

    Returns:
        Empty list if valid, else list with SM_TOO_OLD reason.
    """
    reasons: list[Reason] = []

    if sm_version < XFORMERS_MIN_SM:
        reasons.append(make_reason(
            SM_TOO_OLD,
            f"xFormers requires SM {XFORMERS_MIN_SM}, got {sm_version}"
        ))

    return reasons


def check_head_dim(head_dim: int) -> list[Reason]:
    """Check if head dimension meets xFormers requirements.

    xFormers requires head_dim in range [8, 256] and multiple of 8
    for optimal memory efficiency.

    Args:
        head_dim: Head dimension value.

    Returns:
        Empty list if valid, else list of constraint failure reasons.
    """
    reasons: list[Reason] = []

    if head_dim < XFORMERS_MIN_HEAD_DIM:
        reasons.append(make_reason(
            HEAD_DIM_TOO_SMALL,
            f"head_dim {head_dim} < min {XFORMERS_MIN_HEAD_DIM}"
        ))
    elif head_dim > XFORMERS_MAX_HEAD_DIM:
        reasons.append(make_reason(
            HEAD_DIM_TOO_LARGE,
            f"head_dim {head_dim} > max {XFORMERS_MAX_HEAD_DIM}"
        ))
    elif head_dim % XFORMERS_HEAD_DIM_MULTIPLE != 0:
        reasons.append(make_reason(
            HEAD_DIM_ALIGNMENT,
            f"head_dim {head_dim} not multiple of {XFORMERS_HEAD_DIM_MULTIPLE}"
        ))

    return reasons


def check_dtype(dtype: torch.dtype) -> list[Reason]:
    """Check if dtype is supported by xFormers.

    xFormers only supports float16 and bfloat16. float32 is not
    supported for memory efficient attention.

    Args:
        dtype: Tensor dtype to check.

    Returns:
        Empty list if valid, else list with DTYPE_UNSUPPORTED reason.
    """
    reasons: list[Reason] = []

    if dtype not in XFORMERS_SUPPORTED_DTYPES:
        reasons.append(make_reason(
            DTYPE_UNSUPPORTED,
            f"xFormers does not support dtype {dtype}, "
            f"supported: {[str(d) for d in XFORMERS_SUPPORTED_DTYPES]}"
        ))

    return reasons


def check_stride(stride_last_dim: int) -> list[Reason]:
    """Check if last dimension stride meets xFormers requirements.

    xFormers requires contiguous memory layout with stride[-1] == 1.

    Args:
        stride_last_dim: Stride of the last dimension.

    Returns:
        Empty list if valid, else list with STRIDE_LAST_DIM reason.
    """
    reasons: list[Reason] = []

    if stride_last_dim != 1:
        reasons.append(make_reason(
            STRIDE_LAST_DIM,
            f"xFormers requires stride[-1]=1, got {stride_last_dim}"
        ))

    return reasons


def check_xformers_constraints(
    sm_version: tuple[int, int] | None = None,
    head_dim: int | None = None,
    dtype: torch.dtype | None = None,
    stride_last_dim: int | None = None,
) -> list[Reason]:
    """Check all xFormers constraints at once.

    Convenience function to validate multiple constraints. Only checks
    constraints for which values are provided (non-None).

    Args:
        sm_version: GPU SM version as (major, minor) tuple.
        head_dim: Head dimension value.
        dtype: Tensor dtype.
        stride_last_dim: Stride of the last dimension.

    Returns:
        List of all constraint failure reasons (empty if all valid).
    """
    reasons: list[Reason] = []

    if sm_version is not None:
        reasons.extend(check_sm_version(sm_version))

    if head_dim is not None:
        reasons.extend(check_head_dim(head_dim))

    if dtype is not None:
        reasons.extend(check_dtype(dtype))

    if stride_last_dim is not None:
        reasons.extend(check_stride(stride_last_dim))

    return reasons
