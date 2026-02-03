"""
LayerZero FlashInfer Constraint Checking

Functions for validating FlashInfer kernel constraints.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.reasons import (
    SM_TOO_OLD,
    HEAD_DIM_TOO_LARGE,
    HEAD_DIM_TOO_SMALL,
    HEAD_DIM_ALIGNMENT,
    DTYPE_UNSUPPORTED,
    GQA_UNSUPPORTED,
    Reason,
    make_reason,
)

if TYPE_CHECKING:
    pass


# FlashInfer minimum SM version (Turing)
FLASHINFER_MIN_SM: tuple[int, int] = (7, 5)

# FlashInfer supported head dimensions (optimal performance)
# These are the common head dims that FlashInfer has optimized kernels for
FLASHINFER_SUPPORTED_HEAD_DIMS: frozenset[int] = frozenset([
    32, 40, 48, 56, 64, 72, 80, 88, 96,
    104, 112, 120, 128, 144, 160, 192, 224, 256,
])

# Min/max head dim
FLASHINFER_MIN_HEAD_DIM: int = 32
FLASHINFER_MAX_HEAD_DIM: int = 256

# Head dim must be multiple of 8
FLASHINFER_HEAD_DIM_MULTIPLE: int = 8

# Supported dtypes
FLASHINFER_SUPPORTED_DTYPES: frozenset[torch.dtype] = frozenset([
    torch.float16,
    torch.bfloat16,
])

# Quantized dtypes (requires allow_quantized flag)
FLASHINFER_QUANTIZED_DTYPES: frozenset[torch.dtype] = frozenset([
    torch.int8,
    # FP8 types added dynamically if torch supports them
])


def check_sm_version(sm_version: tuple[int, int]) -> list[Reason]:
    """Check if SM version is supported by FlashInfer.

    Args:
        sm_version: Tuple of (major, minor) SM version.

    Returns:
        List of rejection reasons (empty if valid).
    """
    reasons: list[Reason] = []

    if sm_version < FLASHINFER_MIN_SM:
        reasons.append(make_reason(
            SM_TOO_OLD,
            f"FlashInfer requires SM >= {FLASHINFER_MIN_SM}, got {sm_version}"
        ))

    return reasons


def check_head_dim(head_dim: int) -> list[Reason]:
    """Check if head dimension is valid for FlashInfer.

    Args:
        head_dim: Head dimension value.

    Returns:
        List of rejection reasons (empty if valid).
    """
    reasons: list[Reason] = []

    if head_dim < FLASHINFER_MIN_HEAD_DIM:
        reasons.append(make_reason(
            HEAD_DIM_TOO_SMALL,
            f"FlashInfer requires head_dim >= {FLASHINFER_MIN_HEAD_DIM}, got {head_dim}"
        ))
        return reasons

    if head_dim > FLASHINFER_MAX_HEAD_DIM:
        reasons.append(make_reason(
            HEAD_DIM_TOO_LARGE,
            f"FlashInfer requires head_dim <= {FLASHINFER_MAX_HEAD_DIM}, got {head_dim}"
        ))
        return reasons

    if head_dim % FLASHINFER_HEAD_DIM_MULTIPLE != 0:
        reasons.append(make_reason(
            HEAD_DIM_ALIGNMENT,
            f"FlashInfer requires head_dim multiple of {FLASHINFER_HEAD_DIM_MULTIPLE}, "
            f"got {head_dim}"
        ))

    return reasons


def check_dtype(
    dtype: torch.dtype,
    allow_quantized: bool = False,
) -> list[Reason]:
    """Check if dtype is supported by FlashInfer.

    Args:
        dtype: PyTorch dtype.
        allow_quantized: Whether to allow quantized dtypes (int8, etc.).

    Returns:
        List of rejection reasons (empty if valid).
    """
    reasons: list[Reason] = []

    if dtype in FLASHINFER_SUPPORTED_DTYPES:
        return reasons

    if allow_quantized and dtype in FLASHINFER_QUANTIZED_DTYPES:
        return reasons

    # Build list of supported dtypes for error message
    supported = set(FLASHINFER_SUPPORTED_DTYPES)
    if allow_quantized:
        supported.update(FLASHINFER_QUANTIZED_DTYPES)

    dtype_names = [str(d).replace("torch.", "") for d in supported]
    reasons.append(make_reason(
        DTYPE_UNSUPPORTED,
        f"FlashInfer does not support dtype {str(dtype).replace('torch.', '')}, "
        f"supported: {dtype_names}"
    ))

    return reasons


def check_gqa_compatibility(q_heads: int, kv_heads: int) -> list[Reason]:
    """Check if GQA configuration is valid for FlashInfer.

    Args:
        q_heads: Number of query heads.
        kv_heads: Number of key/value heads.

    Returns:
        List of rejection reasons (empty if valid).
    """
    reasons: list[Reason] = []

    if kv_heads > q_heads:
        reasons.append(make_reason(
            GQA_UNSUPPORTED,
            f"kv_heads ({kv_heads}) cannot be greater than q_heads ({q_heads})"
        ))
        return reasons

    if q_heads % kv_heads != 0:
        reasons.append(make_reason(
            GQA_UNSUPPORTED,
            f"q_heads ({q_heads}) must be divisible by kv_heads ({kv_heads})"
        ))

    return reasons


def check_flashinfer_constraints(
    sm_version: tuple[int, int],
    head_dim: int,
    dtype: torch.dtype,
    q_heads: int,
    kv_heads: int,
    allow_quantized: bool = False,
) -> list[Reason]:
    """Check all FlashInfer constraints.

    Args:
        sm_version: Tuple of (major, minor) SM version.
        head_dim: Head dimension.
        dtype: PyTorch dtype.
        q_heads: Number of query heads.
        kv_heads: Number of key/value heads.
        allow_quantized: Whether to allow quantized dtypes.

    Returns:
        List of all rejection reasons (empty if all valid).
    """
    reasons: list[Reason] = []

    reasons.extend(check_sm_version(sm_version))
    reasons.extend(check_head_dim(head_dim))
    reasons.extend(check_dtype(dtype, allow_quantized))
    reasons.extend(check_gqa_compatibility(q_heads, kv_heads))

    return reasons
