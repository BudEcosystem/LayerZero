"""
LayerZero SDPA Backend Constraints

Constraint checkers for different SDPA backends.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, unique
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch


@unique
class SDPABackendType(str, Enum):
    """SDPA backend types.

    Corresponds to torch.nn.attention.SDPBackend.
    """

    FLASH = "flash"
    EFFICIENT = "efficient"
    CUDNN = "cudnn"
    MATH = "math"


@dataclass(frozen=True, slots=True)
class ConstraintViolation:
    """Constraint violation description.

    Attributes:
        backend: Backend that cannot be used.
        reason: Human-readable reason.
        code: Machine-readable code.
    """

    backend: SDPABackendType
    reason: str
    code: str


def check_flash_constraints(
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
    is_causal: bool,
    has_mask: bool,
    is_contiguous: bool = True,
) -> list[ConstraintViolation]:
    """Check FlashAttention backend constraints.

    FlashAttention requires:
    - SM 8.0+ (Ampere or newer)
    - fp16 or bf16 dtype
    - head_dim <= 256
    - Cannot use attn_mask + is_causal together
    - Contiguous tensors (typically)

    Args:
        sm_version: GPU SM version as (major, minor).
        dtype: Input dtype.
        head_dim: Head dimension.
        is_causal: Whether causal masking is requested.
        has_mask: Whether attention mask is provided.
        is_contiguous: Whether tensors are contiguous.

    Returns:
        List of constraint violations (empty if all pass).
    """
    import torch

    violations: list[ConstraintViolation] = []

    # SM version check (SM 8.0+)
    if sm_version < (8, 0):
        violations.append(ConstraintViolation(
            backend=SDPABackendType.FLASH,
            reason=f"FlashAttention requires SM 8.0+, got SM {sm_version[0]}.{sm_version[1]}",
            code="SM_TOO_OLD",
        ))

    # Dtype check (fp16 or bf16 only)
    if dtype not in (torch.float16, torch.bfloat16):
        violations.append(ConstraintViolation(
            backend=SDPABackendType.FLASH,
            reason=f"FlashAttention requires fp16 or bf16, got {dtype}",
            code="DTYPE_UNSUPPORTED",
        ))

    # Head dim check
    if head_dim > 256:
        violations.append(ConstraintViolation(
            backend=SDPABackendType.FLASH,
            reason=f"FlashAttention requires head_dim <= 256, got {head_dim}",
            code="HEAD_DIM_TOO_LARGE",
        ))

    # Mask + causal check
    if has_mask and is_causal:
        violations.append(ConstraintViolation(
            backend=SDPABackendType.FLASH,
            reason="FlashAttention cannot use attn_mask with is_causal=True",
            code="MASK_PLUS_CAUSAL",
        ))

    # Contiguity check
    if not is_contiguous:
        violations.append(ConstraintViolation(
            backend=SDPABackendType.FLASH,
            reason="FlashAttention requires contiguous tensors",
            code="NOT_CONTIGUOUS",
        ))

    return violations


def check_efficient_constraints(
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
    is_causal: bool,
    has_mask: bool,
    is_contiguous: bool = True,
) -> list[ConstraintViolation]:
    """Check Memory Efficient Attention backend constraints.

    Efficient Attention requires:
    - SM 5.0+ (Maxwell or newer)
    - Any float dtype (fp16, bf16, fp32)
    - Cannot use attn_mask + is_causal together

    Args:
        sm_version: GPU SM version as (major, minor).
        dtype: Input dtype.
        head_dim: Head dimension.
        is_causal: Whether causal masking is requested.
        has_mask: Whether attention mask is provided.
        is_contiguous: Whether tensors are contiguous.

    Returns:
        List of constraint violations (empty if all pass).
    """
    import torch

    violations: list[ConstraintViolation] = []

    # SM version check (SM 5.0+)
    if sm_version < (5, 0):
        violations.append(ConstraintViolation(
            backend=SDPABackendType.EFFICIENT,
            reason=f"Memory Efficient requires SM 5.0+, got SM {sm_version[0]}.{sm_version[1]}",
            code="SM_TOO_OLD",
        ))

    # Dtype check (more permissive)
    if dtype not in (torch.float16, torch.bfloat16, torch.float32):
        violations.append(ConstraintViolation(
            backend=SDPABackendType.EFFICIENT,
            reason=f"Memory Efficient requires float dtype, got {dtype}",
            code="DTYPE_UNSUPPORTED",
        ))

    # Mask + causal check
    if has_mask and is_causal:
        violations.append(ConstraintViolation(
            backend=SDPABackendType.EFFICIENT,
            reason="Memory Efficient cannot use attn_mask with is_causal=True",
            code="MASK_PLUS_CAUSAL",
        ))

    return violations


def check_cudnn_constraints(
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
    is_causal: bool,
    has_mask: bool,
    is_contiguous: bool = True,
) -> list[ConstraintViolation]:
    """Check cuDNN Attention backend constraints.

    cuDNN Attention requires:
    - SM 8.0+ (Ampere or newer)
    - fp16 or bf16 dtype
    - head_dim <= 128
    - Contiguous tensors

    Args:
        sm_version: GPU SM version as (major, minor).
        dtype: Input dtype.
        head_dim: Head dimension.
        is_causal: Whether causal masking is requested.
        has_mask: Whether attention mask is provided.
        is_contiguous: Whether tensors are contiguous.

    Returns:
        List of constraint violations (empty if all pass).
    """
    import torch

    violations: list[ConstraintViolation] = []

    # SM version check (SM 8.0+)
    if sm_version < (8, 0):
        violations.append(ConstraintViolation(
            backend=SDPABackendType.CUDNN,
            reason=f"cuDNN Attention requires SM 8.0+, got SM {sm_version[0]}.{sm_version[1]}",
            code="SM_TOO_OLD",
        ))

    # Dtype check (fp16 or bf16 only)
    if dtype not in (torch.float16, torch.bfloat16):
        violations.append(ConstraintViolation(
            backend=SDPABackendType.CUDNN,
            reason=f"cuDNN Attention requires fp16 or bf16, got {dtype}",
            code="DTYPE_UNSUPPORTED",
        ))

    # Head dim check (stricter than Flash)
    if head_dim > 128:
        violations.append(ConstraintViolation(
            backend=SDPABackendType.CUDNN,
            reason=f"cuDNN Attention requires head_dim <= 128, got {head_dim}",
            code="HEAD_DIM_TOO_LARGE",
        ))

    # Contiguity check
    if not is_contiguous:
        violations.append(ConstraintViolation(
            backend=SDPABackendType.CUDNN,
            reason="cuDNN Attention requires contiguous tensors",
            code="NOT_CONTIGUOUS",
        ))

    return violations


def check_math_constraints(
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
    is_causal: bool,
    has_mask: bool,
    is_contiguous: bool = True,
) -> list[ConstraintViolation]:
    """Check Math (reference) backend constraints.

    Math backend has no constraints - always works.

    Args:
        sm_version: GPU SM version (unused).
        dtype: Input dtype (unused).
        head_dim: Head dimension (unused).
        is_causal: Whether causal masking is requested (unused).
        has_mask: Whether attention mask is provided (unused).
        is_contiguous: Whether tensors are contiguous (unused).

    Returns:
        Empty list (Math backend always works).
    """
    return []


_BACKEND_CHECKERS = {
    SDPABackendType.FLASH: check_flash_constraints,
    SDPABackendType.EFFICIENT: check_efficient_constraints,
    SDPABackendType.CUDNN: check_cudnn_constraints,
    SDPABackendType.MATH: check_math_constraints,
}


def get_available_backends(
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
    is_causal: bool,
    has_mask: bool,
    is_contiguous: bool = True,
) -> list[SDPABackendType]:
    """Get list of available SDPA backends for given constraints.

    Args:
        sm_version: GPU SM version as (major, minor).
        dtype: Input dtype.
        head_dim: Head dimension.
        is_causal: Whether causal masking is requested.
        has_mask: Whether attention mask is provided.
        is_contiguous: Whether tensors are contiguous.

    Returns:
        List of backends that can handle these constraints.
    """
    available: list[SDPABackendType] = []

    for backend, checker in _BACKEND_CHECKERS.items():
        violations = checker(
            sm_version=sm_version,
            dtype=dtype,
            head_dim=head_dim,
            is_causal=is_causal,
            has_mask=has_mask,
            is_contiguous=is_contiguous,
        )
        if not violations:
            available.append(backend)

    return available


def can_use_backend(
    backend: SDPABackendType,
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
    is_causal: bool,
    has_mask: bool,
    is_contiguous: bool = True,
) -> bool:
    """Check if a specific backend can be used.

    Args:
        backend: Backend to check.
        sm_version: GPU SM version as (major, minor).
        dtype: Input dtype.
        head_dim: Head dimension.
        is_causal: Whether causal masking is requested.
        has_mask: Whether attention mask is provided.
        is_contiguous: Whether tensors are contiguous.

    Returns:
        True if backend can be used, False otherwise.
    """
    checker = _BACKEND_CHECKERS.get(backend, check_math_constraints)
    violations = checker(
        sm_version=sm_version,
        dtype=dtype,
        head_dim=head_dim,
        is_causal=is_causal,
        has_mask=has_mask,
        is_contiguous=is_contiguous,
    )
    return len(violations) == 0
