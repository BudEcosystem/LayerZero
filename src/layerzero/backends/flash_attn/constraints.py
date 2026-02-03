"""
LayerZero FlashAttention Constraints

Constraint checkers for different FlashAttention variants.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from layerzero.backends.flash_attn.version import FAVariant, select_fa_variant

if TYPE_CHECKING:
    import torch


@dataclass(frozen=True, slots=True)
class FAConstraintViolation:
    """FlashAttention constraint violation.

    Attributes:
        variant: FA variant that cannot be used.
        reason: Human-readable reason.
        code: Machine-readable code.
    """

    variant: FAVariant | None
    reason: str
    code: str


def check_fa2_constraints(
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
) -> list[FAConstraintViolation]:
    """Check FA2 (Ampere/Ada) constraints.

    FA2 requires:
    - SM 8.0-8.9 (Ampere or Ada)
    - fp16 or bf16 dtype
    - head_dim <= 256 and multiple of 8

    Args:
        sm_version: GPU SM version as (major, minor).
        dtype: Input dtype.
        head_dim: Head dimension.

    Returns:
        List of constraint violations (empty if all pass).
    """
    import torch

    violations: list[FAConstraintViolation] = []

    major, minor = sm_version

    # SM version check (8.0 <= SM <= 8.9)
    if major < 8:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA2,
            reason=f"FA2 requires SM 8.0+, got SM {major}.{minor}",
            code="SM_TOO_OLD",
        ))
    elif major > 8:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA2,
            reason=f"FA2 is for SM 8.x, got SM {major}.{minor}. Use FA3 or FA4.",
            code="SM_TOO_NEW",
        ))

    # Dtype check (fp16 or bf16 only)
    if dtype not in (torch.float16, torch.bfloat16):
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA2,
            reason=f"FA2 requires fp16 or bf16, got {dtype}",
            code="DTYPE_UNSUPPORTED",
        ))

    # Head dim check
    if head_dim > 256:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA2,
            reason=f"FA2 requires head_dim <= 256, got {head_dim}",
            code="HEAD_DIM_TOO_LARGE",
        ))

    # Head dim multiple check
    if head_dim % 8 != 0:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA2,
            reason=f"FA2 requires head_dim multiple of 8, got {head_dim}",
            code="HEAD_DIM_ALIGNMENT",
        ))

    return violations


def check_fa3_constraints(
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
) -> list[FAConstraintViolation]:
    """Check FA3 (Hopper) constraints.

    FA3 requires:
    - SM 9.0 (Hopper)
    - fp16 or bf16 dtype
    - head_dim <= 256 and multiple of 8

    Args:
        sm_version: GPU SM version as (major, minor).
        dtype: Input dtype.
        head_dim: Head dimension.

    Returns:
        List of constraint violations (empty if all pass).
    """
    import torch

    violations: list[FAConstraintViolation] = []

    major, minor = sm_version

    # SM version check (9.0 only)
    if major < 9:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA3,
            reason=f"FA3 requires SM 9.0+, got SM {major}.{minor}. Use FA2.",
            code="SM_TOO_OLD",
        ))
    elif major > 9:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA3,
            reason=f"FA3 is for SM 9.x, got SM {major}.{minor}. Use FA4.",
            code="SM_TOO_NEW",
        ))

    # Dtype check
    if dtype not in (torch.float16, torch.bfloat16):
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA3,
            reason=f"FA3 requires fp16 or bf16, got {dtype}",
            code="DTYPE_UNSUPPORTED",
        ))

    # Head dim check
    if head_dim > 256:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA3,
            reason=f"FA3 requires head_dim <= 256, got {head_dim}",
            code="HEAD_DIM_TOO_LARGE",
        ))

    # Head dim multiple check
    if head_dim % 8 != 0:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA3,
            reason=f"FA3 requires head_dim multiple of 8, got {head_dim}",
            code="HEAD_DIM_ALIGNMENT",
        ))

    return violations


def check_fa4_constraints(
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
) -> list[FAConstraintViolation]:
    """Check FA4 (Blackwell+) constraints.

    FA4 requires:
    - SM 10.0+ (Blackwell and beyond)
    - fp16 or bf16 dtype
    - head_dim <= 256 and multiple of 8

    Args:
        sm_version: GPU SM version as (major, minor).
        dtype: Input dtype.
        head_dim: Head dimension.

    Returns:
        List of constraint violations (empty if all pass).
    """
    import torch

    violations: list[FAConstraintViolation] = []

    major, minor = sm_version

    # SM version check (10.0+)
    if major < 10:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA4,
            reason=f"FA4 requires SM 10.0+, got SM {major}.{minor}",
            code="SM_TOO_OLD",
        ))

    # Dtype check
    if dtype not in (torch.float16, torch.bfloat16):
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA4,
            reason=f"FA4 requires fp16 or bf16, got {dtype}",
            code="DTYPE_UNSUPPORTED",
        ))

    # Head dim check
    if head_dim > 256:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA4,
            reason=f"FA4 requires head_dim <= 256, got {head_dim}",
            code="HEAD_DIM_TOO_LARGE",
        ))

    # Head dim multiple check
    if head_dim % 8 != 0:
        violations.append(FAConstraintViolation(
            variant=FAVariant.FA4,
            reason=f"FA4 requires head_dim multiple of 8, got {head_dim}",
            code="HEAD_DIM_ALIGNMENT",
        ))

    return violations


_VARIANT_CHECKERS = {
    FAVariant.FA2: check_fa2_constraints,
    FAVariant.FA3: check_fa3_constraints,
    FAVariant.FA4: check_fa4_constraints,
}


def check_fa_constraints(
    sm_version: tuple[int, int],
    dtype: "torch.dtype",
    head_dim: int,
) -> list[FAConstraintViolation]:
    """Check FlashAttention constraints, auto-selecting variant.

    Automatically selects the appropriate FA variant based on SM version,
    then checks constraints for that variant.

    Args:
        sm_version: GPU SM version as (major, minor).
        dtype: Input dtype.
        head_dim: Head dimension.

    Returns:
        List of constraint violations (empty if all pass).
    """
    variant = select_fa_variant(sm_version)

    if variant is None:
        return [FAConstraintViolation(
            variant=None,
            reason=f"FlashAttention requires SM 8.0+, got SM {sm_version[0]}.{sm_version[1]}",
            code="SM_UNSUPPORTED",
        )]

    checker = _VARIANT_CHECKERS[variant]
    return checker(sm_version=sm_version, dtype=dtype, head_dim=head_dim)
