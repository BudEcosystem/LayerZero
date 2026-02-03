"""FlashAttention 4 constraints for FP8 support.

This module handles FA4-specific constraint checking,
particularly for FP8 dtype support on Blackwell.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# FA4 FP8 format support
FA4_SUPPORTED_FP8_FORMATS = frozenset([
    "fp8_e4m3",
    "fp8_e5m2",
])


@dataclass(frozen=True)
class FA4FP8ConstraintViolation:
    """FA4 FP8 constraint violation.

    Attributes:
        reason: Human-readable reason.
        code: Machine-readable code.
    """

    reason: str
    code: str


def check_fa4_fp8_constraints(
    sm_version: tuple[int, int],
    dtype_str: str,
    head_dim: int,
) -> list[FA4FP8ConstraintViolation]:
    """Check FA4 FP8 constraints.

    FA4 supports FP8 (E4M3, E5M2) on Blackwell GPUs.

    Args:
        sm_version: GPU SM version as (major, minor).
        dtype_str: Data type string (e.g., "fp8_e4m3").
        head_dim: Head dimension.

    Returns:
        List of constraint violations (empty if all pass).
    """
    violations: list[FA4FP8ConstraintViolation] = []

    major, minor = sm_version

    # SM version check for FP8
    if major < 10:
        violations.append(FA4FP8ConstraintViolation(
            reason=f"FA4 FP8 requires SM 10.0+, got SM {major}.{minor}",
            code="SM_TOO_OLD_FOR_FP8",
        ))
        return violations  # No point checking further

    # FP8 format check
    if dtype_str not in FA4_SUPPORTED_FP8_FORMATS:
        violations.append(FA4FP8ConstraintViolation(
            reason=f"FA4 FP8 requires {FA4_SUPPORTED_FP8_FORMATS}, got {dtype_str}",
            code="FP8_FORMAT_UNSUPPORTED",
        ))

    # Head dim must be multiple of 16 for FP8
    if head_dim % 16 != 0:
        violations.append(FA4FP8ConstraintViolation(
            reason=f"FA4 FP8 requires head_dim multiple of 16, got {head_dim}",
            code="HEAD_DIM_ALIGNMENT_FP8",
        ))

    # Head dim max is 128 for FP8 (more restrictive)
    if head_dim > 128:
        violations.append(FA4FP8ConstraintViolation(
            reason=f"FA4 FP8 requires head_dim <= 128, got {head_dim}",
            code="HEAD_DIM_TOO_LARGE_FP8",
        ))

    return violations
