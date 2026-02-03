"""
Constraint Validation

Provides validation logic for attention kernel constraints.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TYPE_CHECKING

import torch

from layerzero.enums import Layout
from layerzero import reasons as ReasonCodes

if TYPE_CHECKING:
    pass


# Supported head dimensions for most backends
SUPPORTED_HEAD_DIMS = {32, 64, 128, 256}

# Extended head dims for specific backends
EXTENDED_HEAD_DIMS = {32, 64, 128, 256, 320}

# Maximum CUDA blocks (grid dimension limit)
MAX_CUDA_BLOCKS = 65535

# Maximum grid dimension (x, y, z)
MAX_GRID_DIM_X = 2147483647
MAX_GRID_DIM_Y = 65535
MAX_GRID_DIM_Z = 65535

# Supported dtypes for attention
SUPPORTED_DTYPES = {
    torch.float16,
    torch.bfloat16,
    torch.float32,
}


@dataclass
class ValidationResult:
    """Result of a validation check.

    Attributes:
        valid: Whether the validation passed.
        reason: Reason code if validation failed.
        message: Optional detailed message.
    """

    valid: bool
    reason: str | None = None
    message: str = ""


def validate_head_dim(
    head_dim: int,
    backend: str | None = None,
) -> ValidationResult:
    """Validate head dimension.

    Args:
        head_dim: Head dimension to validate.
        backend: Optional backend name for specific validation.

    Returns:
        ValidationResult indicating if head_dim is valid.
    """
    # Check alignment (must be multiple of 8 for efficiency)
    if head_dim % 8 != 0:
        return ValidationResult(
            valid=False,
            reason=ReasonCodes.HEAD_DIM_ALIGNMENT,
            message=f"head_dim={head_dim} is not a multiple of 8",
        )

    # Check if in supported set
    supported = EXTENDED_HEAD_DIMS if backend in ("flash_infer", "triton") else SUPPORTED_HEAD_DIMS

    if head_dim not in supported:
        # Not in standard set, but may still work
        # Only reject if too large
        if head_dim > 320:
            return ValidationResult(
                valid=False,
                reason=ReasonCodes.HEAD_DIM_INVALID,
                message=f"head_dim={head_dim} exceeds maximum supported (320)",
            )

        # Accept non-standard but aligned values with warning
        # (could add a warnings field to ValidationResult)

    return ValidationResult(valid=True)


def validate_cuda_block_limits(
    batch: int,
    heads: int,
    seq_len: int | None = None,
) -> ValidationResult:
    """Validate CUDA block/grid limits.

    Args:
        batch: Batch size.
        heads: Number of attention heads.
        seq_len: Optional sequence length for grid validation.

    Returns:
        ValidationResult indicating if within limits.
    """
    # Check batch * heads limit (grid.x dimension)
    total_blocks = batch * heads

    if total_blocks > MAX_CUDA_BLOCKS:
        return ValidationResult(
            valid=False,
            reason=ReasonCodes.CUDA_BLOCK_LIMIT_EXCEEDED,
            message=f"batch * heads = {total_blocks} exceeds limit of {MAX_CUDA_BLOCKS}",
        )

    # Check sequence length grid dimension
    if seq_len is not None:
        if seq_len > MAX_GRID_DIM_X:
            return ValidationResult(
                valid=False,
                reason=ReasonCodes.CUDA_BLOCK_LIMIT_EXCEEDED,
                message=f"seq_len={seq_len} exceeds maximum grid dimension",
            )

    return ValidationResult(valid=True)


def detect_layout(
    tensor: torch.Tensor,
    expected_heads: int,
    expected_dim: int,
) -> Layout:
    """Detect tensor layout (BSHD or BHSD).

    Args:
        tensor: 4D tensor to analyze.
        expected_heads: Expected number of heads.
        expected_dim: Expected head dimension.

    Returns:
        Detected Layout enum value.
    """
    if tensor.dim() != 4:
        # Non-4D tensors default to BHSD for consistency
        return Layout.BHSD

    shape = tensor.shape
    # shape = [dim0, dim1, dim2, dim3]

    # If dim3 matches expected_dim, check dim1/dim2 for heads/seq
    if shape[3] == expected_dim:
        if shape[1] == expected_heads:
            return Layout.BHSD  # [B, H, S, D]
        elif shape[2] == expected_heads:
            return Layout.BSHD  # [B, S, H, D]

    # Fallback: assume BHSD
    return Layout.BHSD


def is_layout_ambiguous(
    tensor: torch.Tensor,
    expected_heads: int,
    expected_dim: int,
) -> bool:
    """Check if layout is ambiguous.

    Layout is ambiguous when seq_len == num_heads.

    Args:
        tensor: 4D tensor to check.
        expected_heads: Expected number of heads.
        expected_dim: Expected head dimension.

    Returns:
        True if layout cannot be determined unambiguously.
    """
    if tensor.dim() != 4:
        return False

    shape = tensor.shape

    # Check if dim1 == dim2 (ambiguous between BSHD and BHSD)
    return shape[1] == shape[2]


def validate_layout(
    tensor: torch.Tensor,
    expected_layout: Layout,
) -> ValidationResult:
    """Validate tensor layout.

    Args:
        tensor: Tensor to validate.
        expected_layout: Expected layout.

    Returns:
        ValidationResult.
    """
    if tensor.dim() not in (3, 4):
        return ValidationResult(
            valid=False,
            reason=ReasonCodes.LAYOUT_UNSUPPORTED,
            message=f"Expected 3D or 4D tensor, got {tensor.dim()}D",
        )

    return ValidationResult(valid=True)


def validate_dtype(
    dtype: torch.dtype,
    *other_dtypes: torch.dtype,
) -> ValidationResult:
    """Validate tensor dtype(s).

    Args:
        dtype: Primary dtype to check.
        *other_dtypes: Additional dtypes to check for consistency.

    Returns:
        ValidationResult.
    """
    # Check primary dtype
    if dtype not in SUPPORTED_DTYPES:
        return ValidationResult(
            valid=False,
            reason=ReasonCodes.DTYPE_UNSUPPORTED,
            message=f"dtype={dtype} is not supported for attention",
        )

    # Check other dtypes match (or are compatible)
    for other in other_dtypes:
        if other != dtype:
            # Allow mixed precision in some cases
            if {dtype, other} <= SUPPORTED_DTYPES:
                # Both supported but different - may need conversion
                return ValidationResult(
                    valid=True,
                    message=f"Mixed dtypes {dtype} and {other} - may require conversion",
                )
            else:
                return ValidationResult(
                    valid=False,
                    reason=ReasonCodes.DTYPE_UNSUPPORTED,
                    message=f"Incompatible dtypes: {dtype} and {other}",
                )

    return ValidationResult(valid=True)


def can_convert_dtype(from_dtype: torch.dtype, to_dtype: torch.dtype) -> bool:
    """Check if dtype conversion is possible.

    Args:
        from_dtype: Source dtype.
        to_dtype: Target dtype.

    Returns:
        True if conversion is possible.
    """
    # All floating point conversions are possible
    if from_dtype.is_floating_point and to_dtype.is_floating_point:
        return True

    # Integer to float is possible
    if not from_dtype.is_floating_point and to_dtype.is_floating_point:
        return True

    return False


def get_fallback_dtype(dtype: torch.dtype) -> torch.dtype:
    """Get fallback dtype when original is not supported.

    Args:
        dtype: Original dtype.

    Returns:
        Fallback dtype (may be same as input if supported).
    """
    if dtype in SUPPORTED_DTYPES:
        return dtype

    # BF16 fallback to FP16 on older hardware
    if dtype == torch.bfloat16:
        return torch.float16

    # Default fallback to FP32
    return torch.float32


def validate_attention_inputs(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
) -> ValidationResult:
    """Validate all attention inputs.

    Comprehensive validation of query, key, value tensors.

    Args:
        query: Query tensor.
        key: Key tensor.
        value: Value tensor.
        attn_mask: Optional attention mask.

    Returns:
        ValidationResult with overall status.
    """
    # Check dimensions
    if query.dim() not in (3, 4):
        return ValidationResult(
            valid=False,
            reason=ReasonCodes.LAYOUT_UNSUPPORTED,
            message=f"Query must be 3D or 4D, got {query.dim()}D",
        )

    if key.dim() != query.dim() or value.dim() != query.dim():
        return ValidationResult(
            valid=False,
            reason=ReasonCodes.LAYOUT_UNSUPPORTED,
            message="Query, key, value must have same number of dimensions",
        )

    # Check dtype
    dtype_result = validate_dtype(query.dtype, key.dtype, value.dtype)
    if not dtype_result.valid:
        return dtype_result

    # Check shape compatibility
    if query.dim() == 4:
        batch, heads, seq_q, head_dim = query.shape
        _, _, seq_k, _ = key.shape

        # Validate head dim
        head_dim_result = validate_head_dim(head_dim)
        if not head_dim_result.valid:
            return head_dim_result

        # Validate CUDA limits
        cuda_result = validate_cuda_block_limits(batch, heads, seq_q)
        if not cuda_result.valid:
            return cuda_result

    return ValidationResult(valid=True)
