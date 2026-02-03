"""
LayerZero Core Module

Core utilities and validation logic.
"""
from layerzero.core.validation import (
    ValidationResult,
    validate_head_dim,
    validate_cuda_block_limits,
    validate_layout,
    validate_dtype,
    detect_layout,
    is_layout_ambiguous,
    validate_attention_inputs,
    can_convert_dtype,
    get_fallback_dtype,
    SUPPORTED_HEAD_DIMS,
    MAX_CUDA_BLOCKS,
)

__all__ = [
    "ValidationResult",
    "validate_head_dim",
    "validate_cuda_block_limits",
    "validate_layout",
    "validate_dtype",
    "detect_layout",
    "is_layout_ambiguous",
    "validate_attention_inputs",
    "can_convert_dtype",
    "get_fallback_dtype",
    "SUPPORTED_HEAD_DIMS",
    "MAX_CUDA_BLOCKS",
]
