"""
LayerZero xFormers Backend

Adapter for xFormers memory_efficient_attention from Facebook Research.
"""
from __future__ import annotations

from layerzero.backends.xformers.adapter import XFormersAdapter
from layerzero.backends.xformers.version import (
    detect_xformers_version,
    get_available_backends,
    get_xformers_backend_info,
    is_xformers_available,
)
from layerzero.backends.xformers.constraints import (
    check_dtype,
    check_head_dim,
    check_sm_version,
    check_stride,
    check_xformers_constraints,
    XFORMERS_HEAD_DIM_MULTIPLE,
    XFORMERS_MAX_HEAD_DIM,
    XFORMERS_MIN_HEAD_DIM,
    XFORMERS_MIN_SM,
)
from layerzero.backends.xformers.bias import (
    check_bias_broadcast,
    check_bias_device,
    expand_attn_bias,
    validate_attn_bias,
)

__all__ = [
    # Adapter
    "XFormersAdapter",
    # Version detection
    "is_xformers_available",
    "detect_xformers_version",
    "get_xformers_backend_info",
    "get_available_backends",
    # Constraints
    "check_sm_version",
    "check_head_dim",
    "check_dtype",
    "check_stride",
    "check_xformers_constraints",
    "XFORMERS_MIN_SM",
    "XFORMERS_MIN_HEAD_DIM",
    "XFORMERS_MAX_HEAD_DIM",
    "XFORMERS_HEAD_DIM_MULTIPLE",
    # Bias handling
    "check_bias_device",
    "check_bias_broadcast",
    "validate_attn_bias",
    "expand_attn_bias",
]
