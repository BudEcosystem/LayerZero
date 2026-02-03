"""
LayerZero Liger Backend

Adapters for Liger Triton kernels (RMSNorm, RoPE, SwiGLU, CrossEntropy).
"""
from __future__ import annotations

from layerzero.backends.liger.version import (
    check_triton_compatibility,
    detect_liger_version,
    detect_triton_version,
    get_available_kernels,
    get_liger_info,
    is_liger_available,
    is_triton_available,
)
from layerzero.backends.liger.constraints import (
    check_dtype,
    check_liger_constraints,
    check_platform,
    check_triton_version,
    LIGER_MIN_TRITON_VERSION,
    LIGER_SUPPORTED_DTYPES,
    LIGER_SUPPORTED_PLATFORMS,
)
from layerzero.backends.liger.rms_norm import LigerRMSNormAdapter
from layerzero.backends.liger.rope import LigerRoPEAdapter
from layerzero.backends.liger.swiglu import LigerSwiGLUAdapter
from layerzero.backends.liger.cross_entropy import LigerCrossEntropyAdapter

__all__ = [
    # Version detection
    "is_liger_available",
    "detect_liger_version",
    "is_triton_available",
    "detect_triton_version",
    "check_triton_compatibility",
    "get_available_kernels",
    "get_liger_info",
    # Constraints
    "check_triton_version",
    "check_dtype",
    "check_platform",
    "check_liger_constraints",
    "LIGER_MIN_TRITON_VERSION",
    "LIGER_SUPPORTED_DTYPES",
    "LIGER_SUPPORTED_PLATFORMS",
    # Adapters
    "LigerRMSNormAdapter",
    "LigerRoPEAdapter",
    "LigerSwiGLUAdapter",
    "LigerCrossEntropyAdapter",
]
