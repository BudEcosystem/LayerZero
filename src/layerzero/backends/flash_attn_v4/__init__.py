"""FlashAttention 4 backend for Blackwell GPUs.

FA4 provides optimized attention kernels for NVIDIA Blackwell (SM 10.0+)
using tcgen05.mma tensor core operations.
"""
from __future__ import annotations

from layerzero.backends.flash_attn_v4.adapter import FlashAttnV4Adapter
from layerzero.backends.flash_attn_v4.availability import (
    check_cuda_version_for_fa4,
    detect_fa4_version,
    is_fa4_available,
    FA4_MIN_VERSION,
    FA4_MIN_CUDA_VERSION,
)
from layerzero.backends.flash_attn_v4.constraints import (
    check_fa4_fp8_constraints,
    FA4FP8ConstraintViolation,
)
from layerzero.backends.flash_attn_v4.hardware import (
    check_fa4_memory_requirements,
    has_tcgen05_support,
    is_fa4_compatible_sm,
    FA4MemoryRequirement,
)
from layerzero.backends.flash_attn_v4.specs import (
    FA4_KERNEL_SPEC,
    create_fa4_kernel_spec,
)

__all__ = [
    # Adapter
    "FlashAttnV4Adapter",
    # Availability
    "check_cuda_version_for_fa4",
    "detect_fa4_version",
    "is_fa4_available",
    "FA4_MIN_VERSION",
    "FA4_MIN_CUDA_VERSION",
    # Constraints
    "check_fa4_fp8_constraints",
    "FA4FP8ConstraintViolation",
    # Hardware
    "check_fa4_memory_requirements",
    "has_tcgen05_support",
    "is_fa4_compatible_sm",
    "FA4MemoryRequirement",
    # Specs
    "FA4_KERNEL_SPEC",
    "create_fa4_kernel_spec",
]
