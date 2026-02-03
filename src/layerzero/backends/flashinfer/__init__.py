"""
LayerZero FlashInfer Backend

FlashInfer adapter for high-performance attention with paged KV cache support.
MLSys 2025 Best Paper winner.
"""
from layerzero.backends.flashinfer.adapter import (
    FlashInferPrefillAdapter,
    FlashInferDecodeAdapter,
    FlashInferPagedAdapter,
)
from layerzero.backends.flashinfer.version import (
    detect_flashinfer_version,
    get_flashinfer_backend_info,
    is_flashinfer_available,
    is_jit_cache_available,
)
from layerzero.backends.flashinfer.layout import (
    bshd_to_nhd,
    bhsd_to_nhd,
    nhd_to_bshd,
    nhd_to_bhsd,
    hnd_to_nhd,
    nhd_to_hnd,
    convert_layout_for_flashinfer,
)
from layerzero.backends.flashinfer.constraints import (
    check_flashinfer_constraints,
    check_sm_version,
    check_head_dim,
    check_dtype,
    check_gqa_compatibility,
    FLASHINFER_MIN_SM,
    FLASHINFER_SUPPORTED_HEAD_DIMS,
)

__all__ = [
    # Adapters
    "FlashInferPrefillAdapter",
    "FlashInferDecodeAdapter",
    "FlashInferPagedAdapter",
    # Version detection
    "detect_flashinfer_version",
    "get_flashinfer_backend_info",
    "is_flashinfer_available",
    "is_jit_cache_available",
    # Layout conversion
    "bshd_to_nhd",
    "bhsd_to_nhd",
    "nhd_to_bshd",
    "nhd_to_bhsd",
    "hnd_to_nhd",
    "nhd_to_hnd",
    "convert_layout_for_flashinfer",
    # Constraints
    "check_flashinfer_constraints",
    "check_sm_version",
    "check_head_dim",
    "check_dtype",
    "check_gqa_compatibility",
    "FLASHINFER_MIN_SM",
    "FLASHINFER_SUPPORTED_HEAD_DIMS",
]
