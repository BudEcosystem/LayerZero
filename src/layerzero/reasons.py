"""
LayerZero Reason Codes

Structured reason codes for kernel selection filtering and rejection.
Each code has a unique string value for serialization and debugging.

This module provides:
- ReasonCategory: Categories of rejection reasons
- Reason: Frozen dataclass for structured rejection reasons
- Individual reason code constants (50+ codes)
- ALL_REASON_CODES: Mapping of all codes to their categories
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum, unique
from typing import Any


@unique
class ReasonCategory(str, Enum):
    """Categories of rejection reasons.

    Each category groups related reason codes for easier filtering
    and reporting. Categories are stored as lowercase strings.
    """

    HARDWARE = "hardware"
    DTYPE = "dtype"
    SHAPE = "shape"
    ATTENTION = "attention"
    CUDA = "cuda"
    BACKEND = "backend"
    MEMORY = "memory"
    TOKENIZER = "tokenizer"
    KV_CACHE = "kv_cache"
    QUANTIZATION = "quantization"
    DISTRIBUTED = "distributed"
    SPECULATIVE = "speculative"
    SCHEMA = "schema"


@dataclass(frozen=True, slots=True)
class Reason:
    """Structured rejection reason.

    Immutable (frozen) dataclass representing why a kernel was
    filtered out during selection. Hashable for use in sets/dicts.

    Attributes:
        code: Unique string identifier (SCREAMING_SNAKE_CASE)
        message: Human-readable description of the reason
        category: ReasonCategory for grouping
    """

    code: str
    message: str
    category: ReasonCategory

    def __str__(self) -> str:
        """Return formatted string representation."""
        return f"[{self.code}] {self.message}"

    def __repr__(self) -> str:
        """Return detailed representation."""
        return (
            f"Reason(code={self.code!r}, message={self.message!r}, "
            f"category={self.category!r})"
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON compatibility.

        Returns:
            Dict with 'code', 'message', 'category' keys.
        """
        return {
            "code": self.code,
            "message": self.message,
            "category": self.category.value,
        }

    def to_json(self) -> str:
        """Serialize to JSON string.

        Returns:
            JSON string representation.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Reason:
        """Deserialize from dictionary.

        Args:
            d: Dict with 'code', 'message', 'category' keys.

        Returns:
            New Reason instance.
        """
        return cls(
            code=d["code"],
            message=d["message"],
            category=ReasonCategory(d["category"]),
        )

    @classmethod
    def from_json(cls, json_str: str) -> Reason:
        """Deserialize from JSON string.

        Args:
            json_str: JSON string representation.

        Returns:
            New Reason instance.
        """
        return cls.from_dict(json.loads(json_str))


# =============================================================================
# Hardware Reason Codes
# =============================================================================

PLATFORM_MISMATCH = "PLATFORM_MISMATCH"
"""Kernel requires different platform (CUDA vs ROCm vs CPU)."""

SM_TOO_OLD = "SM_TOO_OLD"
"""GPU SM version is too old for this kernel."""

SM_TOO_NEW = "SM_TOO_NEW"
"""GPU SM version is too new (kernel not tested/built for it)."""

GPU_GENERATION_UNSUPPORTED = "GPU_GENERATION_UNSUPPORTED"
"""Kernel does not support this GPU generation."""

TENSOR_CORE_GEN_UNSUPPORTED = "TENSOR_CORE_GEN_UNSUPPORTED"
"""Kernel requires different tensor core generation."""

INSTRUCTION_SET_MISMATCH = "INSTRUCTION_SET_MISMATCH"
"""Required instruction set (AVX-512, AMX, etc.) not available."""

DEVICE_CAPABILITY_UNSUPPORTED = "DEVICE_CAPABILITY_UNSUPPORTED"
"""Device lacks required capability (bf16, fp8, etc.)."""

DRIVER_VERSION_UNSUPPORTED = "DRIVER_VERSION_UNSUPPORTED"
"""Driver version does not meet kernel requirements."""


# =============================================================================
# Data Type Reason Codes
# =============================================================================

DTYPE_UNSUPPORTED = "DTYPE_UNSUPPORTED"
"""Kernel does not support the input/output dtype."""

QUANT_FORMAT_UNSUPPORTED = "QUANT_FORMAT_UNSUPPORTED"
"""Kernel does not support the quantization format."""

QUANT_ACCURACY_THRESHOLD_EXCEEDED = "QUANT_ACCURACY_THRESHOLD_EXCEEDED"
"""Quantization would exceed accuracy threshold."""

REQUANTIZATION_REQUIRED = "REQUANTIZATION_REQUIRED"
"""Weights would need re-quantization for this kernel."""

MIXED_DTYPE_UNSUPPORTED = "MIXED_DTYPE_UNSUPPORTED"
"""Kernel does not support mixed dtype inputs."""


# =============================================================================
# Shape/Dimension Reason Codes
# =============================================================================

HEAD_DIM_INVALID = "HEAD_DIM_INVALID"
"""Head dimension is not supported by this kernel."""

HEAD_DIM_ALIGNMENT = "HEAD_DIM_ALIGNMENT"
"""Head dimension does not meet alignment requirements."""

HEAD_DIM_TOO_LARGE = "HEAD_DIM_TOO_LARGE"
"""Head dimension exceeds kernel maximum."""

HEAD_DIM_TOO_SMALL = "HEAD_DIM_TOO_SMALL"
"""Head dimension below kernel minimum."""

SEQ_TOO_LONG = "SEQ_TOO_LONG"
"""Sequence length exceeds kernel maximum."""

SEQ_TOO_SHORT = "SEQ_TOO_SHORT"
"""Sequence length below kernel minimum."""

BATCH_SIZE_INVALID = "BATCH_SIZE_INVALID"
"""Batch size not supported by kernel."""

GQA_UNSUPPORTED = "GQA_UNSUPPORTED"
"""Kernel does not support grouped query attention."""

GQA_HEADS_MISMATCH = "GQA_HEADS_MISMATCH"
"""GQA head count ratio not supported."""

MQA_UNSUPPORTED = "MQA_UNSUPPORTED"
"""Kernel does not support multi-query attention."""


# =============================================================================
# Attention Mask Reason Codes
# =============================================================================

ATTN_MASK_UNSUPPORTED = "ATTN_MASK_UNSUPPORTED"
"""Kernel does not support the provided attention mask type."""

ATTN_MASK_INVALID = "ATTN_MASK_INVALID"
"""Attention mask is invalid (e.g., incompatible with is_causal)."""

ATTN_MASK_CAUSAL_CONFLICT = "ATTN_MASK_CAUSAL_CONFLICT"
"""Both attention mask and is_causal=True specified."""

BIAS_UNSUPPORTED = "BIAS_UNSUPPORTED"
"""Kernel does not support attention bias."""

ALIBI_UNSUPPORTED = "ALIBI_UNSUPPORTED"
"""Kernel does not support ALiBi positional encoding."""

ATTN_BIAS_DEVICE_MISMATCH = "ATTN_BIAS_DEVICE_MISMATCH"
"""Attention bias is on different device than query tensor."""

ATTN_BIAS_BROADCAST_BATCH = "ATTN_BIAS_BROADCAST_BATCH"
"""Attention bias batch dimension cannot be broadcast (xFormers requirement)."""

ATTN_BIAS_BROADCAST_HEAD = "ATTN_BIAS_BROADCAST_HEAD"
"""Attention bias head dimension cannot be broadcast (xFormers requirement)."""


# =============================================================================
# CUDA/GPU Reason Codes
# =============================================================================

CUDA_GRAPH_UNSAFE = "CUDA_GRAPH_UNSAFE"
"""Kernel is not safe for CUDA graph capture."""

CUDA_BLOCK_LIMIT_EXCEEDED = "CUDA_BLOCK_LIMIT_EXCEEDED"
"""Kernel launch would exceed CUDA block limits."""

CUDA_GRID_DIM_EXCEEDED = "CUDA_GRID_DIM_EXCEEDED"
"""Kernel launch would exceed CUDA grid dimension limits."""

NON_DETERMINISTIC = "NON_DETERMINISTIC"
"""Kernel is non-deterministic when determinism is required."""

WORKSPACE_TOO_LARGE = "WORKSPACE_TOO_LARGE"
"""Kernel workspace exceeds available memory."""

SHARED_MEMORY_EXCEEDED = "SHARED_MEMORY_EXCEEDED"
"""Kernel requires more shared memory than available."""


# =============================================================================
# Backend Reason Codes
# =============================================================================

NOT_INSTALLED = "NOT_INSTALLED"
"""Backend is not installed."""

BACKEND_IMPORT_FAILED = "BACKEND_IMPORT_FAILED"
"""Backend import failed (version mismatch, missing deps, etc.)."""

JIT_DISABLED = "JIT_DISABLED"
"""JIT compilation is disabled and kernel requires JIT."""

BACKEND_ERROR = "BACKEND_ERROR"
"""Backend returned an error during operation."""

BACKEND_UNHEALTHY = "BACKEND_UNHEALTHY"
"""Backend is marked unhealthy due to repeated failures."""

BACKEND_VERSION_MISMATCH = "BACKEND_VERSION_MISMATCH"
"""Backend version does not meet requirements."""


# =============================================================================
# Memory/Layout Reason Codes
# =============================================================================

NOT_CONTIGUOUS = "NOT_CONTIGUOUS"
"""Tensor is not contiguous and kernel requires contiguous input."""

STRIDE_LAST_DIM = "STRIDE_LAST_DIM"
"""Last dimension stride is not 1."""

LAYOUT_UNSUPPORTED = "LAYOUT_UNSUPPORTED"
"""Tensor layout (BSHD/BHSD/NHD) not supported by kernel."""

MEMORY_HEADROOM_EXCEEDED = "MEMORY_HEADROOM_EXCEEDED"
"""Selection would exceed configured memory headroom."""

PLAN_BUCKET_MISS = "PLAN_BUCKET_MISS"
"""Input shape does not match any pre-planned bucket."""

ALIGNMENT_VIOLATION = "ALIGNMENT_VIOLATION"
"""Tensor does not meet alignment requirements."""


# =============================================================================
# Tokenizer Reason Codes
# =============================================================================

TOKENIZER_ID_MISMATCH = "TOKENIZER_ID_MISMATCH"
"""Tokenizer ID does not match expected value."""

VOCAB_HASH_MISMATCH = "VOCAB_HASH_MISMATCH"
"""Vocabulary hash does not match cached value."""

NORMALIZER_MISMATCH = "NORMALIZER_MISMATCH"
"""Normalizer configuration does not match."""

MERGES_HASH_MISMATCH = "MERGES_HASH_MISMATCH"
"""BPE merges hash does not match cached value."""

ADDED_TOKENS_MISMATCH = "ADDED_TOKENS_MISMATCH"
"""Added tokens configuration does not match."""

SPECIAL_TOKENS_MISMATCH = "SPECIAL_TOKENS_MISMATCH"
"""Special tokens configuration does not match."""

PRETOKENIZER_MISMATCH = "PRETOKENIZER_MISMATCH"
"""Pre-tokenizer configuration does not match."""


# =============================================================================
# KV Cache Reason Codes
# =============================================================================

KV_CACHE_LAYOUT_MISMATCH = "KV_CACHE_LAYOUT_MISMATCH"
"""KV cache layout does not match kernel expectations."""

KV_CACHE_DTYPE_MISMATCH = "KV_CACHE_DTYPE_MISMATCH"
"""KV cache dtype does not match kernel expectations."""

KV_STRATEGY_UNSUPPORTED = "KV_STRATEGY_UNSUPPORTED"
"""KV cache strategy (paged, contiguous, etc.) not supported."""

VIRTUAL_MEMORY_EXHAUSTED = "VIRTUAL_MEMORY_EXHAUSTED"
"""Virtual memory pool exhausted for KV cache."""

PAGE_SIZE_MISMATCH = "PAGE_SIZE_MISMATCH"
"""Paged attention page size does not match kernel."""


# =============================================================================
# Quantization Reason Codes
# =============================================================================

PACKED_WEIGHTS_REQUIRED = "PACKED_WEIGHTS_REQUIRED"
"""Kernel requires pre-packed weights."""

SCALE_LAYOUT_MISMATCH = "SCALE_LAYOUT_MISMATCH"
"""Quantization scale layout does not match kernel."""

ZERO_POINT_UNSUPPORTED = "ZERO_POINT_UNSUPPORTED"
"""Kernel does not support asymmetric quantization."""

GROUP_SIZE_INVALID = "GROUP_SIZE_INVALID"
"""Quantization group size not supported."""


# =============================================================================
# Distributed/Tensor Parallel Reason Codes
# =============================================================================

TP_INVARIANCE_REQUIRED = "TP_INVARIANCE_REQUIRED"
"""Kernel requires tensor parallel invariant computation."""

TP_SIZE_EXCEEDED = "TP_SIZE_EXCEEDED"
"""Tensor parallel size exceeds kernel maximum."""

REDUCTION_ORDER_MISMATCH = "REDUCTION_ORDER_MISMATCH"
"""Reduction order incompatible with distributed setting."""

RANK_MISMATCH = "RANK_MISMATCH"
"""Rank configuration does not match kernel requirements."""

PP_SIZE_EXCEEDED = "PP_SIZE_EXCEEDED"
"""Pipeline parallel size exceeds kernel maximum."""

VERSION_MISMATCH = "VERSION_MISMATCH"
"""LayerZero version mismatch across distributed ranks."""

SELECTION_HASH_MISMATCH = "SELECTION_HASH_MISMATCH"
"""Selection hash mismatch across distributed ranks."""

BROADCAST_FAILED = "BROADCAST_FAILED"
"""Failed to broadcast selection across distributed ranks."""

TP_INVARIANT_KERNEL_REQUIRED = "TP_INVARIANT_KERNEL_REQUIRED"
"""TP-invariant kernel required but not available."""


# =============================================================================
# Speculative Decoding Reason Codes
# =============================================================================

SPEC_DECODE_PP_INCOMPATIBLE = "SPEC_DECODE_PP_INCOMPATIBLE"
"""Speculative decoding incompatible with pipeline parallelism."""

SPEC_DECODE_DRAFT_TP_CONSTRAINT = "SPEC_DECODE_DRAFT_TP_CONSTRAINT"
"""Draft model TP constraint violated."""

SPEC_DECODE_KV_INCOMPATIBLE = "SPEC_DECODE_KV_INCOMPATIBLE"
"""KV cache incompatible with speculative decoding."""

SPEC_DECODE_ALGORITHM_UNSUPPORTED = "SPEC_DECODE_ALGORITHM_UNSUPPORTED"
"""Speculative decoding algorithm not supported."""


# =============================================================================
# Schema Reason Codes
# =============================================================================

CAPABILITIES_SCHEMA_MISMATCH = "CAPABILITIES_SCHEMA_MISMATCH"
"""Capabilities JSON schema does not match expected version."""

DISPATCH_TABLE_INVALID = "DISPATCH_TABLE_INVALID"
"""Dispatch table schema is invalid."""

CONFIG_VALIDATION_FAILED = "CONFIG_VALIDATION_FAILED"
"""Configuration validation failed."""


# =============================================================================
# ALL_REASON_CODES mapping
# =============================================================================

ALL_REASON_CODES: dict[str, ReasonCategory] = {
    # Hardware
    PLATFORM_MISMATCH: ReasonCategory.HARDWARE,
    SM_TOO_OLD: ReasonCategory.HARDWARE,
    SM_TOO_NEW: ReasonCategory.HARDWARE,
    GPU_GENERATION_UNSUPPORTED: ReasonCategory.HARDWARE,
    TENSOR_CORE_GEN_UNSUPPORTED: ReasonCategory.HARDWARE,
    INSTRUCTION_SET_MISMATCH: ReasonCategory.HARDWARE,
    DEVICE_CAPABILITY_UNSUPPORTED: ReasonCategory.HARDWARE,
    DRIVER_VERSION_UNSUPPORTED: ReasonCategory.HARDWARE,
    # Dtype
    DTYPE_UNSUPPORTED: ReasonCategory.DTYPE,
    QUANT_FORMAT_UNSUPPORTED: ReasonCategory.DTYPE,
    QUANT_ACCURACY_THRESHOLD_EXCEEDED: ReasonCategory.DTYPE,
    REQUANTIZATION_REQUIRED: ReasonCategory.DTYPE,
    MIXED_DTYPE_UNSUPPORTED: ReasonCategory.DTYPE,
    # Shape
    HEAD_DIM_INVALID: ReasonCategory.SHAPE,
    HEAD_DIM_ALIGNMENT: ReasonCategory.SHAPE,
    HEAD_DIM_TOO_LARGE: ReasonCategory.SHAPE,
    HEAD_DIM_TOO_SMALL: ReasonCategory.SHAPE,
    SEQ_TOO_LONG: ReasonCategory.SHAPE,
    SEQ_TOO_SHORT: ReasonCategory.SHAPE,
    BATCH_SIZE_INVALID: ReasonCategory.SHAPE,
    GQA_UNSUPPORTED: ReasonCategory.SHAPE,
    GQA_HEADS_MISMATCH: ReasonCategory.SHAPE,
    MQA_UNSUPPORTED: ReasonCategory.SHAPE,
    # Attention
    ATTN_MASK_UNSUPPORTED: ReasonCategory.ATTENTION,
    ATTN_MASK_INVALID: ReasonCategory.ATTENTION,
    ATTN_MASK_CAUSAL_CONFLICT: ReasonCategory.ATTENTION,
    BIAS_UNSUPPORTED: ReasonCategory.ATTENTION,
    ALIBI_UNSUPPORTED: ReasonCategory.ATTENTION,
    ATTN_BIAS_DEVICE_MISMATCH: ReasonCategory.ATTENTION,
    ATTN_BIAS_BROADCAST_BATCH: ReasonCategory.ATTENTION,
    ATTN_BIAS_BROADCAST_HEAD: ReasonCategory.ATTENTION,
    # CUDA
    CUDA_GRAPH_UNSAFE: ReasonCategory.CUDA,
    CUDA_BLOCK_LIMIT_EXCEEDED: ReasonCategory.CUDA,
    CUDA_GRID_DIM_EXCEEDED: ReasonCategory.CUDA,
    NON_DETERMINISTIC: ReasonCategory.CUDA,
    WORKSPACE_TOO_LARGE: ReasonCategory.CUDA,
    SHARED_MEMORY_EXCEEDED: ReasonCategory.CUDA,
    # Backend
    NOT_INSTALLED: ReasonCategory.BACKEND,
    BACKEND_IMPORT_FAILED: ReasonCategory.BACKEND,
    JIT_DISABLED: ReasonCategory.BACKEND,
    BACKEND_ERROR: ReasonCategory.BACKEND,
    BACKEND_UNHEALTHY: ReasonCategory.BACKEND,
    BACKEND_VERSION_MISMATCH: ReasonCategory.BACKEND,
    # Memory
    NOT_CONTIGUOUS: ReasonCategory.MEMORY,
    STRIDE_LAST_DIM: ReasonCategory.MEMORY,
    LAYOUT_UNSUPPORTED: ReasonCategory.MEMORY,
    MEMORY_HEADROOM_EXCEEDED: ReasonCategory.MEMORY,
    PLAN_BUCKET_MISS: ReasonCategory.MEMORY,
    ALIGNMENT_VIOLATION: ReasonCategory.MEMORY,
    # Tokenizer
    TOKENIZER_ID_MISMATCH: ReasonCategory.TOKENIZER,
    VOCAB_HASH_MISMATCH: ReasonCategory.TOKENIZER,
    NORMALIZER_MISMATCH: ReasonCategory.TOKENIZER,
    MERGES_HASH_MISMATCH: ReasonCategory.TOKENIZER,
    ADDED_TOKENS_MISMATCH: ReasonCategory.TOKENIZER,
    SPECIAL_TOKENS_MISMATCH: ReasonCategory.TOKENIZER,
    PRETOKENIZER_MISMATCH: ReasonCategory.TOKENIZER,
    # KV Cache
    KV_CACHE_LAYOUT_MISMATCH: ReasonCategory.KV_CACHE,
    KV_CACHE_DTYPE_MISMATCH: ReasonCategory.KV_CACHE,
    KV_STRATEGY_UNSUPPORTED: ReasonCategory.KV_CACHE,
    VIRTUAL_MEMORY_EXHAUSTED: ReasonCategory.KV_CACHE,
    PAGE_SIZE_MISMATCH: ReasonCategory.KV_CACHE,
    # Quantization
    PACKED_WEIGHTS_REQUIRED: ReasonCategory.QUANTIZATION,
    SCALE_LAYOUT_MISMATCH: ReasonCategory.QUANTIZATION,
    ZERO_POINT_UNSUPPORTED: ReasonCategory.QUANTIZATION,
    GROUP_SIZE_INVALID: ReasonCategory.QUANTIZATION,
    # Distributed
    TP_INVARIANCE_REQUIRED: ReasonCategory.DISTRIBUTED,
    TP_SIZE_EXCEEDED: ReasonCategory.DISTRIBUTED,
    REDUCTION_ORDER_MISMATCH: ReasonCategory.DISTRIBUTED,
    RANK_MISMATCH: ReasonCategory.DISTRIBUTED,
    PP_SIZE_EXCEEDED: ReasonCategory.DISTRIBUTED,
    VERSION_MISMATCH: ReasonCategory.DISTRIBUTED,
    SELECTION_HASH_MISMATCH: ReasonCategory.DISTRIBUTED,
    BROADCAST_FAILED: ReasonCategory.DISTRIBUTED,
    TP_INVARIANT_KERNEL_REQUIRED: ReasonCategory.DISTRIBUTED,
    # Speculative
    SPEC_DECODE_PP_INCOMPATIBLE: ReasonCategory.SPECULATIVE,
    SPEC_DECODE_DRAFT_TP_CONSTRAINT: ReasonCategory.SPECULATIVE,
    SPEC_DECODE_KV_INCOMPATIBLE: ReasonCategory.SPECULATIVE,
    SPEC_DECODE_ALGORITHM_UNSUPPORTED: ReasonCategory.SPECULATIVE,
    # Schema
    CAPABILITIES_SCHEMA_MISMATCH: ReasonCategory.SCHEMA,
    DISPATCH_TABLE_INVALID: ReasonCategory.SCHEMA,
    CONFIG_VALIDATION_FAILED: ReasonCategory.SCHEMA,
}


def make_reason(code: str, message: str) -> Reason:
    """Factory function to create a Reason with automatic category lookup.

    Args:
        code: Reason code string (must be in ALL_REASON_CODES).
        message: Human-readable message.

    Returns:
        New Reason instance with correct category.

    Raises:
        ValueError: If code is not found in ALL_REASON_CODES.
    """
    if code not in ALL_REASON_CODES:
        raise ValueError(f"Unknown reason code: {code}")
    return Reason(code=code, message=message, category=ALL_REASON_CODES[code])
