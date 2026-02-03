"""FlashAttention 4 hardware requirements.

This module handles hardware compatibility checks for FA4,
including SM version, tcgen05 support, and memory requirements.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Final

logger = logging.getLogger(__name__)

# FA4 hardware requirements
FA4_MIN_SM: Final[tuple[int, int]] = (10, 0)
FA4_MIN_SHARED_MEMORY_KB: Final[int] = 228  # Blackwell shared memory


@dataclass(frozen=True)
class FA4MemoryRequirement:
    """FA4 memory requirement result.

    Attributes:
        is_sufficient: Whether memory is sufficient.
        required_bytes: Required memory in bytes.
        available_bytes: Available memory in bytes.
        message: Human-readable message.
    """

    is_sufficient: bool
    required_bytes: int
    available_bytes: int
    message: str


def is_fa4_compatible_sm(sm_version: tuple[int, int]) -> bool:
    """Check if SM version is compatible with FA4.

    FA4 requires SM 10.0+ (Blackwell and newer).

    Args:
        sm_version: GPU SM version as (major, minor).

    Returns:
        True if FA4 compatible.
    """
    return sm_version >= FA4_MIN_SM


def has_tcgen05_support(sm_version: tuple[int, int]) -> bool:
    """Check if SM version has tcgen05.mma support.

    tcgen05 tensor core operations are available on Blackwell (SM 10.0+).

    Args:
        sm_version: GPU SM version as (major, minor).

    Returns:
        True if tcgen05 is available.
    """
    # tcgen05.mma is a Blackwell-specific feature
    return sm_version[0] >= 10


def check_fa4_memory_requirements(
    available_memory_bytes: int,
    sequence_length: int,
    head_dim: int,
    batch_size: int,
    num_heads: int = 32,
    num_kv_heads: int | None = None,
    dtype_bytes: int = 2,  # fp16/bf16
) -> FA4MemoryRequirement:
    """Check if memory is sufficient for FA4.

    FA4 has specific memory requirements due to its tiling strategy.

    Args:
        available_memory_bytes: Available GPU memory in bytes.
        sequence_length: Sequence length.
        head_dim: Head dimension.
        batch_size: Batch size.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads (for GQA).
        dtype_bytes: Bytes per element.

    Returns:
        FA4MemoryRequirement with check result.
    """
    if num_kv_heads is None:
        num_kv_heads = num_heads

    # Estimate memory for Q, K, V tensors
    q_size = batch_size * sequence_length * num_heads * head_dim * dtype_bytes
    k_size = batch_size * sequence_length * num_kv_heads * head_dim * dtype_bytes
    v_size = batch_size * sequence_length * num_kv_heads * head_dim * dtype_bytes
    o_size = batch_size * sequence_length * num_heads * head_dim * dtype_bytes

    # Workspace for FA4 (softmax LSE, etc.)
    # FA4 uses more workspace for its tiling strategy
    workspace_size = batch_size * num_heads * sequence_length * 4  # fp32 LSE

    total_required = q_size + k_size + v_size + o_size + workspace_size

    is_sufficient = available_memory_bytes >= total_required

    if is_sufficient:
        message = f"Memory sufficient: {available_memory_bytes // (1024**2)} MB available"
    else:
        message = (
            f"Insufficient memory: need {total_required // (1024**2)} MB, "
            f"have {available_memory_bytes // (1024**2)} MB"
        )

    return FA4MemoryRequirement(
        is_sufficient=is_sufficient,
        required_bytes=total_required,
        available_bytes=available_memory_bytes,
        message=message,
    )
