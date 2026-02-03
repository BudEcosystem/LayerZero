"""
LayerZero GPU Generation Detection

Maps NVIDIA SM versions to GPU generations and tensor core generations.
Provides type-safe GPU generation enum with ordering support.

This module provides:
- GPUGeneration: Ordered enum of NVIDIA GPU architectures
- sm_to_generation(): Map SM version to generation
- get_tensor_core_gen(): Get tensor core generation for GPU generation
- SM_TO_GENERATION: Direct mapping dictionary
- GENERATION_TO_TC_GEN: Tensor core generation mapping
"""
from __future__ import annotations

from enum import Enum, unique
from functools import total_ordering
from typing import Final


@total_ordering
@unique
class GPUGeneration(str, Enum):
    """NVIDIA GPU architecture generations.

    Ordered enum supporting comparison operators for generation checks.
    Newer generations compare as greater than older generations.

    Members:
        UNKNOWN: Unknown/unsupported generation (compares as lowest)
        TURING: SM 7.5 (RTX 20-series, GTX 16-series)
        AMPERE: SM 8.0-8.7 (RTX 30-series, A100, A10)
        ADA_LOVELACE: SM 8.9 (RTX 40-series, L4, L40)
        HOPPER: SM 9.0 (H100, H200)
        BLACKWELL: SM 10.0+ (RTX 50-series, B100, B200)

    Example:
        >>> GPUGeneration.AMPERE < GPUGeneration.HOPPER
        True
        >>> GPUGeneration.BLACKWELL >= GPUGeneration.HOPPER
        True
    """

    UNKNOWN = "unknown"
    TURING = "turing"
    AMPERE = "ampere"
    ADA_LOVELACE = "ada"
    HOPPER = "hopper"
    BLACKWELL = "blackwell"

    def __lt__(self, other: object) -> bool:
        """Compare generations by architecture order."""
        if not isinstance(other, GPUGeneration):
            return NotImplemented
        return _GENERATION_ORDER.index(self) < _GENERATION_ORDER.index(other)

    def __le__(self, other: object) -> bool:
        """Compare generations by architecture order (less than or equal)."""
        if not isinstance(other, GPUGeneration):
            return NotImplemented
        return _GENERATION_ORDER.index(self) <= _GENERATION_ORDER.index(other)

    def __gt__(self, other: object) -> bool:
        """Compare generations by architecture order (greater than)."""
        if not isinstance(other, GPUGeneration):
            return NotImplemented
        return _GENERATION_ORDER.index(self) > _GENERATION_ORDER.index(other)

    def __ge__(self, other: object) -> bool:
        """Compare generations by architecture order (greater than or equal)."""
        if not isinstance(other, GPUGeneration):
            return NotImplemented
        return _GENERATION_ORDER.index(self) >= _GENERATION_ORDER.index(other)


# Generation ordering for comparison operators
_GENERATION_ORDER: Final[list[GPUGeneration]] = [
    GPUGeneration.UNKNOWN,
    GPUGeneration.TURING,
    GPUGeneration.AMPERE,
    GPUGeneration.ADA_LOVELACE,
    GPUGeneration.HOPPER,
    GPUGeneration.BLACKWELL,
]


# =============================================================================
# SM Version to Generation Mapping
# =============================================================================

SM_TO_GENERATION: Final[dict[tuple[int, int], GPUGeneration]] = {
    # Turing (SM 7.5)
    (7, 5): GPUGeneration.TURING,
    # Ampere (SM 8.0-8.7)
    (8, 0): GPUGeneration.AMPERE,  # A100, A30
    (8, 6): GPUGeneration.AMPERE,  # RTX 30xx, A10, A40
    (8, 7): GPUGeneration.AMPERE,  # Jetson Orin
    # Ada Lovelace (SM 8.9)
    (8, 9): GPUGeneration.ADA_LOVELACE,  # RTX 40xx, L4, L40
    # Hopper (SM 9.0)
    (9, 0): GPUGeneration.HOPPER,  # H100, H200
    # Blackwell (SM 10.0+, also SM 12.0 for RTX 50xx)
    (10, 0): GPUGeneration.BLACKWELL,  # B100, B200
    (12, 0): GPUGeneration.BLACKWELL,  # RTX 50xx series
}


def sm_to_generation(major: int, minor: int) -> GPUGeneration:
    """Map SM (Streaming Multiprocessor) version to GPU generation.

    Handles both exact matches and range-based lookups for future
    SM versions within known architecture families.

    Args:
        major: SM major version (e.g., 8 for SM 8.6)
        minor: SM minor version (e.g., 6 for SM 8.6)

    Returns:
        GPUGeneration enum value.

    Example:
        >>> sm_to_generation(8, 6)
        <GPUGeneration.AMPERE: 'ampere'>
        >>> sm_to_generation(9, 0)
        <GPUGeneration.HOPPER: 'hopper'>
    """
    # Handle invalid/negative versions
    if major < 0 or minor < 0:
        return GPUGeneration.UNKNOWN

    # Try exact match first
    key = (major, minor)
    if key in SM_TO_GENERATION:
        return SM_TO_GENERATION[key]

    # Range-based fallback for SM versions not in exact map
    if major == 7 and minor >= 5:
        return GPUGeneration.TURING
    if major == 8 and minor < 9:
        return GPUGeneration.AMPERE
    if major == 8 and minor >= 9:
        return GPUGeneration.ADA_LOVELACE
    if major == 9:
        return GPUGeneration.HOPPER
    if major >= 10:
        # All future architectures >= SM 10.0 default to BLACKWELL
        # until a new generation is added
        return GPUGeneration.BLACKWELL

    return GPUGeneration.UNKNOWN


# =============================================================================
# Tensor Core Generation Mapping
# =============================================================================

GENERATION_TO_TC_GEN: Final[dict[GPUGeneration, int]] = {
    GPUGeneration.UNKNOWN: 0,
    GPUGeneration.TURING: 2,  # 1st gen FP16 tensor cores, INT8/INT4
    GPUGeneration.AMPERE: 3,  # TF32, sparse tensor cores
    GPUGeneration.ADA_LOVELACE: 3,  # Enhanced TC3 with FP8
    GPUGeneration.HOPPER: 4,  # TMA, warp-specialized TC
    GPUGeneration.BLACKWELL: 5,  # FP4, enhanced memory hierarchy
}


def get_tensor_core_gen(generation: GPUGeneration) -> int:
    """Get tensor core generation for a GPU generation.

    Tensor core generations indicate hardware tensor operation capabilities:
    - Gen 0: No tensor cores (pre-Volta)
    - Gen 2: Turing - FP16/INT8/INT4 tensor ops
    - Gen 3: Ampere/Ada - TF32, BF16, sparsity, FP8 (Ada only)
    - Gen 4: Hopper - TMA, warp specialization, better FP8
    - Gen 5: Blackwell - FP4, NVFP4, enhanced memory

    Args:
        generation: GPUGeneration enum value.

    Returns:
        Tensor core generation (0-5).

    Example:
        >>> get_tensor_core_gen(GPUGeneration.AMPERE)
        3
        >>> get_tensor_core_gen(GPUGeneration.HOPPER)
        4
    """
    return GENERATION_TO_TC_GEN.get(generation, 0)


# =============================================================================
# GPU Feature Support
# =============================================================================

def supports_bf16(generation: GPUGeneration) -> bool:
    """Check if GPU generation supports BF16 (bfloat16).

    BF16 is supported starting from Ampere (SM 8.0+).

    Args:
        generation: GPUGeneration enum value.

    Returns:
        True if BF16 is supported.
    """
    return generation >= GPUGeneration.AMPERE


def supports_fp8(generation: GPUGeneration) -> bool:
    """Check if GPU generation supports FP8.

    FP8 (E4M3/E5M2) is supported starting from Ada Lovelace (SM 8.9+).

    Args:
        generation: GPUGeneration enum value.

    Returns:
        True if FP8 is supported.
    """
    return generation >= GPUGeneration.ADA_LOVELACE


def supports_fp4(generation: GPUGeneration) -> bool:
    """Check if GPU generation supports FP4/NVFP4.

    NVFP4 is supported starting from Blackwell (SM 10.0+).

    Args:
        generation: GPUGeneration enum value.

    Returns:
        True if FP4 is supported.
    """
    return generation >= GPUGeneration.BLACKWELL


def supports_tma(generation: GPUGeneration) -> bool:
    """Check if GPU generation supports Tensor Memory Access (TMA).

    TMA is a Hopper+ feature for efficient tensor data movement.

    Args:
        generation: GPUGeneration enum value.

    Returns:
        True if TMA is supported.
    """
    return generation >= GPUGeneration.HOPPER


def get_max_shared_memory_kb(generation: GPUGeneration) -> int:
    """Get maximum shared memory per SM in KB.

    Returns approximate maximum configurable shared memory.

    Args:
        generation: GPUGeneration enum value.

    Returns:
        Maximum shared memory in KB.
    """
    _shared_memory_map: dict[GPUGeneration, int] = {
        GPUGeneration.UNKNOWN: 48,
        GPUGeneration.TURING: 64,
        GPUGeneration.AMPERE: 164,
        GPUGeneration.ADA_LOVELACE: 164,
        GPUGeneration.HOPPER: 228,
        GPUGeneration.BLACKWELL: 228,  # TBD, using Hopper as baseline
    }
    return _shared_memory_map.get(generation, 48)
