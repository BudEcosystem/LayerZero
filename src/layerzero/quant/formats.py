"""Quantization format definitions.

This module defines quantization formats including:
- INT8 (8-bit integer)
- FP8 E4M3 (8-bit float with 4 exponent, 3 mantissa bits)
- FP8 E5M2 (8-bit float with 5 exponent, 2 mantissa bits)
- MXFP4 (Microscaling FP4 - block-scaled 4-bit)
- NVFP4 (NVIDIA FP4 - proprietary 4-bit format)
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Final

from layerzero.device import GPUGeneration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QuantFormat:
    """Quantization format specification.

    Attributes:
        name: Format identifier.
        bits: Total bits per element.
        mantissa_bits: Mantissa bits (for floats).
        exponent_bits: Exponent bits (for floats).
        is_signed: Whether format is signed.
        is_floating_point: Whether format is floating point.
        is_block_scaled: Whether format uses block scaling.
        block_size: Block size for block-scaled formats.
        min_value: Minimum representable value.
        max_value: Maximum representable value.
        supported_generations: GPU generations supporting this format.
    """

    name: str
    bits: int
    mantissa_bits: int = 0
    exponent_bits: int = 0
    is_signed: bool = True
    is_floating_point: bool = False
    is_block_scaled: bool = False
    block_size: int = 1
    min_value: float = 0.0
    max_value: float = 0.0
    supported_generations: frozenset[GPUGeneration] = field(
        default_factory=frozenset
    )


# ============================================================================
# Format Definitions
# ============================================================================

INT8_FORMAT = QuantFormat(
    name="int8",
    bits=8,
    mantissa_bits=0,
    exponent_bits=0,
    is_signed=True,
    is_floating_point=False,
    is_block_scaled=False,
    min_value=-128.0,
    max_value=127.0,
    supported_generations=frozenset([
        GPUGeneration.TURING,
        GPUGeneration.AMPERE,
        GPUGeneration.ADA_LOVELACE,
        GPUGeneration.HOPPER,
        GPUGeneration.BLACKWELL,
    ]),
)

FP8_E4M3_FORMAT = QuantFormat(
    name="fp8_e4m3",
    bits=8,
    mantissa_bits=3,
    exponent_bits=4,
    is_signed=True,
    is_floating_point=True,
    is_block_scaled=False,
    min_value=-448.0,
    max_value=448.0,
    supported_generations=frozenset([
        GPUGeneration.ADA_LOVELACE,  # Ada has limited FP8
        GPUGeneration.HOPPER,
        GPUGeneration.BLACKWELL,
    ]),
)

FP8_E5M2_FORMAT = QuantFormat(
    name="fp8_e5m2",
    bits=8,
    mantissa_bits=2,
    exponent_bits=5,
    is_signed=True,
    is_floating_point=True,
    is_block_scaled=False,
    min_value=-57344.0,
    max_value=57344.0,
    supported_generations=frozenset([
        GPUGeneration.ADA_LOVELACE,
        GPUGeneration.HOPPER,
        GPUGeneration.BLACKWELL,
    ]),
)

MXFP4_FORMAT = QuantFormat(
    name="mxfp4",
    bits=4,
    mantissa_bits=2,
    exponent_bits=1,
    is_signed=True,
    is_floating_point=True,
    is_block_scaled=True,
    block_size=32,
    min_value=-6.0,
    max_value=6.0,
    supported_generations=frozenset([
        GPUGeneration.BLACKWELL,
    ]),
)

NVFP4_FORMAT = QuantFormat(
    name="nvfp4",
    bits=4,
    mantissa_bits=2,
    exponent_bits=1,
    is_signed=True,
    is_floating_point=True,
    is_block_scaled=True,
    block_size=32,
    min_value=-6.0,
    max_value=6.0,
    supported_generations=frozenset([
        GPUGeneration.BLACKWELL,
    ]),
)

# ============================================================================
# Format Registry
# ============================================================================

FORMAT_REGISTRY: Final[dict[str, QuantFormat]] = {
    "int8": INT8_FORMAT,
    "fp8_e4m3": FP8_E4M3_FORMAT,
    "fp8_e5m2": FP8_E5M2_FORMAT,
    "mxfp4": MXFP4_FORMAT,
    "nvfp4": NVFP4_FORMAT,
}


def get_format(name: str) -> QuantFormat | None:
    """Get quantization format by name.

    Args:
        name: Format name.

    Returns:
        QuantFormat or None if not found.
    """
    return FORMAT_REGISTRY.get(name)


def is_format_supported(name: str, gpu_generation: GPUGeneration) -> bool:
    """Check if format is supported on a GPU generation.

    Args:
        name: Format name.
        gpu_generation: GPU generation to check.

    Returns:
        True if format is supported.
    """
    fmt = get_format(name)
    if fmt is None:
        return False

    return gpu_generation in fmt.supported_generations


def get_supported_formats(gpu_generation: GPUGeneration) -> list[str]:
    """Get all formats supported on a GPU generation.

    Args:
        gpu_generation: GPU generation.

    Returns:
        List of supported format names.
    """
    supported = []
    for name, fmt in FORMAT_REGISTRY.items():
        if gpu_generation in fmt.supported_generations:
            supported.append(name)

    return supported
