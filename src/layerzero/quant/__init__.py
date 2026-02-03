"""Quantization format selection engine.

This module provides:
- Quantization format definitions (INT8, FP8, FP4)
- Hardware support detection
- Scale handling (per-tensor, per-channel, blockwise)
- Format selection and kernel filtering
"""
from __future__ import annotations

from layerzero.quant.formats import (
    FORMAT_REGISTRY,
    FP8_E4M3_FORMAT,
    FP8_E5M2_FORMAT,
    INT8_FORMAT,
    MXFP4_FORMAT,
    NVFP4_FORMAT,
    QuantFormat,
    get_format,
    get_supported_formats,
    is_format_supported,
)
from layerzero.quant.format_selection import (
    QuantFormatSelector,
    QuantizationConfig,
    select_best_format,
    select_format_with_fallback,
    validate_format_for_hardware,
)
from layerzero.quant.scales import (
    QuantScales,
    ScaleGranularity,
    compute_scale,
)

__all__ = [
    # Formats
    "QuantFormat",
    "FORMAT_REGISTRY",
    "INT8_FORMAT",
    "FP8_E4M3_FORMAT",
    "FP8_E5M2_FORMAT",
    "MXFP4_FORMAT",
    "NVFP4_FORMAT",
    "get_format",
    "get_supported_formats",
    "is_format_supported",
    # Selection
    "QuantFormatSelector",
    "QuantizationConfig",
    "select_best_format",
    "select_format_with_fallback",
    "validate_format_for_hardware",
    # Scales
    "QuantScales",
    "ScaleGranularity",
    "compute_scale",
]
