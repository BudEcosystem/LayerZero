"""Quantization format selection engine.

This module provides:
- QuantizationConfig for describing quantization settings
- QuantFormatSelector for filtering kernels by format support
- Format selection utilities
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any

from layerzero.device import GPUGeneration
from layerzero.quant.formats import (
    FORMAT_REGISTRY,
    get_format,
    is_format_supported,
)
from layerzero.quant.scales import ScaleGranularity

logger = logging.getLogger(__name__)


@dataclass
class QuantizationConfig:
    """Configuration for quantization.

    Attributes:
        format_name: Quantization format name (e.g., "int8", "fp8_e4m3").
        is_enabled: Whether quantization is enabled.
        scale_granularity: Scale granularity level.
        symmetric: Use symmetric quantization.
        calibration_method: Calibration method name.
    """

    format_name: str | None = None
    is_enabled: bool = False
    scale_granularity: ScaleGranularity = ScaleGranularity.PER_TENSOR
    symmetric: bool = True
    calibration_method: str = "minmax"


class QuantFormatSelector:
    """Quantization format selector.

    Filters kernels based on quantization format support and
    validates hardware compatibility.
    """

    def __init__(self) -> None:
        """Initialize format selector."""
        self._lock = RLock()

        logger.debug("QuantFormatSelector initialized")

    def filter_kernels_for_format(
        self,
        kernels: list[dict[str, Any]],
        config: QuantizationConfig,
    ) -> list[dict[str, Any]]:
        """Filter kernels by quantization format support.

        Args:
            kernels: List of kernel dicts.
            config: Quantization configuration.

        Returns:
            Filtered list of kernels supporting the format.
        """
        with self._lock:
            if not config.is_enabled or config.format_name is None:
                # Return all kernels when quantization disabled
                return kernels

            filtered = []
            format_key = self._get_format_support_key(config.format_name)

            for kernel in kernels:
                # Check if kernel supports the format
                if kernel.get(format_key, False):
                    filtered.append(kernel)

            logger.debug(
                "Filtered %d kernels to %d for format %s",
                len(kernels),
                len(filtered),
                config.format_name,
            )
            return filtered

    def _get_format_support_key(self, format_name: str) -> str:
        """Get kernel dict key for format support.

        Args:
            format_name: Quantization format name.

        Returns:
            Key for checking format support.
        """
        # Map format names to kernel support keys
        format_keys = {
            "int8": "supports_int8",
            "fp8_e4m3": "supports_fp8",
            "fp8_e5m2": "supports_fp8",
            "mxfp4": "supports_fp4",
            "nvfp4": "supports_fp4",
        }
        return format_keys.get(format_name, f"supports_{format_name}")


def validate_format_for_hardware(
    format_name: str,
    gpu_generation: GPUGeneration,
) -> bool:
    """Validate format is supported on hardware.

    Args:
        format_name: Quantization format name.
        gpu_generation: GPU generation.

    Returns:
        True if format is supported.
    """
    return is_format_supported(format_name, gpu_generation)


def select_best_format(
    gpu_generation: GPUGeneration,
    available_formats: list[str],
    prefer_lower_precision: bool = True,
) -> str | None:
    """Select best quantization format for hardware.

    Args:
        gpu_generation: Target GPU generation.
        available_formats: Available format names.
        prefer_lower_precision: Prefer lower precision formats.

    Returns:
        Best format name or None if none supported.
    """
    # Filter to supported formats
    supported = [
        f for f in available_formats
        if is_format_supported(f, gpu_generation)
    ]

    if not supported:
        return None

    if prefer_lower_precision:
        # Priority: FP4 > FP8 > INT8
        priority = ["mxfp4", "nvfp4", "fp8_e4m3", "fp8_e5m2", "int8"]
        for fmt in priority:
            if fmt in supported:
                return fmt

    # Return first supported
    return supported[0]


def select_format_with_fallback(
    requested_format: str,
    gpu_generation: GPUGeneration,
    fallback_format: str | None = None,
) -> str | None:
    """Select format with fallback if not supported.

    Args:
        requested_format: Requested format name.
        gpu_generation: Target GPU generation.
        fallback_format: Fallback format if requested not supported.

    Returns:
        Selected format name or None.
    """
    # Check if requested format is supported
    if is_format_supported(requested_format, gpu_generation):
        return requested_format

    logger.debug(
        "Format %s not supported on %s, using fallback %s",
        requested_format,
        gpu_generation.value,
        fallback_format,
    )

    # Check fallback
    if fallback_format and is_format_supported(fallback_format, gpu_generation):
        return fallback_format

    return None
