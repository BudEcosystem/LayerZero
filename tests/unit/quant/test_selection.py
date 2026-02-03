"""Tests for quantization format selection."""
from __future__ import annotations

import pytest
from typing import Any

from layerzero.device import GPUGeneration


class TestQuantizationKernelSelection:
    """Tests for quantization-aware kernel selection."""

    def test_quant_kernel_selected_for_int8(self, ampere_device) -> None:
        """INT8 kernel selected when enabled."""
        from layerzero.quant.format_selection import (
            QuantFormatSelector,
            QuantizationConfig,
        )

        selector = QuantFormatSelector()
        config = QuantizationConfig(
            format_name="int8",
            is_enabled=True,
        )

        # Get kernels supporting INT8
        kernels = selector.filter_kernels_for_format(
            kernels=[
                {"id": "k1", "supports_int8": True},
                {"id": "k2", "supports_int8": False},
                {"id": "k3", "supports_int8": True},
            ],
            config=config,
        )

        ids = [k["id"] for k in kernels]
        assert "k1" in ids
        assert "k3" in ids
        assert "k2" not in ids

    def test_quant_kernel_selected_for_fp8(self, hopper_device) -> None:
        """FP8 kernel selected when enabled."""
        from layerzero.quant.format_selection import (
            QuantFormatSelector,
            QuantizationConfig,
        )

        selector = QuantFormatSelector()
        config = QuantizationConfig(
            format_name="fp8_e4m3",
            is_enabled=True,
        )

        kernels = selector.filter_kernels_for_format(
            kernels=[
                {"id": "k1", "supports_fp8": True},
                {"id": "k2", "supports_fp8": False},
            ],
            config=config,
        )

        ids = [k["id"] for k in kernels]
        assert "k1" in ids
        assert "k2" not in ids

    def test_no_filtering_when_disabled(self) -> None:
        """No filtering when quantization disabled."""
        from layerzero.quant.format_selection import (
            QuantFormatSelector,
            QuantizationConfig,
        )

        selector = QuantFormatSelector()
        config = QuantizationConfig(
            format_name=None,
            is_enabled=False,
        )

        kernels = [
            {"id": "k1", "supports_int8": True},
            {"id": "k2", "supports_int8": False},
        ]

        # Should return all kernels when disabled
        result = selector.filter_kernels_for_format(kernels, config)
        assert len(result) == 2

    def test_select_best_quant_format(self) -> None:
        """Select best quantization format for device."""
        from layerzero.quant.format_selection import (
            QuantFormatSelector,
            select_best_format,
        )

        # Blackwell gets FP8 or FP4
        format_bw = select_best_format(
            GPUGeneration.BLACKWELL,
            available_formats=["int8", "fp8_e4m3", "mxfp4"],
            prefer_lower_precision=True,
        )
        assert format_bw in ["fp8_e4m3", "mxfp4"]

        # Hopper gets FP8
        format_hp = select_best_format(
            GPUGeneration.HOPPER,
            available_formats=["int8", "fp8_e4m3"],
            prefer_lower_precision=True,
        )
        assert format_hp == "fp8_e4m3"

        # Ampere gets INT8
        format_amp = select_best_format(
            GPUGeneration.AMPERE,
            available_formats=["int8", "fp8_e4m3"],
            prefer_lower_precision=True,
        )
        assert format_amp == "int8"

    def test_format_selector_validates_hardware(self) -> None:
        """Format selector validates hardware support."""
        from layerzero.quant.format_selection import (
            QuantFormatSelector,
            QuantizationConfig,
            validate_format_for_hardware,
        )

        # FP8 on Ampere should fail
        is_valid = validate_format_for_hardware(
            format_name="fp8_e4m3",
            gpu_generation=GPUGeneration.AMPERE,
        )
        assert is_valid is False

        # FP8 on Hopper should pass
        is_valid = validate_format_for_hardware(
            format_name="fp8_e4m3",
            gpu_generation=GPUGeneration.HOPPER,
        )
        assert is_valid is True

    def test_format_selector_with_fallback(self) -> None:
        """Format selector falls back to supported format."""
        from layerzero.quant.format_selection import (
            select_format_with_fallback,
        )

        # Request FP8 on Ampere, should fall back to INT8
        result = select_format_with_fallback(
            requested_format="fp8_e4m3",
            gpu_generation=GPUGeneration.AMPERE,
            fallback_format="int8",
        )
        assert result == "int8"

        # Request FP8 on Hopper, should work
        result = select_format_with_fallback(
            requested_format="fp8_e4m3",
            gpu_generation=GPUGeneration.HOPPER,
            fallback_format="int8",
        )
        assert result == "fp8_e4m3"


class TestQuantizationSelectorThreadSafety:
    """Tests for format selector thread safety."""

    def test_concurrent_format_selection(self) -> None:
        """Selector handles concurrent access."""
        import threading

        from layerzero.quant.format_selection import (
            QuantFormatSelector,
            QuantizationConfig,
        )

        selector = QuantFormatSelector()
        errors: list[Exception] = []

        def worker() -> None:
            try:
                for _ in range(100):
                    config = QuantizationConfig(
                        format_name="int8",
                        is_enabled=True,
                    )
                    kernels = [{"id": "k1", "supports_int8": True}]
                    selector.filter_kernels_for_format(kernels, config)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
