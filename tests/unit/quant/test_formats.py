"""Tests for quantization format definitions."""
from __future__ import annotations

import pytest


class TestQuantizationFormats:
    """Tests for quantization format definitions."""

    def test_int8_format_definition(self) -> None:
        """INT8 format defined correctly."""
        from layerzero.quant.formats import QuantFormat, INT8_FORMAT

        assert INT8_FORMAT.name == "int8"
        assert INT8_FORMAT.bits == 8
        assert INT8_FORMAT.is_signed is True
        assert INT8_FORMAT.mantissa_bits == 0  # Integer
        assert INT8_FORMAT.exponent_bits == 0  # Integer

    def test_fp8_e4m3_format_definition(self) -> None:
        """FP8 E4M3 format defined correctly."""
        from layerzero.quant.formats import QuantFormat, FP8_E4M3_FORMAT

        assert FP8_E4M3_FORMAT.name == "fp8_e4m3"
        assert FP8_E4M3_FORMAT.bits == 8
        assert FP8_E4M3_FORMAT.mantissa_bits == 3
        assert FP8_E4M3_FORMAT.exponent_bits == 4
        assert FP8_E4M3_FORMAT.is_floating_point is True

    def test_fp8_e5m2_format_definition(self) -> None:
        """FP8 E5M2 format defined correctly."""
        from layerzero.quant.formats import QuantFormat, FP8_E5M2_FORMAT

        assert FP8_E5M2_FORMAT.name == "fp8_e5m2"
        assert FP8_E5M2_FORMAT.bits == 8
        assert FP8_E5M2_FORMAT.mantissa_bits == 2
        assert FP8_E5M2_FORMAT.exponent_bits == 5
        assert FP8_E5M2_FORMAT.is_floating_point is True

    def test_mxfp4_format_definition(self) -> None:
        """MXFP4 format defined correctly."""
        from layerzero.quant.formats import QuantFormat, MXFP4_FORMAT

        assert MXFP4_FORMAT.name == "mxfp4"
        assert MXFP4_FORMAT.bits == 4
        assert MXFP4_FORMAT.is_block_scaled is True
        assert MXFP4_FORMAT.block_size == 32

    def test_nvfp4_format_definition(self) -> None:
        """NVFP4 format defined correctly."""
        from layerzero.quant.formats import QuantFormat, NVFP4_FORMAT

        assert NVFP4_FORMAT.name == "nvfp4"
        assert NVFP4_FORMAT.bits == 4
        assert NVFP4_FORMAT.is_block_scaled is True

    def test_format_registry_contains_all(self) -> None:
        """Format registry contains all formats."""
        from layerzero.quant.formats import (
            FORMAT_REGISTRY,
            INT8_FORMAT,
            FP8_E4M3_FORMAT,
            FP8_E5M2_FORMAT,
            MXFP4_FORMAT,
            NVFP4_FORMAT,
        )

        assert "int8" in FORMAT_REGISTRY
        assert "fp8_e4m3" in FORMAT_REGISTRY
        assert "fp8_e5m2" in FORMAT_REGISTRY
        assert "mxfp4" in FORMAT_REGISTRY
        assert "nvfp4" in FORMAT_REGISTRY

    def test_format_lookup(self) -> None:
        """Format lookup by name works."""
        from layerzero.quant.formats import get_format

        fmt = get_format("fp8_e4m3")
        assert fmt is not None
        assert fmt.name == "fp8_e4m3"

    def test_format_lookup_not_found(self) -> None:
        """Format lookup returns None for unknown."""
        from layerzero.quant.formats import get_format

        fmt = get_format("unknown_format")
        assert fmt is None


class TestQuantizationHardwareSupport:
    """Tests for quantization hardware support detection."""

    def test_int8_supported_ampere(self, ampere_device) -> None:
        """INT8 supported on Ampere+."""
        from layerzero.quant.formats import is_format_supported
        from layerzero.device import GPUGeneration

        assert is_format_supported("int8", GPUGeneration.AMPERE) is True
        assert is_format_supported("int8", GPUGeneration.HOPPER) is True
        assert is_format_supported("int8", GPUGeneration.BLACKWELL) is True

    def test_int8_not_supported_turing(self) -> None:
        """INT8 tensor cores limited on Turing."""
        from layerzero.quant.formats import is_format_supported
        from layerzero.device import GPUGeneration

        # Turing has limited INT8 tensor core support
        assert is_format_supported("int8", GPUGeneration.TURING) is True

    def test_fp8_supported_hopper(self, hopper_device) -> None:
        """FP8 supported on Hopper+."""
        from layerzero.quant.formats import is_format_supported
        from layerzero.device import GPUGeneration

        assert is_format_supported("fp8_e4m3", GPUGeneration.HOPPER) is True
        assert is_format_supported("fp8_e5m2", GPUGeneration.HOPPER) is True
        assert is_format_supported("fp8_e4m3", GPUGeneration.BLACKWELL) is True

    def test_fp8_not_supported_ampere(self, ampere_device) -> None:
        """FP8 not supported on Ampere."""
        from layerzero.quant.formats import is_format_supported
        from layerzero.device import GPUGeneration

        assert is_format_supported("fp8_e4m3", GPUGeneration.AMPERE) is False
        assert is_format_supported("fp8_e5m2", GPUGeneration.AMPERE) is False

    def test_mxfp4_supported_blackwell(self, blackwell_device) -> None:
        """MXFP4 supported on Blackwell."""
        from layerzero.quant.formats import is_format_supported
        from layerzero.device import GPUGeneration

        assert is_format_supported("mxfp4", GPUGeneration.BLACKWELL) is True

    def test_mxfp4_not_supported_hopper(self, hopper_device) -> None:
        """MXFP4 not supported on Hopper."""
        from layerzero.quant.formats import is_format_supported
        from layerzero.device import GPUGeneration

        assert is_format_supported("mxfp4", GPUGeneration.HOPPER) is False

    def test_nvfp4_supported_blackwell(self, blackwell_device) -> None:
        """NVFP4 supported on Blackwell."""
        from layerzero.quant.formats import is_format_supported
        from layerzero.device import GPUGeneration

        assert is_format_supported("nvfp4", GPUGeneration.BLACKWELL) is True

    def test_nvfp4_not_supported_hopper(self, hopper_device) -> None:
        """NVFP4 not supported on Hopper."""
        from layerzero.quant.formats import is_format_supported
        from layerzero.device import GPUGeneration

        assert is_format_supported("nvfp4", GPUGeneration.HOPPER) is False

    def test_get_supported_formats_for_generation(self) -> None:
        """Get all supported formats for a generation."""
        from layerzero.quant.formats import get_supported_formats
        from layerzero.device import GPUGeneration

        # Ampere: INT8 only
        ampere_formats = get_supported_formats(GPUGeneration.AMPERE)
        assert "int8" in ampere_formats
        assert "fp8_e4m3" not in ampere_formats

        # Hopper: INT8, FP8
        hopper_formats = get_supported_formats(GPUGeneration.HOPPER)
        assert "int8" in hopper_formats
        assert "fp8_e4m3" in hopper_formats
        assert "mxfp4" not in hopper_formats

        # Blackwell: All formats
        blackwell_formats = get_supported_formats(GPUGeneration.BLACKWELL)
        assert "int8" in blackwell_formats
        assert "fp8_e4m3" in blackwell_formats
        assert "mxfp4" in blackwell_formats
        assert "nvfp4" in blackwell_formats
