"""Tests for quantization scales handling."""
from __future__ import annotations

import pytest
import torch


class TestQuantizationScales:
    """Tests for quantization scale handling."""

    def test_per_tensor_scales(self) -> None:
        """Per-tensor scales handled."""
        from layerzero.quant.scales import (
            QuantScales,
            ScaleGranularity,
        )

        scales = QuantScales(
            scale=torch.tensor([0.1]),
            zero_point=torch.tensor([0]),
            granularity=ScaleGranularity.PER_TENSOR,
        )

        assert scales.granularity == ScaleGranularity.PER_TENSOR
        assert scales.scale.shape == (1,)

    def test_per_channel_scales(self) -> None:
        """Per-channel scales handled."""
        from layerzero.quant.scales import (
            QuantScales,
            ScaleGranularity,
        )

        num_channels = 128
        scales = QuantScales(
            scale=torch.randn(num_channels),
            zero_point=torch.zeros(num_channels, dtype=torch.int8),
            granularity=ScaleGranularity.PER_CHANNEL,
            channel_axis=0,
        )

        assert scales.granularity == ScaleGranularity.PER_CHANNEL
        assert scales.scale.shape == (num_channels,)
        assert scales.channel_axis == 0

    def test_blockwise_scales(self) -> None:
        """Blockwise scales handled."""
        from layerzero.quant.scales import (
            QuantScales,
            ScaleGranularity,
        )

        block_size = 32
        num_blocks = 16
        scales = QuantScales(
            scale=torch.randn(num_blocks),
            zero_point=None,
            granularity=ScaleGranularity.BLOCKWISE,
            block_size=block_size,
        )

        assert scales.granularity == ScaleGranularity.BLOCKWISE
        assert scales.block_size == block_size
        assert scales.scale.shape == (num_blocks,)

    def test_scales_serialization(self) -> None:
        """Scales can be serialized to dict."""
        from layerzero.quant.scales import (
            QuantScales,
            ScaleGranularity,
        )

        scales = QuantScales(
            scale=torch.tensor([0.1, 0.2]),
            zero_point=torch.tensor([0, 0]),
            granularity=ScaleGranularity.PER_CHANNEL,
            channel_axis=1,
        )

        d = scales.to_dict()

        assert d["granularity"] == "per_channel"
        assert d["channel_axis"] == 1
        assert "scale" in d
        assert "zero_point" in d

    def test_scales_from_dict(self) -> None:
        """Scales can be deserialized from dict."""
        from layerzero.quant.scales import (
            QuantScales,
            ScaleGranularity,
        )

        d = {
            "scale": [0.1, 0.2],
            "zero_point": [0, 0],
            "granularity": "per_channel",
            "channel_axis": 1,
        }

        scales = QuantScales.from_dict(d)

        assert scales.granularity == ScaleGranularity.PER_CHANNEL
        assert scales.channel_axis == 1
        assert torch.allclose(scales.scale, torch.tensor([0.1, 0.2]))

    def test_scale_computation_per_tensor(self) -> None:
        """Compute per-tensor scale from tensor."""
        from layerzero.quant.scales import compute_scale

        tensor = torch.randn(32, 64) * 10
        scale, zero_point = compute_scale(
            tensor,
            target_dtype="int8",
            per_tensor=True,
        )

        assert scale.shape == (1,)
        assert scale.item() > 0

    def test_scale_computation_per_channel(self) -> None:
        """Compute per-channel scale from tensor."""
        from layerzero.quant.scales import compute_scale

        tensor = torch.randn(32, 64) * 10
        scale, zero_point = compute_scale(
            tensor,
            target_dtype="int8",
            per_tensor=False,
            channel_axis=0,
        )

        assert scale.shape == (32,)

    def test_symmetric_vs_asymmetric_quantization(self) -> None:
        """Symmetric vs asymmetric quantization."""
        from layerzero.quant.scales import compute_scale

        tensor = torch.randn(32, 64)

        # Symmetric (zero_point = 0)
        scale_sym, zp_sym = compute_scale(tensor, symmetric=True)
        assert zp_sym is None or torch.all(zp_sym == 0)

        # Asymmetric (zero_point can be non-zero)
        scale_asym, zp_asym = compute_scale(tensor, symmetric=False)
        # zp_asym may or may not be zero depending on data


class TestQuantizationContext:
    """Tests for quantization in SelectionContext."""

    def test_quant_format_in_context(self) -> None:
        """SelectionContext includes quant_format."""
        from layerzero.quant.format_selection import QuantizationConfig

        config = QuantizationConfig(
            format_name="int8",
            is_enabled=True,
        )

        assert config.format_name == "int8"
        assert config.is_enabled is True

    def test_scale_granularity_in_context(self) -> None:
        """Scale granularity in context."""
        from layerzero.quant.format_selection import QuantizationConfig
        from layerzero.quant.scales import ScaleGranularity

        config = QuantizationConfig(
            format_name="fp8_e4m3",
            is_enabled=True,
            scale_granularity=ScaleGranularity.PER_TENSOR,
        )

        assert config.scale_granularity == ScaleGranularity.PER_TENSOR

    def test_quant_config_disabled(self) -> None:
        """Quantization can be disabled."""
        from layerzero.quant.format_selection import QuantizationConfig

        config = QuantizationConfig(
            format_name=None,
            is_enabled=False,
        )

        assert config.is_enabled is False
        assert config.format_name is None
