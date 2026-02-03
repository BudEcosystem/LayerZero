"""Quantization scale handling.

This module provides:
- Scale granularity definitions
- QuantScales dataclass
- Scale computation utilities
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, unique
from typing import Any

import torch

logger = logging.getLogger(__name__)


@unique
class ScaleGranularity(str, Enum):
    """Scale granularity for quantization.

    PER_TENSOR: Single scale for entire tensor.
    PER_CHANNEL: One scale per channel.
    BLOCKWISE: One scale per block of elements.
    """

    PER_TENSOR = "per_tensor"
    PER_CHANNEL = "per_channel"
    BLOCKWISE = "blockwise"


@dataclass
class QuantScales:
    """Quantization scales container.

    Attributes:
        scale: Scale tensor.
        zero_point: Zero point tensor (for asymmetric quantization).
        granularity: Scale granularity level.
        channel_axis: Axis for per-channel quantization.
        block_size: Block size for blockwise quantization.
    """

    scale: torch.Tensor
    zero_point: torch.Tensor | None = None
    granularity: ScaleGranularity = ScaleGranularity.PER_TENSOR
    channel_axis: int | None = None
    block_size: int | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation.
        """
        d = {
            "scale": self.scale.tolist(),
            "granularity": self.granularity.value,
        }

        if self.zero_point is not None:
            d["zero_point"] = self.zero_point.tolist()

        if self.channel_axis is not None:
            d["channel_axis"] = self.channel_axis

        if self.block_size is not None:
            d["block_size"] = self.block_size

        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "QuantScales":
        """Deserialize from dictionary.

        Args:
            d: Dictionary representation.

        Returns:
            QuantScales instance.
        """
        scale = torch.tensor(d["scale"])
        zero_point = None
        if "zero_point" in d and d["zero_point"] is not None:
            zero_point = torch.tensor(d["zero_point"])

        granularity = ScaleGranularity(d.get("granularity", "per_tensor"))

        return cls(
            scale=scale,
            zero_point=zero_point,
            granularity=granularity,
            channel_axis=d.get("channel_axis"),
            block_size=d.get("block_size"),
        )


def compute_scale(
    tensor: torch.Tensor,
    target_dtype: str = "int8",
    per_tensor: bool = True,
    channel_axis: int | None = None,
    symmetric: bool = True,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Compute quantization scale from tensor.

    Args:
        tensor: Input tensor to compute scale from.
        target_dtype: Target quantization dtype.
        per_tensor: Use per-tensor scale (vs per-channel).
        channel_axis: Axis for per-channel quantization.
        symmetric: Use symmetric quantization (zero_point = 0).

    Returns:
        Tuple of (scale, zero_point).
    """
    # Get min/max values for target dtype
    if target_dtype == "int8":
        qmin, qmax = -128, 127
    elif target_dtype in ("fp8_e4m3", "fp8_e5m2"):
        qmin, qmax = -448, 448  # Approximate
    else:
        qmin, qmax = -128, 127  # Default

    if per_tensor:
        # Per-tensor scale
        min_val = tensor.min()
        max_val = tensor.max()

        if symmetric:
            abs_max = torch.max(torch.abs(min_val), torch.abs(max_val))
            scale = abs_max / max(abs(qmin), abs(qmax))
            zero_point = None
        else:
            scale = (max_val - min_val) / (qmax - qmin)
            zero_point = torch.round(-min_val / scale) + qmin
            zero_point = torch.clamp(zero_point, qmin, qmax)

        scale = scale.unsqueeze(0)
        if zero_point is not None:
            zero_point = zero_point.unsqueeze(0).to(torch.int8)

    else:
        # Per-channel scale
        if channel_axis is None:
            channel_axis = 0

        # Move channel axis to first position
        tensor_t = tensor.transpose(0, channel_axis)
        flat = tensor_t.reshape(tensor_t.shape[0], -1)

        min_vals = flat.min(dim=1).values
        max_vals = flat.max(dim=1).values

        if symmetric:
            abs_max = torch.max(torch.abs(min_vals), torch.abs(max_vals))
            scale = abs_max / max(abs(qmin), abs(qmax))
            zero_point = None
        else:
            scale = (max_vals - min_vals) / (qmax - qmin)
            zero_point = torch.round(-min_vals / scale) + qmin
            zero_point = torch.clamp(zero_point, qmin, qmax).to(torch.int8)

    # Handle zero scale
    scale = torch.where(scale == 0, torch.ones_like(scale), scale)

    return scale, zero_point
