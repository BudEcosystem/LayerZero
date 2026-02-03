"""
Temperature Scaling Utilities

Applies temperature scaling to logits for controlling sampling randomness.
"""
from __future__ import annotations

import torch
from typing import Optional


def apply_temperature(
    logits: torch.Tensor,
    temperature: float,
    *,
    min_temperature: float = 1e-8,
) -> torch.Tensor:
    """Apply temperature scaling to logits.

    Temperature controls the randomness of sampling:
    - T < 1.0: More deterministic (sharper distribution)
    - T = 1.0: No change
    - T > 1.0: More random (flatter distribution)

    Args:
        logits: Input logits tensor of shape (..., vocab_size).
        temperature: Temperature value. Must be positive.
        min_temperature: Minimum temperature to prevent division by zero.

    Returns:
        Scaled logits tensor of same shape.

    Raises:
        ValueError: If temperature is non-positive.
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    # Clamp to minimum to prevent numerical issues
    temp = max(temperature, min_temperature)

    return logits / temp


def apply_temperature_inplace(
    logits: torch.Tensor,
    temperature: float,
    *,
    min_temperature: float = 1e-8,
) -> torch.Tensor:
    """Apply temperature scaling in-place.

    Same as apply_temperature but modifies the input tensor.

    Args:
        logits: Input logits tensor to modify in-place.
        temperature: Temperature value.
        min_temperature: Minimum temperature.

    Returns:
        The same tensor (modified in-place).
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    temp = max(temperature, min_temperature)
    logits.div_(temp)
    return logits
