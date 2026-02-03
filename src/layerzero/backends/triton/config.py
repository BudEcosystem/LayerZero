"""
LayerZero Triton Grid/Block Configuration

Validation for Triton kernel grid and block configurations.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Union

from layerzero.reasons import (
    CUDA_BLOCK_LIMIT_EXCEEDED,
    CUDA_GRID_DIM_EXCEEDED,
    Reason,
    ReasonCategory,
)

logger = logging.getLogger(__name__)

# Standard CUDA limits (conservative defaults)
# Grid dimensions: (x, y, z) max values
# Modern GPUs support larger grids, but these are safe defaults
DEFAULT_MAX_GRID: tuple[int, int, int] = (2**31 - 1, 65535, 65535)

# Maximum threads per block
DEFAULT_MAX_THREADS: int = 1024


@dataclass
class GridConfig:
    """Triton grid configuration.

    Describes the grid dimensions for kernel launch.
    Each dimension can be a static integer or a callable that
    computes the dimension based on kernel metadata.

    Attributes:
        x: X dimension of grid (required, can be int or callable)
        y: Y dimension of grid (default: 1)
        z: Z dimension of grid (default: 1)
    """

    x: Union[int, Callable[[dict], int]]
    y: int = field(default=1)
    z: int = field(default=1)

    def __post_init__(self) -> None:
        """Validate dimensions after initialization."""
        # y and z must be positive integers
        if isinstance(self.y, int) and self.y < 0:
            raise ValueError(f"Grid y dimension must be non-negative, got {self.y}")
        if isinstance(self.z, int) and self.z < 0:
            raise ValueError(f"Grid z dimension must be non-negative, got {self.z}")

    def compute(self, meta: dict) -> tuple[int, int, int]:
        """Compute grid dimensions from metadata.

        Args:
            meta: Kernel metadata dictionary with constexpr values.

        Returns:
            Tuple of (x, y, z) grid dimensions.
        """
        x_val = self.x(meta) if callable(self.x) else self.x
        return (x_val, self.y, self.z)


def validate_grid_config(
    config: GridConfig,
    max_grid: tuple[int, int, int] = DEFAULT_MAX_GRID,
) -> list[Reason]:
    """Validate grid configuration against hardware limits.

    Validates static dimensions. Callable dimensions are validated
    at runtime, not registration time.

    Args:
        config: Grid configuration to validate.
        max_grid: Maximum grid dimensions (x, y, z).

    Returns:
        Empty list if valid, else list of failure reasons.
    """
    reasons: list[Reason] = []

    # Check x dimension (skip if callable)
    if isinstance(config.x, int):
        if config.x <= 0:
            reasons.append(Reason(
                code=CUDA_GRID_DIM_EXCEEDED,
                message=f"Grid x dimension must be positive, got {config.x}",
                category=ReasonCategory.CUDA,
            ))
        elif config.x > max_grid[0]:
            reasons.append(Reason(
                code=CUDA_GRID_DIM_EXCEEDED,
                message=f"Grid x={config.x} exceeds max {max_grid[0]}",
                category=ReasonCategory.CUDA,
            ))

    # Check y dimension
    if config.y <= 0:
        reasons.append(Reason(
            code=CUDA_GRID_DIM_EXCEEDED,
            message=f"Grid y dimension must be positive, got {config.y}",
            category=ReasonCategory.CUDA,
        ))
    elif config.y > max_grid[1]:
        reasons.append(Reason(
            code=CUDA_GRID_DIM_EXCEEDED,
            message=f"Grid y={config.y} exceeds max {max_grid[1]}",
            category=ReasonCategory.CUDA,
        ))

    # Check z dimension
    if config.z <= 0:
        reasons.append(Reason(
            code=CUDA_GRID_DIM_EXCEEDED,
            message=f"Grid z dimension must be positive, got {config.z}",
            category=ReasonCategory.CUDA,
        ))
    elif config.z > max_grid[2]:
        reasons.append(Reason(
            code=CUDA_GRID_DIM_EXCEEDED,
            message=f"Grid z={config.z} exceeds max {max_grid[2]}",
            category=ReasonCategory.CUDA,
        ))

    return reasons


def validate_block_config(
    block_size: int,
    max_threads: int = DEFAULT_MAX_THREADS,
) -> list[Reason]:
    """Validate block size against hardware limits.

    Args:
        block_size: Number of threads per block.
        max_threads: Maximum threads per block (default: 1024).

    Returns:
        Empty list if valid, else list of failure reasons.
    """
    reasons: list[Reason] = []

    if block_size <= 0:
        reasons.append(Reason(
            code=CUDA_BLOCK_LIMIT_EXCEEDED,
            message=f"Block size must be positive, got {block_size}",
            category=ReasonCategory.CUDA,
        ))
    elif block_size > max_threads:
        reasons.append(Reason(
            code=CUDA_BLOCK_LIMIT_EXCEEDED,
            message=f"Block size {block_size} exceeds max {max_threads}",
            category=ReasonCategory.CUDA,
        ))

    return reasons
