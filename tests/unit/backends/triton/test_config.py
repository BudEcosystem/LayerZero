"""Tests for Triton grid/block configuration validation."""
from __future__ import annotations

import pytest

from layerzero.backends.triton.config import (
    GridConfig,
    validate_grid_config,
    validate_block_config,
    DEFAULT_MAX_GRID,
    DEFAULT_MAX_THREADS,
)
from layerzero.reasons import (
    CUDA_GRID_DIM_EXCEEDED,
    CUDA_BLOCK_LIMIT_EXCEEDED,
)


class TestGridConfig:
    """Test GridConfig dataclass."""

    def test_grid_config_creation(self) -> None:
        """GridConfig can be created."""
        config = GridConfig(x=128)
        assert config.x == 128
        assert config.y == 1
        assert config.z == 1

    def test_grid_config_with_all_dims(self) -> None:
        """GridConfig with all dimensions."""
        config = GridConfig(x=128, y=64, z=32)
        assert config.x == 128
        assert config.y == 64
        assert config.z == 32

    def test_grid_config_callable_x(self) -> None:
        """GridConfig with callable x dimension."""
        config = GridConfig(x=lambda n: n // 256)
        assert callable(config.x)


class TestGridValidation:
    """Test grid configuration validation."""

    def test_valid_grid_accepted(self) -> None:
        """Valid grid configuration is accepted."""
        config = GridConfig(x=1024, y=1, z=1)
        reasons = validate_grid_config(config, DEFAULT_MAX_GRID)
        assert len(reasons) == 0

    def test_grid_x_too_large_rejected(self) -> None:
        """Grid x dimension too large is rejected."""
        config = GridConfig(x=DEFAULT_MAX_GRID[0] + 1, y=1, z=1)
        reasons = validate_grid_config(config, DEFAULT_MAX_GRID)
        assert len(reasons) == 1
        assert CUDA_GRID_DIM_EXCEEDED in reasons[0].code

    def test_grid_y_too_large_rejected(self) -> None:
        """Grid y dimension too large is rejected."""
        config = GridConfig(x=1, y=DEFAULT_MAX_GRID[1] + 1, z=1)
        reasons = validate_grid_config(config, DEFAULT_MAX_GRID)
        assert len(reasons) == 1
        assert CUDA_GRID_DIM_EXCEEDED in reasons[0].code

    def test_grid_z_too_large_rejected(self) -> None:
        """Grid z dimension too large is rejected."""
        config = GridConfig(x=1, y=1, z=DEFAULT_MAX_GRID[2] + 1)
        reasons = validate_grid_config(config, DEFAULT_MAX_GRID)
        assert len(reasons) == 1
        assert CUDA_GRID_DIM_EXCEEDED in reasons[0].code

    def test_grid_zero_rejected(self) -> None:
        """Grid dimension of zero is rejected."""
        config = GridConfig(x=0, y=1, z=1)
        reasons = validate_grid_config(config, DEFAULT_MAX_GRID)
        assert len(reasons) >= 1

    def test_grid_callable_skips_validation(self) -> None:
        """Grid with callable dimension skips static validation."""
        config = GridConfig(x=lambda n: n, y=1, z=1)
        reasons = validate_grid_config(config, DEFAULT_MAX_GRID)
        # Cannot validate callable at registration time
        assert len(reasons) == 0


class TestBlockValidation:
    """Test block configuration validation."""

    def test_valid_block_accepted(self) -> None:
        """Valid block size is accepted."""
        reasons = validate_block_config(256, DEFAULT_MAX_THREADS)
        assert len(reasons) == 0

    def test_block_1024_accepted(self) -> None:
        """Block size of 1024 is accepted."""
        reasons = validate_block_config(1024, DEFAULT_MAX_THREADS)
        assert len(reasons) == 0

    def test_block_too_large_rejected(self) -> None:
        """Block size too large is rejected."""
        reasons = validate_block_config(DEFAULT_MAX_THREADS + 1, DEFAULT_MAX_THREADS)
        assert len(reasons) == 1
        assert CUDA_BLOCK_LIMIT_EXCEEDED in reasons[0].code

    def test_block_zero_rejected(self) -> None:
        """Block size of zero is rejected."""
        reasons = validate_block_config(0, DEFAULT_MAX_THREADS)
        assert len(reasons) >= 1

    def test_block_negative_rejected(self) -> None:
        """Negative block size is rejected."""
        reasons = validate_block_config(-1, DEFAULT_MAX_THREADS)
        assert len(reasons) >= 1


class TestDefaultLimits:
    """Test default hardware limit constants."""

    def test_default_max_grid(self) -> None:
        """DEFAULT_MAX_GRID has reasonable values."""
        assert len(DEFAULT_MAX_GRID) == 3
        assert all(x > 0 for x in DEFAULT_MAX_GRID)
        # Standard CUDA limits
        assert DEFAULT_MAX_GRID[0] >= 2**16 - 1

    def test_default_max_threads(self) -> None:
        """DEFAULT_MAX_THREADS is reasonable."""
        assert DEFAULT_MAX_THREADS >= 1024
