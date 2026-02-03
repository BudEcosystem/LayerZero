"""Tests for WarmupConfig dataclass."""
from __future__ import annotations

from pathlib import Path

import pytest


class TestWarmupConfig:
    """Tests for WarmupConfig dataclass."""

    def test_config_defaults(self) -> None:
        """Test default config values."""
        from layerzero.warmup.config import WarmupConfig

        config = WarmupConfig()

        assert config.enabled is True
        assert config.blocking is True
        assert config.timeout_ms == 30000.0
        assert config.max_concurrent_jit == 2
        assert config.cache_dir is None
        assert config.persist_cache is True
        assert config.background_compile is True
        assert config.critical_shapes_first is True

    def test_config_custom_values(self, temp_cache_dir: Path) -> None:
        """Test config with custom values."""
        from layerzero.warmup.config import WarmupConfig

        config = WarmupConfig(
            enabled=True,
            blocking=False,
            timeout_ms=5000.0,
            max_concurrent_jit=4,
            cache_dir=temp_cache_dir,
            persist_cache=True,
            background_compile=False,
            critical_shapes_first=False,
        )

        assert config.blocking is False
        assert config.timeout_ms == 5000.0
        assert config.max_concurrent_jit == 4
        assert config.cache_dir == temp_cache_dir

    def test_config_immutable(self) -> None:
        """Test config is frozen (immutable)."""
        from layerzero.warmup.config import WarmupConfig

        config = WarmupConfig()

        with pytest.raises((AttributeError, TypeError)):
            config.enabled = False  # type: ignore

    def test_config_hashable(self) -> None:
        """Test config is hashable for use as dict key."""
        from layerzero.warmup.config import WarmupConfig

        config = WarmupConfig()
        hash_value = hash(config)
        assert isinstance(hash_value, int)

    def test_config_equality(self) -> None:
        """Test config equality comparison."""
        from layerzero.warmup.config import WarmupConfig

        config1 = WarmupConfig(timeout_ms=5000.0)
        config2 = WarmupConfig(timeout_ms=5000.0)
        config3 = WarmupConfig(timeout_ms=10000.0)

        assert config1 == config2
        assert config1 != config3

    def test_config_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test config can be loaded from environment variables."""
        from layerzero.warmup.config import WarmupConfig

        monkeypatch.setenv("LAYERZERO_JIT_CACHE_DIR", "/tmp/lz_cache")
        monkeypatch.setenv("LAYERZERO_WARMUP_TIMEOUT_MS", "10000")

        config = WarmupConfig.from_env()

        assert config.cache_dir == Path("/tmp/lz_cache")
        assert config.timeout_ms == 10000.0

    def test_config_validation_timeout(self) -> None:
        """Test config validates timeout is positive."""
        from layerzero.warmup.config import WarmupConfig

        with pytest.raises(ValueError, match="timeout_ms must be positive"):
            WarmupConfig(timeout_ms=-1.0)

    def test_config_validation_concurrent(self) -> None:
        """Test config validates max_concurrent_jit is positive."""
        from layerzero.warmup.config import WarmupConfig

        with pytest.raises(ValueError, match="max_concurrent_jit must be positive"):
            WarmupConfig(max_concurrent_jit=0)


class TestWarmupReport:
    """Tests for WarmupReport dataclass."""

    def test_report_creation(self) -> None:
        """Test creating warmup report."""
        from layerzero.warmup.config import WarmupReport

        report = WarmupReport(
            total_shapes=10,
            compiled_shapes=8,
            cached_shapes=2,
            failed_shapes=0,
            total_time_ms=5000.0,
            errors=[],
        )

        assert report.total_shapes == 10
        assert report.compiled_shapes == 8
        assert report.success_rate == 1.0

    def test_report_success_rate_with_failures(self) -> None:
        """Test success rate calculation with failures."""
        from layerzero.warmup.config import WarmupReport

        report = WarmupReport(
            total_shapes=10,
            compiled_shapes=6,
            cached_shapes=2,
            failed_shapes=2,
            total_time_ms=5000.0,
            errors=["Shape 1 timed out", "Shape 2 failed"],
        )

        assert report.success_rate == 0.8  # 8/10 successful

    def test_report_str_representation(self) -> None:
        """Test string representation of report."""
        from layerzero.warmup.config import WarmupReport

        report = WarmupReport(
            total_shapes=10,
            compiled_shapes=8,
            cached_shapes=2,
            failed_shapes=0,
            total_time_ms=5000.0,
            errors=[],
        )

        report_str = str(report)
        assert "10" in report_str  # total shapes
        assert "5000" in report_str or "5.0" in report_str  # time


class TestShapeWarmupResult:
    """Tests for ShapeWarmupResult dataclass."""

    def test_result_success(self) -> None:
        """Test successful warmup result."""
        from layerzero.warmup.config import ShapeWarmupResult

        result = ShapeWarmupResult(
            shape_key="attention.causal:fp16:1:1024:128:32",
            success=True,
            cached=False,
            compile_time_ms=150.0,
            error=None,
        )

        assert result.success is True
        assert result.cached is False
        assert result.error is None

    def test_result_cached(self) -> None:
        """Test cached warmup result."""
        from layerzero.warmup.config import ShapeWarmupResult

        result = ShapeWarmupResult(
            shape_key="attention.causal:fp16:1:1024:128:32",
            success=True,
            cached=True,
            compile_time_ms=0.1,
            error=None,
        )

        assert result.success is True
        assert result.cached is True

    def test_result_failure(self) -> None:
        """Test failed warmup result."""
        from layerzero.warmup.config import ShapeWarmupResult

        result = ShapeWarmupResult(
            shape_key="attention.causal:fp16:1:1024:128:32",
            success=False,
            cached=False,
            compile_time_ms=30000.0,
            error="JIT compilation timed out",
        )

        assert result.success is False
        assert result.error is not None
