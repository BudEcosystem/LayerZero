"""
Warmup configuration and result dataclasses.

This module provides:
- WarmupConfig: Configuration for JIT warmup behavior
- WarmupReport: Summary of warmup execution
- ShapeWarmupResult: Result for individual shape warmup
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar


@dataclass(frozen=True)
class WarmupConfig:
    """Configuration for JIT warmup protocol.

    Attributes:
        enabled: Whether warmup is enabled.
        blocking: If True, warmup blocks until critical shapes compiled.
        timeout_ms: Maximum time per shape in milliseconds.
        max_concurrent_jit: Maximum concurrent JIT compilations.
        cache_dir: Directory for JIT cache. If None, uses default.
        persist_cache: Whether to save cache to disk.
        background_compile: Whether to compile non-critical shapes in background.
        critical_shapes_first: Whether to compile critical shapes first.

    Example:
        config = WarmupConfig(
            timeout_ms=30000.0,
            max_concurrent_jit=2,
            cache_dir=Path("/tmp/lz_cache"),
        )
    """

    DEFAULT_CACHE_DIR: ClassVar[str] = "~/.cache/layerzero/jit"
    DEFAULT_TIMEOUT_MS: ClassVar[float] = 30000.0
    DEFAULT_MAX_CONCURRENT: ClassVar[int] = 2

    enabled: bool = True
    blocking: bool = True
    timeout_ms: float = 30000.0
    max_concurrent_jit: int = 2
    cache_dir: Path | None = None
    persist_cache: bool = True
    background_compile: bool = True
    critical_shapes_first: bool = True

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        if self.max_concurrent_jit <= 0:
            raise ValueError("max_concurrent_jit must be positive")

    @classmethod
    def from_env(cls) -> "WarmupConfig":
        """Create config from environment variables.

        Environment variables:
            LAYERZERO_JIT_CACHE_DIR: Cache directory path
            LAYERZERO_WARMUP_TIMEOUT_MS: Timeout in milliseconds
            LAYERZERO_WARMUP_ENABLED: "0" or "false" to disable
            LAYERZERO_WARMUP_BLOCKING: "0" or "false" for non-blocking
            LAYERZERO_WARMUP_MAX_CONCURRENT: Max concurrent compilations

        Returns:
            WarmupConfig with values from environment.
        """
        cache_dir_str = os.environ.get("LAYERZERO_JIT_CACHE_DIR")
        cache_dir = Path(cache_dir_str) if cache_dir_str else None

        timeout_str = os.environ.get("LAYERZERO_WARMUP_TIMEOUT_MS")
        timeout_ms = float(timeout_str) if timeout_str else cls.DEFAULT_TIMEOUT_MS

        enabled_str = os.environ.get("LAYERZERO_WARMUP_ENABLED", "1")
        enabled = enabled_str.lower() not in ("0", "false", "no")

        blocking_str = os.environ.get("LAYERZERO_WARMUP_BLOCKING", "1")
        blocking = blocking_str.lower() not in ("0", "false", "no")

        max_concurrent_str = os.environ.get("LAYERZERO_WARMUP_MAX_CONCURRENT")
        max_concurrent = (
            int(max_concurrent_str) if max_concurrent_str else cls.DEFAULT_MAX_CONCURRENT
        )

        return cls(
            enabled=enabled,
            blocking=blocking,
            timeout_ms=timeout_ms,
            max_concurrent_jit=max_concurrent,
            cache_dir=cache_dir,
        )

    def get_cache_dir(self) -> Path:
        """Get resolved cache directory path.

        Returns:
            Expanded cache directory path.
        """
        if self.cache_dir is not None:
            return self.cache_dir.expanduser()
        return Path(self.DEFAULT_CACHE_DIR).expanduser()


@dataclass
class WarmupReport:
    """Summary of warmup execution.

    Attributes:
        total_shapes: Total number of shapes to warmup.
        compiled_shapes: Number of shapes successfully compiled.
        cached_shapes: Number of shapes found in cache (no compile needed).
        failed_shapes: Number of shapes that failed to compile.
        total_time_ms: Total warmup time in milliseconds.
        errors: List of error messages for failed shapes.
        shape_results: Per-shape results (optional, for detailed analysis).

    Example:
        if report.success_rate < 1.0:
            print(f"Warmup failures: {report.errors}")
    """

    total_shapes: int
    compiled_shapes: int
    cached_shapes: int
    failed_shapes: int
    total_time_ms: float
    errors: list[str] = field(default_factory=list)
    shape_results: list["ShapeWarmupResult"] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        """Calculate success rate (compiled + cached / total).

        Returns:
            Success rate as float between 0.0 and 1.0.
        """
        if self.total_shapes == 0:
            return 1.0
        successful = self.compiled_shapes + self.cached_shapes
        return successful / self.total_shapes

    @property
    def is_complete(self) -> bool:
        """Check if all shapes were warmed up successfully.

        Returns:
            True if no failures occurred.
        """
        return self.failed_shapes == 0

    def __str__(self) -> str:
        """Human-readable summary."""
        return (
            f"WarmupReport("
            f"total={self.total_shapes}, "
            f"compiled={self.compiled_shapes}, "
            f"cached={self.cached_shapes}, "
            f"failed={self.failed_shapes}, "
            f"time={self.total_time_ms:.1f}ms, "
            f"success_rate={self.success_rate:.1%})"
        )


@dataclass
class ShapeWarmupResult:
    """Result for individual shape warmup.

    Attributes:
        shape_key: Unique key identifying the shape.
        success: Whether warmup succeeded.
        cached: Whether shape was found in cache.
        compile_time_ms: Time taken in milliseconds.
        error: Error message if failed, None otherwise.
        backend: Backend used for compilation.

    Example:
        if not result.success:
            logger.warning(f"Shape {result.shape_key} failed: {result.error}")
    """

    shape_key: str
    success: bool
    cached: bool
    compile_time_ms: float
    error: str | None = None
    backend: str | None = None

    @property
    def status(self) -> str:
        """Get status string for logging.

        Returns:
            "cached", "compiled", or "failed".
        """
        if self.cached:
            return "cached"
        elif self.success:
            return "compiled"
        else:
            return "failed"
