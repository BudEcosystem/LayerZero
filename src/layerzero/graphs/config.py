"""
CUDA Graph safety configuration and result dataclasses.

This module provides:
- GraphSafetyConfig: Configuration for graph validation behavior
- GraphValidationResult: Result of graph validation
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GraphSafetyConfig:
    """Configuration for CUDA graph safety validation.

    Attributes:
        strict_mode: If True, require validation before graph capture.
        memory_delta_warning_mb: Memory change threshold for warnings.
        warmup_iterations: Number of warmup iterations before capture.
        default_graph_safe: Default safety for unknown kernels.
        validate_before_production: If True, run dummy capture for validation.
        capture_timeout_ms: Timeout for capture operations.

    Example:
        config = GraphSafetyConfig(
            strict_mode=True,
            memory_delta_warning_mb=1.0,
        )
    """

    strict_mode: bool = False
    memory_delta_warning_mb: float = 1.0
    warmup_iterations: int = 3
    default_graph_safe: bool = False
    validate_before_production: bool = True
    capture_timeout_ms: float = 30000.0

    def __post_init__(self) -> None:
        """Validate configuration values."""
        if self.memory_delta_warning_mb < 0:
            raise ValueError("memory_delta_warning_mb must be non-negative")
        if self.warmup_iterations < 0:
            raise ValueError("warmup_iterations must be non-negative")
        if self.capture_timeout_ms <= 0:
            raise ValueError("capture_timeout_ms must be positive")

    @classmethod
    def from_env(cls) -> "GraphSafetyConfig":
        """Create config from environment variables.

        Environment variables:
            LAYERZERO_GRAPH_STRICT_MODE: "1" or "true" to enable
            LAYERZERO_GRAPH_MEMORY_WARNING_MB: Memory threshold
            LAYERZERO_GRAPH_WARMUP_ITERATIONS: Warmup count
            LAYERZERO_GRAPH_DEFAULT_SAFE: "1" or "true" for default safe

        Returns:
            GraphSafetyConfig with values from environment.
        """
        strict_str = os.environ.get("LAYERZERO_GRAPH_STRICT_MODE", "0")
        strict_mode = strict_str.lower() in ("1", "true", "yes")

        memory_str = os.environ.get("LAYERZERO_GRAPH_MEMORY_WARNING_MB")
        memory_delta = float(memory_str) if memory_str else 1.0

        warmup_str = os.environ.get("LAYERZERO_GRAPH_WARMUP_ITERATIONS")
        warmup_iterations = int(warmup_str) if warmup_str else 3

        default_safe_str = os.environ.get("LAYERZERO_GRAPH_DEFAULT_SAFE", "0")
        default_safe = default_safe_str.lower() in ("1", "true", "yes")

        return cls(
            strict_mode=strict_mode,
            memory_delta_warning_mb=memory_delta,
            warmup_iterations=warmup_iterations,
            default_graph_safe=default_safe,
        )


@dataclass
class GraphValidationResult:
    """Result of CUDA graph validation.

    Attributes:
        success: Whether validation succeeded.
        kernel_id: ID of kernel being validated.
        operation: Operation being validated.
        error: Error message if failed.
        warnings: List of warning messages.
        memory_delta_mb: Memory change during capture.
        capture_time_ms: Time taken for capture.
        cublas_initialized: Whether cuBLAS was initialized.
        cudnn_initialized: Whether cuDNN was initialized.
        metadata: Additional metadata for debugging.

    Example:
        if not result.success:
            print(f"Validation failed: {result.error}")
            for warning in result.warnings:
                print(f"Warning: {warning}")
    """

    success: bool
    kernel_id: str | None = None
    operation: str | None = None
    error: str | None = None
    warnings: list[str] = field(default_factory=list)
    memory_delta_mb: float = 0.0
    capture_time_ms: float = 0.0
    cublas_initialized: bool = False
    cudnn_initialized: bool = False
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def has_warnings(self) -> bool:
        """Check if there are any warnings."""
        return len(self.warnings) > 0

    def add_warning(self, message: str) -> None:
        """Add a warning message."""
        self.warnings.append(message)

    def __str__(self) -> str:
        """Human-readable representation."""
        status = "SUCCESS" if self.success else "FAILED"
        parts = [f"GraphValidationResult({status}"]

        if self.kernel_id:
            parts.append(f", kernel={self.kernel_id}")
        if self.operation:
            parts.append(f", op={self.operation}")
        if self.error:
            parts.append(f", error={self.error}")
        if self.memory_delta_mb > 0:
            parts.append(f", memory_delta={self.memory_delta_mb:.2f}MB")
        if self.capture_time_ms > 0:
            parts.append(f", time={self.capture_time_ms:.1f}ms")

        parts.append(")")
        return "".join(parts)
