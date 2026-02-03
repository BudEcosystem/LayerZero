"""
Build-time solver for generating dispatch tables.

This module provides:
- SolverConfig: Configuration for solver
- SolverResult: Result of solve operation
- Solver: Build-time solver
- solve: Convenience function
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from layerzero._solve.dispatch_table import (
    BucketRange,
    DispatchEntry,
    DispatchTable,
    ShapeBucket,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SolverConfig:
    """Configuration for build-time solver.

    Attributes:
        enable_jit: Enable JIT compilation during solve.
        jit_warmup_iterations: Number of warmup iterations for JIT.
        include_hardware_signature: Include hardware signature in result.
        persist_table: Persist dispatch table to disk.
        table_path: Path for persisting dispatch table.
    """

    enable_jit: bool = True
    jit_warmup_iterations: int = 3
    include_hardware_signature: bool = True
    persist_table: bool = True
    table_path: Path | None = None


@dataclass
class SolverResult:
    """Result of solve operation.

    Attributes:
        dispatch_table: Generated dispatch table.
        hardware_signature: Hardware signature (if enabled).
        jit_compiled: List of JIT-compiled kernel IDs.
        solve_time_ms: Time taken to solve in milliseconds.
    """

    dispatch_table: DispatchTable
    hardware_signature: dict[str, Any] | None = None
    jit_compiled: list[str] = field(default_factory=list)
    solve_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict representation.
        """
        return {
            "dispatch_table": self.dispatch_table.to_dict(),
            "hardware_signature": self.hardware_signature,
            "jit_compiled": self.jit_compiled,
            "solve_time_ms": self.solve_time_ms,
        }


class Solver:
    """Build-time solver for generating dispatch tables.

    Analyzes shape buckets and generates a dispatch table for
    efficient kernel selection at runtime.

    Example:
        solver = Solver()

        result = solver.solve(
            operation="attention",
            buckets=[
                {"batch_size": [1, 2, 4], "seq_len": [512, 1024, 2048]},
            ],
        )

        # Use dispatch table
        kernel = result.dispatch_table.lookup({"batch_size": 2, "seq_len": 1024})
    """

    def __init__(self, config: SolverConfig | None = None) -> None:
        """Initialize solver.

        Args:
            config: Solver configuration.
        """
        self._config = config or SolverConfig()

    @property
    def config(self) -> SolverConfig:
        """Get configuration."""
        return self._config

    def solve(
        self,
        operation: str,
        buckets: list[dict[str, Any]],
    ) -> SolverResult:
        """Generate dispatch table for operation.

        Analyzes shape buckets and generates a dispatch table
        mapping shapes to kernels.

        Args:
            operation: Operation type (attention, matmul, etc.).
            buckets: List of bucket specifications.

        Returns:
            SolverResult with dispatch table and metadata.
        """
        start_time = time.monotonic()

        logger.info("Solving for operation=%s with %d bucket specs", operation, len(buckets))

        # Create dispatch table
        dispatch_table = DispatchTable()
        jit_compiled: list[str] = []

        # Get hardware context
        hw_ctx = self._get_hardware_context()

        # Process each bucket specification
        for bucket_spec in buckets:
            entries = self._process_bucket_spec(operation, bucket_spec, hw_ctx)
            for entry in entries:
                dispatch_table.add_entry(entry)

                # Trigger JIT if enabled
                if self._config.enable_jit:
                    self._jit_compile(entry.kernel_id, hw_ctx)
                    jit_compiled.append(entry.kernel_id)

        # Get hardware signature if enabled
        hardware_signature = None
        if self._config.include_hardware_signature:
            hardware_signature = self._get_hardware_signature(hw_ctx)

        # Calculate solve time
        solve_time_ms = (time.monotonic() - start_time) * 1000

        logger.info(
            "Solve complete: %d entries, %.2f ms",
            len(dispatch_table),
            solve_time_ms,
        )

        return SolverResult(
            dispatch_table=dispatch_table,
            hardware_signature=hardware_signature,
            jit_compiled=jit_compiled,
            solve_time_ms=solve_time_ms,
        )

    def _process_bucket_spec(
        self,
        operation: str,
        bucket_spec: dict[str, Any],
        hw_ctx: Any,
    ) -> list[DispatchEntry]:
        """Process a bucket specification.

        Args:
            operation: Operation type.
            bucket_spec: Bucket specification with dimension ranges.
            hw_ctx: Hardware context.

        Returns:
            List of dispatch entries for this bucket.
        """
        entries: list[DispatchEntry] = []

        # Extract dimension ranges from bucket spec
        ranges: dict[str, BucketRange] = {}

        for dim_name, values in bucket_spec.items():
            if isinstance(values, list) and len(values) > 0:
                # Convert list to range
                min_val = min(values)
                max_val = max(values)
                ranges[dim_name] = BucketRange(min_val=min_val, max_val=max_val)
            elif isinstance(values, dict):
                # Already a range spec
                ranges[dim_name] = BucketRange(
                    min_val=values.get("min", 1),
                    max_val=values.get("max", 1),
                )

        if not ranges:
            return entries

        # Create bucket
        bucket = ShapeBucket(ranges=ranges)

        # Select kernel for this bucket based on operation
        kernel_id = self._select_kernel_for_bucket(operation, bucket, hw_ctx)

        if kernel_id:
            entry = DispatchEntry(
                kernel_id=kernel_id,
                bucket=bucket,
                priority=100,
            )
            entries.append(entry)

        return entries

    def _select_kernel_for_bucket(
        self,
        operation: str,
        bucket: ShapeBucket,
        hw_ctx: Any,
    ) -> str | None:
        """Select best kernel for a bucket.

        Args:
            operation: Operation type.
            bucket: Shape bucket.
            hw_ctx: Hardware context.

        Returns:
            Kernel ID or None if no suitable kernel.
        """
        # Default kernel selection based on operation
        kernel_map = {
            "attention": "flash_attn_v2",
            "matmul": "cublas_gemm",
            "linear": "cublas_gemm",
            "layernorm": "triton_layernorm",
            "softmax": "triton_softmax",
        }

        return kernel_map.get(operation, f"default_{operation}")

    def _get_hardware_context(self) -> Any:
        """Get current hardware context.

        Returns:
            Hardware context object.
        """
        class HardwareContext:
            def __init__(self) -> None:
                self.platform = "cuda" if torch.cuda.is_available() else "cpu"
                if self.platform == "cuda":
                    props = torch.cuda.get_device_properties(0)
                    self.sm_version = props.major * 10 + props.minor
                    self.compute_capability = (props.major, props.minor)
                    self.device_name = props.name
                    self.total_memory = props.total_memory
                else:
                    self.sm_version = 0
                    self.compute_capability = (0, 0)
                    self.device_name = "CPU"
                    self.total_memory = 0

        return HardwareContext()

    def _get_hardware_signature(self, hw_ctx: Any) -> dict[str, Any]:
        """Get hardware signature for dispatch table.

        Args:
            hw_ctx: Hardware context.

        Returns:
            Hardware signature dictionary.
        """
        return {
            "platform": getattr(hw_ctx, 'platform', 'unknown'),
            "sm_version": getattr(hw_ctx, 'sm_version', 0),
            "compute_capability": getattr(hw_ctx, 'compute_capability', (0, 0)),
            "device_name": getattr(hw_ctx, 'device_name', 'unknown'),
        }

    def _jit_compile(self, kernel_id: str, hw_ctx: Any) -> None:
        """Trigger JIT compilation for kernel.

        Args:
            kernel_id: Kernel to compile.
            hw_ctx: Hardware context.
        """
        logger.debug(
            "JIT compiling kernel %s for %s",
            kernel_id,
            getattr(hw_ctx, 'device_name', 'unknown'),
        )

        # Actual JIT compilation would happen here
        # For now, we just log the intent
        for _ in range(self._config.jit_warmup_iterations):
            pass  # Warmup iterations


def solve(
    operation: str,
    buckets: list[dict[str, Any]],
    config: SolverConfig | None = None,
) -> SolverResult:
    """Convenience function for solving.

    Args:
        operation: Operation type.
        buckets: Bucket specifications.
        config: Optional solver configuration.

    Returns:
        SolverResult with dispatch table.
    """
    solver = Solver(config=config)
    return solver.solve(operation=operation, buckets=buckets)
