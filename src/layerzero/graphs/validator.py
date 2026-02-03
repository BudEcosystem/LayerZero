"""
CUDA Graph validation for kernel safety.

This module provides:
- GraphValidator: Validate graph capture safety
"""
from __future__ import annotations

import logging
import time
from typing import Any, Callable, TYPE_CHECKING

import torch

from layerzero.graphs.config import GraphSafetyConfig, GraphValidationResult
from layerzero.graphs.memory_tracker import MemoryTracker
from layerzero.graphs.warmup import GraphWarmupProtocol
from layerzero.graphs.whitelist import GraphWhitelist

if TYPE_CHECKING:
    from layerzero.models.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)


class GraphValidator:
    """Validator for CUDA graph capture safety.

    Validates that kernels can be safely captured in CUDA graphs by:
    1. Checking whitelist for known safe/unsafe kernels
    2. Running warmup before capture
    3. Monitoring memory changes during capture
    4. Optionally running dummy capture in strict mode

    Example:
        config = GraphSafetyConfig(strict_mode=True)
        validator = GraphValidator(config)

        # Validate before capture
        result = validator.validate_capture(my_func, inputs)
        if not result.success:
            print(f"Validation failed: {result.error}")
    """

    def __init__(
        self,
        config: GraphSafetyConfig | None = None,
        whitelist: GraphWhitelist | None = None,
    ) -> None:
        """Initialize validator.

        Args:
            config: Graph safety configuration.
            whitelist: Kernel whitelist.
        """
        self._config = config or GraphSafetyConfig()
        self._whitelist = whitelist or GraphWhitelist(
            default_safe=self._config.default_graph_safe
        )
        self._warmup_protocol = GraphWarmupProtocol(
            warmup_iterations=self._config.warmup_iterations
        )
        self._memory_tracker = MemoryTracker()

    @property
    def config(self) -> GraphSafetyConfig:
        """Get configuration."""
        return self._config

    @property
    def whitelist(self) -> GraphWhitelist:
        """Get whitelist."""
        return self._whitelist

    def is_graph_safe(
        self,
        kernel_id: str,
        strict: bool | None = None,
    ) -> bool:
        """Check if kernel is graph-safe.

        Args:
            kernel_id: Kernel identifier.
            strict: If True, reject unknown kernels. Uses config if None.

        Returns:
            True if kernel is graph-safe.
        """
        if strict is None:
            strict = self._config.strict_mode
        return self._whitelist.is_graph_safe(kernel_id, strict=strict)

    def validate_kernel(
        self,
        kernel: "KernelSpec",
    ) -> GraphValidationResult:
        """Validate kernel for graph safety.

        Args:
            kernel: KernelSpec to validate.

        Returns:
            GraphValidationResult with validation status.
        """
        result = GraphValidationResult(
            success=True,
            kernel_id=kernel.kernel_id,
            operation=kernel.operation,
        )

        # Check whitelist
        is_safe = self._whitelist.is_graph_safe_kernel(
            kernel, strict=self._config.strict_mode
        )

        if not is_safe:
            result.success = False
            result.error = f"Kernel {kernel.kernel_id} is not graph-safe"

        # Check explicit kernel flag
        if kernel.is_cuda_graph_safe is False:
            result.success = False
            result.error = f"Kernel {kernel.kernel_id} explicitly marked as graph-unsafe"

        return result

    def validate_capture(
        self,
        func: Callable[..., Any],
        *args: Any,
        kernel_id: str | None = None,
        operation: str | None = None,
        **kwargs: Any,
    ) -> GraphValidationResult:
        """Validate function can be captured in CUDA graph.

        Performs full validation including:
        1. Warmup
        2. Memory tracking
        3. Optional dummy capture (in strict mode)

        Args:
            func: Function to validate.
            *args: Arguments to pass to function.
            kernel_id: Optional kernel ID for reporting.
            operation: Optional operation name for reporting.
            **kwargs: Keyword arguments to pass.

        Returns:
            GraphValidationResult with validation status.
        """
        start_time = time.perf_counter()

        result = GraphValidationResult(
            success=True,
            kernel_id=kernel_id,
            operation=operation,
        )

        if not torch.cuda.is_available():
            result.add_warning("CUDA not available, skipping validation")
            return result

        try:
            # Step 1: Warmup
            warmup_state = self._warmup_protocol.warmup(func, *args, **kwargs)
            result.cublas_initialized = warmup_state.cublas_initialized
            result.cudnn_initialized = warmup_state.cudnn_initialized

            if not warmup_state.is_ready:
                result.add_warning("Warmup incomplete")
                for error in warmup_state.errors:
                    result.add_warning(f"Warmup error: {error}")

            # Step 2: Memory tracking
            self._memory_tracker.reset()
            self._memory_tracker.snapshot("pre_capture")

            # Step 3: Dummy capture (if strict mode or validate_before_production)
            if self._config.strict_mode or self._config.validate_before_production:
                capture_result = self._dummy_capture(func, *args, **kwargs)
                if not capture_result["success"]:
                    result.success = False
                    result.error = capture_result.get("error", "Capture failed")
                    return result

            self._memory_tracker.snapshot("post_capture")

            # Step 4: Check memory delta
            within_threshold, message = self._memory_tracker.check_memory_delta(
                "pre_capture",
                "post_capture",
                threshold_mb=self._config.memory_delta_warning_mb,
            )

            if not within_threshold:
                result.add_warning(message)

            delta = self._memory_tracker.get_delta("pre_capture", "post_capture")
            if delta:
                result.memory_delta_mb = delta.allocated_delta_mb

        except Exception as e:
            result.success = False
            result.error = f"Validation failed: {e}"
            logger.exception("Graph validation error")

        result.capture_time_ms = (time.perf_counter() - start_time) * 1000
        return result

    def _dummy_capture(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Run dummy capture to validate graph safety.

        Args:
            func: Function to capture.
            *args: Arguments to pass.
            **kwargs: Keyword arguments to pass.

        Returns:
            Dict with success status and error if failed.
        """
        try:
            g = torch.cuda.CUDAGraph()
            s = torch.cuda.Stream()

            # Pre-capture run
            with torch.cuda.stream(s):
                _ = func(*args, **kwargs)
            s.synchronize()

            # Capture
            with torch.cuda.graph(g, stream=s):
                output = func(*args, **kwargs)

            # Verify output is valid
            if isinstance(output, torch.Tensor) and not torch.isfinite(output).all():
                return {
                    "success": False,
                    "error": "Capture produced non-finite output",
                }

            # Test replay
            g.replay()
            torch.cuda.synchronize()

            return {"success": True, "graph": g}

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
            }

    def validate_batch(
        self,
        kernels: list["KernelSpec"],
    ) -> dict[str, GraphValidationResult]:
        """Validate multiple kernels.

        Args:
            kernels: List of KernelSpecs to validate.

        Returns:
            Dict mapping kernel_id to validation result.
        """
        results = {}
        for kernel in kernels:
            results[kernel.kernel_id] = self.validate_kernel(kernel)
        return results


# Global validator instance
_global_validator: GraphValidator | None = None


def get_global_validator() -> GraphValidator:
    """Get global validator instance.

    Returns:
        Global GraphValidator instance.
    """
    global _global_validator
    if _global_validator is None:
        _global_validator = GraphValidator()
    return _global_validator


def is_graph_safe(kernel_id: str, strict: bool | None = None) -> bool:
    """Check if kernel is graph-safe using global validator.

    Args:
        kernel_id: Kernel identifier.
        strict: If True, reject unknown kernels.

    Returns:
        True if kernel is graph-safe.
    """
    return get_global_validator().is_graph_safe(kernel_id, strict=strict)
