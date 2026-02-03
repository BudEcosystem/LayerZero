"""
LayerZero CUDA Graph Validation Module

Provides CUDA graph safety validation to ensure kernels can be captured
and replayed without errors.

Main components:
- GraphSafetyConfig: Configuration for graph validation
- GraphWhitelist: Whitelist of graph-safe kernels
- GraphWarmupProtocol: Warmup before graph capture
- GraphValidator: Validate graph capture

Usage:
    from layerzero.graphs import (
        GraphSafetyConfig,
        GraphWhitelist,
        GraphWarmupProtocol,
        GraphValidator,
    )

    config = GraphSafetyConfig(strict_mode=True)
    validator = GraphValidator(config)

    result = validator.validate_capture(my_func, inputs)
    if not result.success:
        print(f"Graph capture failed: {result.error}")
"""
from __future__ import annotations

from layerzero.graphs.config import (
    GraphSafetyConfig,
    GraphValidationResult,
)
from layerzero.graphs.memory_tracker import (
    MemoryTracker,
)
from layerzero.graphs.validator import (
    GraphValidator,
)
from layerzero.graphs.warmup import (
    GraphWarmupProtocol,
)
from layerzero.graphs.whitelist import (
    DEFAULT_SAFE_KERNELS,
    DEFAULT_UNSAFE_KERNELS,
    GraphWhitelist,
)

__all__ = [
    # Config
    "GraphSafetyConfig",
    "GraphValidationResult",
    # Whitelist
    "GraphWhitelist",
    "DEFAULT_SAFE_KERNELS",
    "DEFAULT_UNSAFE_KERNELS",
    # Warmup
    "GraphWarmupProtocol",
    # Validator
    "GraphValidator",
    # Memory
    "MemoryTracker",
]
