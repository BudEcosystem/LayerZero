"""GPU generation-based kernel routing.

This module provides routing logic for selecting kernels
based on GPU architecture generation.
"""
from __future__ import annotations

from layerzero.routing.gpu_routing import (
    filter_by_generation,
    score_by_generation,
    select_best_for_generation,
    GenerationRouter,
    RouterConfig,
)

__all__ = [
    "filter_by_generation",
    "score_by_generation",
    "select_best_for_generation",
    "GenerationRouter",
    "RouterConfig",
]
