"""GPU generation-based kernel routing.

This module provides:
- filter_by_generation: Filter kernels by device GPU generation
- score_by_generation: Score kernels based on generation match
- select_best_for_generation: Select optimal kernel for a generation
- GenerationRouter: High-level routing orchestrator
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, TypeVar

from layerzero.device import GPUGeneration

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class RouterConfig:
    """Configuration for GPU generation router.

    Attributes:
        prefer_native: Prefer kernels native to device generation.
        native_bonus: Score bonus for native generation kernels.
        fallback_to_universal: Allow kernels with empty supported_generations.
        strict_mode: Reject kernels not explicitly supporting the generation.
    """

    prefer_native: bool = True
    native_bonus: int = 100
    fallback_to_universal: bool = True
    strict_mode: bool = False


def filter_by_generation(
    kernels: list[dict[str, Any]],
    device_generation: GPUGeneration,
    config: RouterConfig | None = None,
) -> list[dict[str, Any]]:
    """Filter kernels by device GPU generation.

    Kernels with empty supported_generations are considered universal
    and pass through unless strict_mode is enabled.

    Args:
        kernels: List of kernel dicts with 'supported_generations' key.
        device_generation: Target GPU generation.
        config: Router configuration.

    Returns:
        List of kernels compatible with the device generation.
    """
    config = config or RouterConfig()
    filtered = []

    for kernel in kernels:
        supported = kernel.get("supported_generations", frozenset())

        # Empty means universal (all generations)
        if not supported:
            if config.fallback_to_universal:
                filtered.append(kernel)
            continue

        # Check if device generation is supported
        if device_generation in supported:
            filtered.append(kernel)

    logger.debug(
        "Filtered %d kernels to %d for generation %s",
        len(kernels),
        len(filtered),
        device_generation.value,
    )
    return filtered


def score_by_generation(
    kernel: dict[str, Any],
    device_generation: GPUGeneration,
    config: RouterConfig | None = None,
) -> int:
    """Score a kernel based on generation match.

    Higher scores indicate better match for the device generation.
    Native kernels (designed for this generation) score higher than
    kernels that merely support the generation.

    Args:
        kernel: Kernel dict with generation info.
        device_generation: Target GPU generation.
        config: Router configuration.

    Returns:
        Integer score (higher = better match).
    """
    config = config or RouterConfig()
    score = 0

    supported = kernel.get("supported_generations", frozenset())
    native_gen = kernel.get("native_generation")

    # Base score for supporting the generation
    if not supported or device_generation in supported:
        score += 50

    # Bonus for native generation match
    if config.prefer_native and native_gen == device_generation:
        score += config.native_bonus

    # Penalty for being a fallback/universal kernel
    if not supported and config.strict_mode:
        score -= 25

    return score


def select_best_for_generation(
    kernels: list[dict[str, Any]],
    device_generation: GPUGeneration,
    config: RouterConfig | None = None,
) -> dict[str, Any] | None:
    """Select the best kernel for a GPU generation.

    Filters by generation, then scores and returns the best match.

    Args:
        kernels: List of kernel dicts.
        device_generation: Target GPU generation.
        config: Router configuration.

    Returns:
        Best kernel dict or None if no compatible kernel found.
    """
    config = config or RouterConfig()

    # Filter first
    compatible = filter_by_generation(kernels, device_generation, config)
    if not compatible:
        logger.warning(
            "No compatible kernels for generation %s",
            device_generation.value,
        )
        return None

    # Score and sort
    scored = [
        (kernel, score_by_generation(kernel, device_generation, config))
        for kernel in compatible
    ]
    scored.sort(key=lambda x: x[1], reverse=True)

    best = scored[0][0]
    logger.debug(
        "Selected kernel %s for generation %s (score: %d)",
        best.get("id", "unknown"),
        device_generation.value,
        scored[0][1],
    )
    return best


class GenerationRouter:
    """High-level GPU generation router.

    Orchestrates kernel selection based on GPU generation with
    caching and configuration support.
    """

    def __init__(
        self,
        config: RouterConfig | None = None,
    ) -> None:
        """Initialize the router.

        Args:
            config: Router configuration.
        """
        self._config = config or RouterConfig()
        self._lock = RLock()
        self._cache: dict[tuple[str, GPUGeneration], dict[str, Any] | None] = {}

        logger.debug("GenerationRouter initialized")

    @property
    def config(self) -> RouterConfig:
        """Get router configuration."""
        return self._config

    def route(
        self,
        operation: str,
        kernels: list[dict[str, Any]],
        device_generation: GPUGeneration,
    ) -> dict[str, Any] | None:
        """Route to the best kernel for an operation on a generation.

        Args:
            operation: Operation name (for caching).
            kernels: Available kernels.
            device_generation: Target GPU generation.

        Returns:
            Selected kernel or None.
        """
        cache_key = (operation, device_generation)

        with self._lock:
            # Check cache
            if cache_key in self._cache:
                logger.debug("Cache hit for %s on %s", operation, device_generation.value)
                return self._cache[cache_key]

            # Select best kernel
            result = select_best_for_generation(
                kernels,
                device_generation,
                self._config,
            )

            # Cache result
            self._cache[cache_key] = result
            return result

    def clear_cache(self) -> int:
        """Clear the routing cache.

        Returns:
            Number of entries cleared.
        """
        with self._lock:
            count = len(self._cache)
            self._cache.clear()
            logger.debug("Cleared %d cache entries", count)
            return count

    def get_fa_kernel_for_generation(
        self,
        device_generation: GPUGeneration,
    ) -> str:
        """Get the appropriate FlashAttention version for a generation.

        This is a convenience method for FA version selection.

        Args:
            device_generation: Target GPU generation.

        Returns:
            FA kernel identifier ("fa2", "fa3", or "fa4").
        """
        if device_generation >= GPUGeneration.BLACKWELL:
            return "fa4"
        elif device_generation >= GPUGeneration.HOPPER:
            return "fa3"
        else:
            return "fa2"

    def is_fa3_compatible(self, device_generation: GPUGeneration) -> bool:
        """Check if FA3 is compatible with a GPU generation.

        FA3 supports Ampere through Hopper, but NOT Blackwell.

        Args:
            device_generation: GPU generation to check.

        Returns:
            True if FA3 is compatible.
        """
        return device_generation in {
            GPUGeneration.AMPERE,
            GPUGeneration.ADA_LOVELACE,
            GPUGeneration.HOPPER,
        }

    def is_fa4_compatible(self, device_generation: GPUGeneration) -> bool:
        """Check if FA4 is compatible with a GPU generation.

        FA4 is only for Blackwell and newer.

        Args:
            device_generation: GPU generation to check.

        Returns:
            True if FA4 is compatible.
        """
        return device_generation >= GPUGeneration.BLACKWELL
