"""
LayerZero Triton Kernel Registry

Registry for managing custom Triton kernels.
Provides a singleton registry for kernel registration and lookup.
"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Set

import torch

from layerzero.backends.triton.adapter import TritonKernelAdapter
from layerzero.backends.triton.version import is_triton_available
from layerzero.models.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)

# Global registry instance
_registry: TritonKernelRegistry | None = None
_registry_lock = threading.Lock()


class TritonKernelRegistry:
    """Registry for custom Triton kernels.

    Thread-safe singleton registry for managing custom Triton kernels.
    Kernels are registered by name and can be retrieved for use in
    the kernel selection system.

    Example:
        ```python
        from layerzero.backends.triton import get_registry, register_triton_kernel

        # Register a kernel
        spec = register_triton_kernel(
            name="my_add",
            kernel_fn=add_kernel,
            operation="math.add",
            supported_dtypes={torch.float16},
        )

        # Get registry and lookup kernel
        registry = get_registry()
        adapter = registry.get("my_add")
        ```
    """

    def __init__(self) -> None:
        """Initialize the registry."""
        self._kernels: dict[str, TritonKernelAdapter] = {}
        self._lock = threading.Lock()

    def register(
        self,
        name: str,
        kernel_fn: Callable,
        operation: str,
        supported_dtypes: Set[torch.dtype],
        grid_fn: Callable | None = None,
        priority: int = 60,
        **kwargs: Any,
    ) -> KernelSpec:
        """Register a custom Triton kernel.

        Creates a TritonKernelAdapter for the kernel and stores it
        in the registry.

        Args:
            name: Unique name for this kernel.
            kernel_fn: Triton JIT-compiled kernel function.
            operation: Operation this kernel implements.
            supported_dtypes: Set of supported input dtypes.
            grid_fn: Optional function to compute grid dimensions.
            priority: Selection priority (0-100, default: 60).
            **kwargs: Additional KernelSpec parameters.

        Returns:
            KernelSpec for the registered kernel.

        Raises:
            RuntimeError: If Triton is not available.
            ValueError: If kernel with same name already registered.
        """
        if not is_triton_available():
            raise RuntimeError(
                "Triton is not available. Install with: pip install triton"
            )

        with self._lock:
            if name in self._kernels:
                logger.warning(
                    f"Kernel '{name}' already registered, replacing existing"
                )

            # Create adapter
            adapter = TritonKernelAdapter(
                name=name,
                kernel_fn=kernel_fn,
                operation=operation,
                supported_dtypes=supported_dtypes,
                grid_fn=grid_fn,
                priority=priority,
                **kwargs,
            )

            # Store in registry
            self._kernels[name] = adapter

            logger.debug(f"Registered Triton kernel: {name}")

            return adapter.get_kernel_spec()

    def get(self, name: str) -> TritonKernelAdapter | None:
        """Get registered kernel by name.

        Args:
            name: Kernel name to lookup.

        Returns:
            TritonKernelAdapter if found, None otherwise.
        """
        with self._lock:
            return self._kernels.get(name)

    def list_kernels(self) -> list[str]:
        """List all registered kernel names.

        Returns:
            List of registered kernel names.
        """
        with self._lock:
            return list(self._kernels.keys())

    def unregister(self, name: str) -> bool:
        """Unregister a kernel by name.

        Args:
            name: Kernel name to unregister.

        Returns:
            True if kernel was unregistered, False if not found.
        """
        with self._lock:
            if name in self._kernels:
                del self._kernels[name]
                logger.debug(f"Unregistered Triton kernel: {name}")
                return True
            return False

    def clear(self) -> None:
        """Clear all registered kernels."""
        with self._lock:
            self._kernels.clear()
            logger.debug("Cleared Triton kernel registry")


def get_registry() -> TritonKernelRegistry:
    """Get the global Triton kernel registry.

    Returns a singleton instance of the registry.

    Returns:
        TritonKernelRegistry singleton instance.
    """
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = TritonKernelRegistry()
    return _registry


def register_triton_kernel(
    name: str,
    kernel_fn: Callable,
    operation: str,
    supported_dtypes: Set[torch.dtype],
    grid_fn: Callable | None = None,
    priority: int = 60,
    **kwargs: Any,
) -> KernelSpec:
    """Register a custom Triton kernel with the global registry.

    Convenience function that delegates to the global registry.

    Args:
        name: Unique name for this kernel.
        kernel_fn: Triton JIT-compiled kernel function.
        operation: Operation this kernel implements.
        supported_dtypes: Set of supported input dtypes.
        grid_fn: Optional function to compute grid dimensions.
        priority: Selection priority (0-100, default: 60).
        **kwargs: Additional KernelSpec parameters.

    Returns:
        KernelSpec for the registered kernel.

    Raises:
        RuntimeError: If Triton is not available.
    """
    registry = get_registry()
    return registry.register(
        name=name,
        kernel_fn=kernel_fn,
        operation=operation,
        supported_dtypes=supported_dtypes,
        grid_fn=grid_fn,
        priority=priority,
        **kwargs,
    )
