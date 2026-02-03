"""
CUDA Graph whitelist for graph-safe kernels.

This module provides:
- GraphWhitelist: Manages whitelist of graph-safe kernels
- DEFAULT_SAFE_KERNELS: Known graph-safe operations
- DEFAULT_UNSAFE_KERNELS: Known graph-unsafe operations
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.models.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)

# Known graph-safe operations
DEFAULT_SAFE_KERNELS: frozenset[str] = frozenset([
    # Attention operations
    "attention",
    "attention.causal",
    "attention.prefill",
    "attention.decode",
    "attention.paged",
    # Normalization
    "norm.rms",
    "norm.layer",
    "norm.batch",
    # Matrix operations
    "matmul",
    "linear",
    # Activations
    "activation.gelu",
    "activation.silu",
    "activation.swiglu",
    "activation.geglu",
    "activation.relu",
    # Element-wise
    "elementwise.add",
    "elementwise.mul",
    "elementwise.div",
    # Pooling
    "pool.avg",
    "pool.max",
    # RoPE
    "rope",
    "rope.forward",
    # Cross entropy (forward only, backward may allocate)
    "cross_entropy.forward",
])

# Known graph-unsafe operations (may allocate memory dynamically)
DEFAULT_UNSAFE_KERNELS: frozenset[str] = frozenset([
    # Dynamic shape operations
    "dynamic_shape_op",
    "dynamic_slice",
    "dynamic_concat",
    # Host-device transfers
    "host_to_device",
    "device_to_host",
    "copy_async",
    # Operations with dynamic allocation
    "print_tensor",
    "tensor_to_list",
    "topk_variable",
    # NCCL operations (may synchronize)
    "nccl.all_reduce",
    "nccl.all_gather",
    "nccl.reduce_scatter",
    # Python callbacks
    "python_callback",
    "custom_autograd",
    # Synchronization
    "stream_sync",
    "event_sync",
])


class GraphWhitelist:
    """Manages whitelist of CUDA graph-safe kernels.

    The whitelist determines which kernels can be safely captured in
    CUDA graphs. Unknown kernels are treated based on configuration.

    Thread-safe for concurrent access.

    Example:
        whitelist = GraphWhitelist()

        # Check if kernel is graph-safe
        if whitelist.is_graph_safe("attention.causal"):
            # Safe to capture
            ...

        # Add custom safe kernel
        whitelist.add_safe_kernel("my_custom_kernel")
    """

    def __init__(
        self,
        safe_kernels: frozenset[str] | None = None,
        unsafe_kernels: frozenset[str] | None = None,
        default_safe: bool = False,
    ) -> None:
        """Initialize whitelist.

        Args:
            safe_kernels: Set of known safe kernel IDs. If None, uses defaults.
            unsafe_kernels: Set of known unsafe kernel IDs. If None, uses defaults.
            default_safe: If True, unknown kernels are assumed safe.
        """
        self._safe_kernels: set[str] = set(safe_kernels or DEFAULT_SAFE_KERNELS)
        self._unsafe_kernels: set[str] = set(unsafe_kernels or DEFAULT_UNSAFE_KERNELS)
        self._default_safe = default_safe
        self._lock = threading.RLock()

    @property
    def default_safe(self) -> bool:
        """Get default safety for unknown kernels."""
        return self._default_safe

    def is_graph_safe(
        self,
        kernel_id: str,
        strict: bool | None = None,
    ) -> bool:
        """Check if kernel is safe for CUDA graph capture.

        Args:
            kernel_id: Kernel identifier or operation name.
            strict: If True, reject unknown kernels. If None, uses default_safe.

        Returns:
            True if kernel is graph-safe.
        """
        with self._lock:
            # Check explicit lists first
            if kernel_id in self._safe_kernels:
                return True
            if kernel_id in self._unsafe_kernels:
                return False

            # Also check operation prefixes (e.g., "attention.causal" matches "attention")
            for safe_kernel in self._safe_kernels:
                if kernel_id.startswith(safe_kernel + "."):
                    return True
                if safe_kernel.startswith(kernel_id + "."):
                    return True

            for unsafe_kernel in self._unsafe_kernels:
                if kernel_id.startswith(unsafe_kernel + "."):
                    return False
                if unsafe_kernel.startswith(kernel_id + "."):
                    return False

            # Unknown kernel
            if strict is None:
                return self._default_safe
            return not strict

    def is_graph_safe_kernel(
        self,
        kernel: "KernelSpec",
        strict: bool | None = None,
    ) -> bool:
        """Check if KernelSpec is graph-safe.

        Uses kernel's is_cuda_graph_safe field if set, otherwise
        checks the whitelist.

        Args:
            kernel: KernelSpec to check.
            strict: If True, reject unknown kernels.

        Returns:
            True if kernel is graph-safe.
        """
        # If kernel has explicit graph safety flag, use it
        if kernel.is_cuda_graph_safe is not None:
            return kernel.is_cuda_graph_safe

        # Otherwise check whitelist
        return self.is_graph_safe(kernel.kernel_id, strict=strict)

    def add_safe_kernel(self, kernel_id: str) -> None:
        """Add kernel to safe list.

        Args:
            kernel_id: Kernel identifier to add.
        """
        with self._lock:
            self._safe_kernels.add(kernel_id)
            # Remove from unsafe if present
            self._unsafe_kernels.discard(kernel_id)

    def remove_safe_kernel(self, kernel_id: str) -> None:
        """Remove kernel from safe list.

        Args:
            kernel_id: Kernel identifier to remove.
        """
        with self._lock:
            self._safe_kernels.discard(kernel_id)

    def add_unsafe_kernel(self, kernel_id: str) -> None:
        """Add kernel to unsafe list.

        Args:
            kernel_id: Kernel identifier to add.
        """
        with self._lock:
            self._unsafe_kernels.add(kernel_id)
            # Remove from safe if present
            self._safe_kernels.discard(kernel_id)

    def remove_unsafe_kernel(self, kernel_id: str) -> None:
        """Remove kernel from unsafe list.

        Args:
            kernel_id: Kernel identifier to remove.
        """
        with self._lock:
            self._unsafe_kernels.discard(kernel_id)

    def get_safe_kernels(self) -> frozenset[str]:
        """Get set of graph-safe kernel IDs.

        Returns:
            Frozen set of safe kernel IDs.
        """
        with self._lock:
            return frozenset(self._safe_kernels)

    def get_unsafe_kernels(self) -> frozenset[str]:
        """Get set of graph-unsafe kernel IDs.

        Returns:
            Frozen set of unsafe kernel IDs.
        """
        with self._lock:
            return frozenset(self._unsafe_kernels)

    def __len__(self) -> int:
        """Get number of safe kernels."""
        with self._lock:
            return len(self._safe_kernels)

    def __contains__(self, kernel_id: str) -> bool:
        """Check if kernel is in safe list."""
        return self.is_graph_safe(kernel_id)
