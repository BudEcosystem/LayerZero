"""
LayerZero Kernel Registry

Central registry for all available kernels.
Thread-safe kernel registration and lookup.
"""
from __future__ import annotations

from threading import RLock
from typing import Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from layerzero.enums import Platform
    from layerzero.models.kernel_spec import KernelSpec


class KernelRegistry:
    """Central registry of available kernels.

    Thread-safe kernel registration and lookup.
    Organizes kernels by operation type for efficient filtering.

    All public methods are thread-safe.
    """

    __slots__ = (
        "_lock",
        "_kernels",
        "_by_operation",
        "_by_source",
    )

    def __init__(self) -> None:
        """Initialize empty kernel registry."""
        self._lock = RLock()
        self._kernels: dict[str, "KernelSpec"] = {}
        self._by_operation: dict[str, list[str]] = {}
        self._by_source: dict[str, list[str]] = {}

    def register(self, spec: "KernelSpec") -> None:
        """Register a kernel specification.

        Args:
            spec: Kernel specification to register.

        Raises:
            ValueError: If kernel_id is already registered.
        """
        with self._lock:
            if spec.kernel_id in self._kernels:
                raise ValueError(
                    f"Kernel '{spec.kernel_id}' is already registered"
                )

            self._kernels[spec.kernel_id] = spec

            # Index by operation
            if spec.operation not in self._by_operation:
                self._by_operation[spec.operation] = []
            self._by_operation[spec.operation].append(spec.kernel_id)

            # Index by source
            if spec.source not in self._by_source:
                self._by_source[spec.source] = []
            self._by_source[spec.source].append(spec.kernel_id)

    def register_many(self, specs: Sequence["KernelSpec"]) -> None:
        """Register multiple kernel specifications atomically.

        If any registration fails, all registrations are rolled back.

        Args:
            specs: Sequence of kernel specifications to register.

        Raises:
            ValueError: If any kernel_id is already registered.
        """
        with self._lock:
            # Check for duplicates first
            for spec in specs:
                if spec.kernel_id in self._kernels:
                    raise ValueError(
                        f"Kernel '{spec.kernel_id}' is already registered"
                    )

            # Also check for duplicates within the batch
            kernel_ids = [spec.kernel_id for spec in specs]
            if len(kernel_ids) != len(set(kernel_ids)):
                raise ValueError("Duplicate kernel_id in batch")

            # All checks passed, register
            for spec in specs:
                self._kernels[spec.kernel_id] = spec

                if spec.operation not in self._by_operation:
                    self._by_operation[spec.operation] = []
                self._by_operation[spec.operation].append(spec.kernel_id)

                if spec.source not in self._by_source:
                    self._by_source[spec.source] = []
                self._by_source[spec.source].append(spec.kernel_id)

    def unregister(self, kernel_id: str) -> bool:
        """Unregister a kernel by ID.

        Args:
            kernel_id: Kernel identifier to unregister.

        Returns:
            True if unregistered, False if not found.
        """
        with self._lock:
            if kernel_id not in self._kernels:
                return False

            spec = self._kernels.pop(kernel_id)

            # Remove from operation index
            if spec.operation in self._by_operation:
                self._by_operation[spec.operation].remove(kernel_id)
                if not self._by_operation[spec.operation]:
                    del self._by_operation[spec.operation]

            # Remove from source index
            if spec.source in self._by_source:
                self._by_source[spec.source].remove(kernel_id)
                if not self._by_source[spec.source]:
                    del self._by_source[spec.source]

            return True

    def get(self, kernel_id: str) -> "KernelSpec | None":
        """Get kernel by ID.

        Args:
            kernel_id: Kernel identifier.

        Returns:
            KernelSpec if found, None otherwise.
        """
        with self._lock:
            return self._kernels.get(kernel_id)

    def get_by_operation(self, operation: str) -> list["KernelSpec"]:
        """Get all kernels for an operation type.

        Args:
            operation: Operation identifier (e.g., "attention.causal").

        Returns:
            List of KernelSpecs for the operation (empty if none).
        """
        with self._lock:
            kernel_ids = self._by_operation.get(operation, [])
            return [self._kernels[kid] for kid in kernel_ids]

    def get_by_source(self, source: str) -> list["KernelSpec"]:
        """Get all kernels from a source library.

        Args:
            source: Source library (e.g., "flash_attn").

        Returns:
            List of KernelSpecs from the source (empty if none).
        """
        with self._lock:
            kernel_ids = self._by_source.get(source, [])
            return [self._kernels[kid] for kid in kernel_ids]

    def get_all(self) -> list["KernelSpec"]:
        """Get all registered kernels.

        Returns:
            List of all registered KernelSpecs.
        """
        with self._lock:
            return list(self._kernels.values())

    def filter(
        self,
        operation: str | None = None,
        source: str | None = None,
        platform: "Platform | None" = None,
    ) -> list["KernelSpec"]:
        """Filter kernels by criteria.

        All criteria are AND-ed together.

        Args:
            operation: Filter by operation type.
            source: Filter by source library.
            platform: Filter by platform.

        Returns:
            List of matching KernelSpecs.
        """
        with self._lock:
            results = list(self._kernels.values())

            if operation is not None:
                results = [k for k in results if k.operation == operation]

            if source is not None:
                results = [k for k in results if k.source == source]

            if platform is not None:
                results = [k for k in results if k.platform == platform]

            return results

    def clear(self) -> None:
        """Clear all registered kernels."""
        with self._lock:
            self._kernels.clear()
            self._by_operation.clear()
            self._by_source.clear()

    @property
    def kernel_count(self) -> int:
        """Number of registered kernels."""
        with self._lock:
            return len(self._kernels)

    @property
    def operations(self) -> frozenset[str]:
        """Set of registered operations."""
        with self._lock:
            return frozenset(self._by_operation.keys())
