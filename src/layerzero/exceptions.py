"""
LayerZero Exception Hierarchy

Custom exceptions for LayerZero error handling per spec Section 14.1.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence


class LayerZeroError(Exception):
    """Base exception for all LayerZero errors.

    All LayerZero-specific exceptions inherit from this class,
    allowing users to catch all LayerZero errors with a single
    except clause.

    Attributes:
        message: Human-readable error description.
        context: Optional dict of additional context for debugging.
    """

    def __init__(
        self,
        message: str,
        *,
        context: Optional[dict[str, Any]] = None,
    ) -> None:
        """Initialize LayerZeroError.

        Args:
            message: Error message.
            context: Optional context dict.
        """
        super().__init__(message)
        self.message = message
        self.context = context or {}

    def __repr__(self) -> str:
        """Detailed representation for debugging."""
        ctx_str = f", context={self.context}" if self.context else ""
        return f"{self.__class__.__name__}({self.message!r}{ctx_str})"


class NoKernelFoundError(LayerZeroError):
    """Raised when no suitable kernel is found for an operation.

    This typically occurs when:
    - No kernel supports the requested operation
    - All candidate kernels were filtered out by constraints
    - Policy rules denied all available kernels

    Attributes:
        operation: The operation that failed to find a kernel.
        candidates: List of kernels that were considered.
        filtered_out: Dict of kernel_id -> rejection reasons.
    """

    def __init__(
        self,
        operation: str,
        *,
        candidates: Optional[Sequence[str]] = None,
        filtered_out: Optional[dict[str, list[str]]] = None,
        message: Optional[str] = None,
    ) -> None:
        """Initialize NoKernelFoundError.

        Args:
            operation: Operation identifier.
            candidates: Kernels that were considered.
            filtered_out: Kernels and their rejection reasons.
            message: Optional custom message.
        """
        self.operation = operation
        self.candidates = list(candidates) if candidates else []
        self.filtered_out = filtered_out or {}

        if message is None:
            if self.candidates:
                message = (
                    f"No suitable kernel found for operation '{operation}'. "
                    f"Considered {len(self.candidates)} candidates: {self.candidates}"
                )
            else:
                message = f"No kernel available for operation '{operation}'"

        super().__init__(
            message,
            context={
                "operation": operation,
                "candidates": self.candidates,
                "filtered_out": self.filtered_out,
            },
        )


class CudaGraphUnsafeError(LayerZeroError):
    """Raised when a kernel is unsafe for CUDA graph capture.

    CUDA graphs require deterministic memory allocation patterns.
    Kernels that allocate memory dynamically or use non-deterministic
    operations cannot be captured in CUDA graphs.

    Attributes:
        kernel_id: The kernel that is graph-unsafe.
        reason: Why the kernel is graph-unsafe.
    """

    def __init__(
        self,
        kernel_id: str,
        reason: str,
        *,
        message: Optional[str] = None,
    ) -> None:
        """Initialize CudaGraphUnsafeError.

        Args:
            kernel_id: Kernel identifier.
            reason: Reason for being graph-unsafe.
            message: Optional custom message.
        """
        self.kernel_id = kernel_id
        self.reason = reason

        if message is None:
            message = (
                f"Kernel '{kernel_id}' is not safe for CUDA graph capture: {reason}"
            )

        super().__init__(
            message,
            context={"kernel_id": kernel_id, "reason": reason},
        )


class KernelExecutionError(LayerZeroError):
    """Raised when a kernel fails during execution.

    This wraps the original exception from the kernel with
    additional context about which kernel failed.

    Attributes:
        kernel_id: The kernel that failed.
        original_error: The underlying exception.
    """

    def __init__(
        self,
        kernel_id: str,
        original_error: BaseException,
        *,
        message: Optional[str] = None,
    ) -> None:
        """Initialize KernelExecutionError.

        Args:
            kernel_id: Kernel identifier.
            original_error: The original exception.
            message: Optional custom message.
        """
        self.kernel_id = kernel_id
        self.original_error = original_error

        if message is None:
            message = (
                f"Kernel '{kernel_id}' execution failed: {original_error}"
            )

        super().__init__(
            message,
            context={
                "kernel_id": kernel_id,
                "error_type": type(original_error).__name__,
                "error_message": str(original_error),
            },
        )
        self.__cause__ = original_error


class PolicyValidationError(LayerZeroError):
    """Raised when policy configuration is invalid.

    This occurs when:
    - YAML policy file has syntax errors
    - Policy rules reference non-existent kernels
    - Conflicting rules are defined
    - Required fields are missing

    Attributes:
        policy_file: Optional path to the policy file.
        validation_errors: List of validation error messages.
    """

    def __init__(
        self,
        message: str,
        *,
        policy_file: Optional[str] = None,
        validation_errors: Optional[Sequence[str]] = None,
    ) -> None:
        """Initialize PolicyValidationError.

        Args:
            message: Error message.
            policy_file: Path to invalid policy file.
            validation_errors: List of specific errors.
        """
        self.policy_file = policy_file
        self.validation_errors = list(validation_errors) if validation_errors else []

        super().__init__(
            message,
            context={
                "policy_file": policy_file,
                "validation_errors": self.validation_errors,
            },
        )


class BackendNotAvailableError(LayerZeroError):
    """Raised when a required backend is not available.

    This occurs when:
    - Backend package is not installed
    - Backend import fails due to version mismatch
    - Backend has ABI incompatibility
    - Backend is disabled by policy

    Attributes:
        backend_id: The unavailable backend.
        reason: Why the backend is unavailable.
    """

    def __init__(
        self,
        backend_id: str,
        reason: str,
        *,
        message: Optional[str] = None,
    ) -> None:
        """Initialize BackendNotAvailableError.

        Args:
            backend_id: Backend identifier.
            reason: Why the backend is unavailable.
            message: Optional custom message.
        """
        self.backend_id = backend_id
        self.reason = reason

        if message is None:
            message = f"Backend '{backend_id}' is not available: {reason}"

        super().__init__(
            message,
            context={"backend_id": backend_id, "reason": reason},
        )


class ConfigurationError(LayerZeroError):
    """Raised when LayerZero configuration is invalid.

    This is a general configuration error for issues not covered
    by PolicyValidationError.
    """

    def __init__(
        self,
        message: str,
        *,
        config_key: Optional[str] = None,
        expected: Optional[Any] = None,
        got: Optional[Any] = None,
    ) -> None:
        """Initialize ConfigurationError.

        Args:
            message: Error message.
            config_key: The configuration key with the error.
            expected: Expected value or type.
            got: Actual value received.
        """
        self.config_key = config_key
        self.expected = expected
        self.got = got

        super().__init__(
            message,
            context={
                "config_key": config_key,
                "expected": expected,
                "got": got,
            },
        )


class SelectionTimeoutError(LayerZeroError):
    """Raised when kernel selection times out.

    This can occur when:
    - PerfDB benchmarking takes too long
    - JIT compilation timeout is exceeded
    - Deadlock in selection cache
    """

    def __init__(
        self,
        message: str,
        *,
        timeout_seconds: Optional[float] = None,
        operation: Optional[str] = None,
    ) -> None:
        """Initialize SelectionTimeoutError.

        Args:
            message: Error message.
            timeout_seconds: The timeout that was exceeded.
            operation: The operation being selected.
        """
        self.timeout_seconds = timeout_seconds
        self.operation = operation

        super().__init__(
            message,
            context={
                "timeout_seconds": timeout_seconds,
                "operation": operation,
            },
        )


class CacheCorruptionError(LayerZeroError):
    """Raised when cache data is corrupted or inconsistent.

    This can occur when:
    - PerfDB data is corrupted
    - Selection cache has invalid entries
    - Hash collision detected
    """

    def __init__(
        self,
        message: str,
        *,
        cache_type: Optional[str] = None,
        cache_key: Optional[str] = None,
    ) -> None:
        """Initialize CacheCorruptionError.

        Args:
            message: Error message.
            cache_type: Type of cache (perfdb, selection, etc.).
            cache_key: The corrupted cache key.
        """
        self.cache_type = cache_type
        self.cache_key = cache_key

        super().__init__(
            message,
            context={"cache_type": cache_type, "cache_key": cache_key},
        )


# Aliases for compatibility with existing code
NoKernelAvailableError = NoKernelFoundError
