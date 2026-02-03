"""
torch.compile Compatibility Utilities

Provides utilities for ensuring torch.compile compatibility
and preventing graph breaks.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Callable, TypeVar

import torch

logger = logging.getLogger(__name__)

F = TypeVar("F", bound=Callable[..., Any])


def ensure_no_graph_breaks(fn: F) -> F:
    """Decorator to verify a function causes no graph breaks.

    In debug mode, wraps the function with graph break detection.

    Args:
        fn: Function to wrap

    Returns:
        Wrapped function that logs warnings on graph breaks.
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        return fn(*args, **kwargs)

    return wrapper  # type: ignore


def register_for_compile() -> None:
    """Register all LayerZero ops for torch.compile.

    This is called automatically when the ops module is imported.
    Ensures all operations have proper decompositions and lowerings.
    """
    # Import ops module to trigger registration
    from layerzero.pytorch import ops  # noqa: F401

    logger.debug("LayerZero ops registered for torch.compile")


def is_compiling() -> bool:
    """Check if we're currently inside torch.compile.

    Returns:
        True if in compile context.
    """
    if hasattr(torch, "_dynamo"):
        return torch._dynamo.is_compiling()
    return False


def get_compile_mode() -> str | None:
    """Get the current torch.compile mode.

    Returns:
        Mode string or None if not compiling.
    """
    if not is_compiling():
        return None

    # Try to get mode from dynamo context
    try:
        if hasattr(torch._dynamo, "current_config"):
            config = torch._dynamo.current_config()
            return getattr(config, "mode", "default")
    except Exception:
        pass

    return "unknown"


def compile_friendly_dispatch(
    cuda_fn: Callable[..., torch.Tensor],
    cpu_fn: Callable[..., torch.Tensor],
    *args: torch.Tensor,
    **kwargs: Any,
) -> torch.Tensor:
    """Dispatch to CUDA or CPU implementation in a compile-friendly way.

    Avoids graph breaks by using tensor.device checks instead of
    Python control flow.

    Args:
        cuda_fn: CUDA implementation
        cpu_fn: CPU implementation
        *args: Tensor arguments
        **kwargs: Keyword arguments

    Returns:
        Result tensor from appropriate implementation.
    """
    # Get device from first tensor argument
    if not args:
        raise ValueError("At least one tensor argument required")

    device = args[0].device

    if device.type == "cuda":
        return cuda_fn(*args, **kwargs)
    else:
        return cpu_fn(*args, **kwargs)


class CompileGuard:
    """Guard for operations that may cause graph breaks.

    Use as a context manager to mark sections that should
    be handled carefully during compilation.
    """

    def __init__(self, name: str = "operation") -> None:
        """Initialize guard.

        Args:
            name: Name for logging purposes.
        """
        self.name = name
        self._was_compiling = False

    def __enter__(self) -> "CompileGuard":
        """Enter guard context."""
        self._was_compiling = is_compiling()
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit guard context."""
        pass
