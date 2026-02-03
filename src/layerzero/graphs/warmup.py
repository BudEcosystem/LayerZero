"""
CUDA Graph warmup protocol.

This module provides:
- GraphWarmupProtocol: Warmup cuBLAS/cuDNN before graph capture
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from layerzero.models.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)


@dataclass
class WarmupState:
    """State of warmup completion.

    Attributes:
        cublas_initialized: Whether cuBLAS is initialized.
        cudnn_initialized: Whether cuDNN is initialized.
        workspaces_allocated: Whether workspaces are allocated.
        warmup_runs: Number of warmup runs completed.
        warmup_time_ms: Total warmup time in milliseconds.
        errors: List of errors during warmup.
    """

    cublas_initialized: bool = False
    cudnn_initialized: bool = False
    workspaces_allocated: bool = False
    warmup_runs: int = 0
    warmup_time_ms: float = 0.0
    errors: list[str] = field(default_factory=list)

    @property
    def is_ready(self) -> bool:
        """Check if warmup is complete."""
        return self.cublas_initialized and self.cudnn_initialized


class GraphWarmupProtocol:
    """Protocol for warming up before CUDA graph capture.

    CUDA graph capture requires that all CUDA libraries are initialized
    and any workspaces are allocated before capture begins. This protocol
    handles the warmup sequence to ensure capture succeeds.

    Warmup sequence:
    1. Initialize cuBLAS (via matmul)
    2. Initialize cuDNN (via attention or conv)
    3. Allocate workspaces
    4. Run target function N times

    Example:
        protocol = GraphWarmupProtocol()

        # Warmup before capture
        def my_attention(q, k, v):
            return F.scaled_dot_product_attention(q, k, v)

        protocol.warmup(my_attention, q, k, v)

        # Now safe to capture
        with torch.cuda.graph(g):
            output = my_attention(q, k, v)
    """

    def __init__(
        self,
        warmup_iterations: int = 3,
        device: str | torch.device = "cuda",
    ) -> None:
        """Initialize warmup protocol.

        Args:
            warmup_iterations: Number of warmup runs.
            device: Device for warmup.
        """
        self._warmup_iterations = warmup_iterations
        self._device = torch.device(device) if isinstance(device, str) else device
        self._state = WarmupState()
        self._warmup_history: list[dict[str, Any]] = []

    @property
    def state(self) -> WarmupState:
        """Get current warmup state."""
        return self._state

    @property
    def is_warmed_up(self) -> bool:
        """Check if warmup is complete."""
        return self._state.is_ready

    @property
    def cublas_initialized(self) -> bool:
        """Check if cuBLAS is initialized."""
        return self._state.cublas_initialized

    @property
    def cudnn_initialized(self) -> bool:
        """Check if cuDNN is initialized."""
        return self._state.cudnn_initialized

    @property
    def warmup_history(self) -> list[dict[str, Any]]:
        """Get warmup history for debugging."""
        return self._warmup_history.copy()

    def warmup(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ) -> WarmupState:
        """Execute warmup before graph capture.

        Args:
            func: Function to warm up.
            *args: Arguments to pass to function.
            **kwargs: Keyword arguments to pass.

        Returns:
            WarmupState with warmup results.
        """
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, skipping warmup")
            return self._state

        start_time = time.perf_counter()

        # Step 1: Initialize cuBLAS
        self._init_cublas()

        # Step 2: Initialize cuDNN
        self._init_cudnn()

        # Step 3: Warmup the target function
        for i in range(self._warmup_iterations):
            try:
                func(*args, **kwargs)
                self._state.warmup_runs += 1

                self._warmup_history.append({
                    "iteration": i,
                    "success": True,
                    "time_ms": (time.perf_counter() - start_time) * 1000,
                })
            except Exception as e:
                error_msg = f"Warmup iteration {i} failed: {e}"
                logger.warning(error_msg)
                self._state.errors.append(error_msg)
                self._warmup_history.append({
                    "iteration": i,
                    "success": False,
                    "error": str(e),
                })

        # Synchronize after warmup
        torch.cuda.synchronize()

        self._state.warmup_time_ms = (time.perf_counter() - start_time) * 1000
        self._state.workspaces_allocated = True

        return self._state

    def warmup_cublas(self) -> bool:
        """Initialize cuBLAS.

        Returns:
            True if successful.
        """
        return self._init_cublas()

    def warmup_cudnn(self) -> bool:
        """Initialize cuDNN.

        Returns:
            True if successful.
        """
        return self._init_cudnn()

    def _init_cublas(self) -> bool:
        """Initialize cuBLAS via matmul.

        Returns:
            True if successful.
        """
        if self._state.cublas_initialized:
            return True

        if not torch.cuda.is_available():
            return False

        try:
            # cuBLAS initialization via matmul
            dummy = torch.randn(64, 64, device=self._device)
            _ = torch.matmul(dummy, dummy)
            torch.cuda.synchronize()
            self._state.cublas_initialized = True
            logger.debug("cuBLAS initialized")
            return True
        except Exception as e:
            error_msg = f"cuBLAS initialization failed: {e}"
            logger.error(error_msg)
            self._state.errors.append(error_msg)
            return False

    def _init_cudnn(self) -> bool:
        """Initialize cuDNN via attention.

        Returns:
            True if successful.
        """
        if self._state.cudnn_initialized:
            return True

        if not torch.cuda.is_available():
            return False

        try:
            import torch.nn.functional as F

            # cuDNN initialization via attention
            q = torch.randn(1, 1, 8, 32, device=self._device)
            k = torch.randn(1, 1, 8, 32, device=self._device)
            v = torch.randn(1, 1, 8, 32, device=self._device)
            _ = F.scaled_dot_product_attention(q, k, v)
            torch.cuda.synchronize()
            self._state.cudnn_initialized = True
            logger.debug("cuDNN initialized")
            return True
        except Exception as e:
            error_msg = f"cuDNN initialization failed: {e}"
            logger.error(error_msg)
            self._state.errors.append(error_msg)
            return False

    def reset(self) -> None:
        """Reset warmup state."""
        self._state = WarmupState()
        self._warmup_history.clear()


# Global warmup protocol instance
_global_warmup: GraphWarmupProtocol | None = None


def get_global_warmup() -> GraphWarmupProtocol:
    """Get global warmup protocol instance.

    Returns:
        Global GraphWarmupProtocol instance.
    """
    global _global_warmup
    if _global_warmup is None:
        _global_warmup = GraphWarmupProtocol()
    return _global_warmup


def ensure_warmed_up(
    func: Callable[..., Any],
    *args: Any,
    **kwargs: Any,
) -> WarmupState:
    """Ensure warmup is done before graph capture.

    Uses global warmup protocol. Safe to call multiple times.

    Args:
        func: Function to warm up.
        *args: Arguments to pass.
        **kwargs: Keyword arguments to pass.

    Returns:
        WarmupState with warmup results.
    """
    protocol = get_global_warmup()
    if not protocol.is_warmed_up:
        return protocol.warmup(func, *args, **kwargs)
    return protocol.state
