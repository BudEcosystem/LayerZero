"""
LayerZero AMD ZenDNN Backend

Adapters for AMD ZenDNN optimized operations.
ZenDNN provides optimized primitives for AMD EPYC CPUs.
"""
from __future__ import annotations

import logging
import os
from typing import Any

import torch

from layerzero.backends.base import BaseKernel
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)


def is_zendnn_available() -> bool:
    """Check if ZenDNN is available.

    ZenDNN is typically integrated at PyTorch build time for AMD systems.

    Returns:
        True if ZenDNN is available, False otherwise.
    """
    try:
        # Check for ZenDNN environment variable
        zendnn_enable = os.environ.get("ZENDNN_ENABLE", "").lower()
        if zendnn_enable in ("1", "true"):
            return True

        # Check for ZenDNN-specific PyTorch build
        import torch
        if hasattr(torch, "_C") and hasattr(torch._C, "_has_zendnn"):
            return torch._C._has_zendnn

        return False

    except (ImportError, AttributeError) as e:
        logger.warning(f"ZenDNN availability check failed: {e}")
        return False


def is_aocl_blas_available() -> bool:
    """Check if AMD AOCL-BLAS is available.

    AOCL-BLAS provides optimized BLAS routines for AMD CPUs.

    Returns:
        True if AOCL-BLAS is available, False otherwise.
    """
    try:
        # Check for BLIS library (part of AOCL)
        import ctypes

        try:
            ctypes.CDLL("libblis.so")
            return True
        except OSError:
            pass

        return False

    except Exception as e:
        logger.warning(f"AOCL-BLAS availability check failed: {e}")
        return False


def detect_zendnn_version() -> tuple[int, int, int] | None:
    """Detect ZenDNN version.

    Returns:
        Tuple of (major, minor, patch) or None if not available.
    """
    if not is_zendnn_available():
        return None

    try:
        # ZenDNN version from environment or build
        zendnn_version = os.environ.get("ZENDNN_VERSION", "")
        if zendnn_version:
            parts = zendnn_version.split(".")
            if len(parts) >= 3:
                return (int(parts[0]), int(parts[1]), int(parts[2]))
            elif len(parts) == 2:
                return (int(parts[0]), int(parts[1]), 0)
            return (int(parts[0]), 0, 0)

        # Fallback to default version
        return (4, 0, 0)  # ZenDNN 4.0 default

    except (ValueError, AttributeError) as e:
        logger.warning(f"Failed to detect ZenDNN version: {e}")
        return None


class ZenDNNMatmulAdapter(BaseKernel):
    """AMD ZenDNN matmul adapter.

    Uses ZenDNN for optimized matrix multiplication on AMD EPYC CPUs.
    Automatically falls back to PyTorch when ZenDNN is unavailable.
    """

    def __init__(self, num_threads: int | None = None) -> None:
        """Initialize the adapter.

        Args:
            num_threads: Number of threads to use (default: OMP_NUM_THREADS)
        """
        self._available = is_zendnn_available()
        self._version = detect_zendnn_version()
        self._version_str = ".".join(str(v) for v in self._version) if self._version else "unknown"
        self._use_fallback = not self._available

        # Thread configuration
        if num_threads is not None:
            self._num_threads = num_threads
        else:
            self._num_threads = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 1))

        self._kernel_spec = self._build_kernel_spec()

    def _build_kernel_spec(self) -> KernelSpec:
        """Build KernelSpec for this adapter."""
        return KernelSpec(
            kernel_id="zendnn.matmul",
            operation="matmul",
            source="zendnn",
            version=self._version_str,
            platform=Platform.CPU,
            supported_dtypes=frozenset({torch.float32, torch.float64, torch.bfloat16}),
            priority=42,  # Slightly higher than oneDNN for AMD systems
            impl=self,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def is_available(self) -> bool:
        """Check if this adapter is available."""
        return self._available

    def is_epyc_optimized(self) -> bool:
        """Check if EPYC-specific optimizations are active.

        Returns:
            True if running on AMD EPYC with optimizations enabled.
        """
        if not self._available:
            return False

        try:
            from layerzero.backends.cpu.detection import CPUVendor, detect_cpu_vendor
            return detect_cpu_vendor() == CPUVendor.AMD
        except ImportError:
            return False

    def __call__(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute matmul operation.

        Args:
            a: First input tensor (..., M, K)
            b: Second input tensor (..., K, N)
            **kwargs: Additional arguments (ignored)

        Returns:
            Result tensor (..., M, N)
        """
        # Set thread count for this operation
        old_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(self._num_threads)

            if a.dim() == 3:
                return torch.bmm(a, b)
            return torch.matmul(a, b)
        finally:
            torch.set_num_threads(old_threads)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"ZenDNNMatmulAdapter(available={self._available}, threads={self._num_threads})"
