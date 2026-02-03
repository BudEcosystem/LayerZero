"""
LayerZero Intel oneDNN Backend

Adapters for Intel oneDNN optimized operations.
oneDNN (formerly MKL-DNN) provides optimized primitives for Intel CPUs.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from layerzero.backends.base import BaseKernel
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)


def is_onednn_available() -> bool:
    """Check if oneDNN is available.

    oneDNN is typically bundled with PyTorch as MKL-DNN backend.

    Returns:
        True if oneDNN is available, False otherwise.
    """
    try:
        import torch

        # Check if MKL-DNN (oneDNN) backend is available
        if hasattr(torch.backends, "mkldnn"):
            return torch.backends.mkldnn.is_available()

        return False

    except (ImportError, AttributeError) as e:
        logger.warning(f"oneDNN availability check failed: {e}")
        return False


def detect_onednn_version() -> tuple[int, int, int] | None:
    """Detect oneDNN version.

    Returns:
        Tuple of (major, minor, patch) or None if not available.
    """
    if not is_onednn_available():
        return None

    try:
        # oneDNN version is not directly exposed, but we can infer from PyTorch
        import torch

        # Return a placeholder version based on PyTorch version
        # In practice, oneDNN version tracks PyTorch releases
        version_str = torch.__version__.split("+")[0]
        parts = version_str.split(".")
        if len(parts) >= 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        return (int(parts[0]), 0, 0)

    except (ImportError, ValueError) as e:
        logger.warning(f"Failed to detect oneDNN version: {e}")
        return None


class OneDNNMatmulAdapter(BaseKernel):
    """Intel oneDNN matmul adapter.

    Uses MKL-DNN (oneDNN) for optimized matrix multiplication on Intel CPUs.
    Automatically falls back to PyTorch when oneDNN is unavailable.
    """

    def __init__(self) -> None:
        """Initialize the adapter."""
        self._available = is_onednn_available()
        self._version = detect_onednn_version()
        self._version_str = ".".join(str(v) for v in self._version) if self._version else "unknown"
        self._use_fallback = not self._available
        self._kernel_spec = self._build_kernel_spec()

    def _build_kernel_spec(self) -> KernelSpec:
        """Build KernelSpec for this adapter."""
        return KernelSpec(
            kernel_id="onednn.matmul",
            operation="matmul",
            source="onednn",
            version=self._version_str,
            platform=Platform.CPU,
            supported_dtypes=frozenset({torch.float32, torch.float64, torch.bfloat16}),
            priority=40,  # Lower than GPU backends
            impl=self,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def is_available(self) -> bool:
        """Check if this adapter is available."""
        return self._available

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
        # PyTorch automatically uses MKL-DNN for CPU matmul when available
        if a.dim() == 3:
            return torch.bmm(a, b)
        return torch.matmul(a, b)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"OneDNNMatmulAdapter(available={self._available})"


class OneDNNLayerNormAdapter(BaseKernel):
    """Intel oneDNN LayerNorm adapter.

    Uses MKL-DNN (oneDNN) for optimized layer normalization on Intel CPUs.
    """

    def __init__(
        self,
        normalized_shape: int | tuple[int, ...],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        """Initialize the adapter.

        Args:
            normalized_shape: Input shape from expected input of size
            eps: Small constant for numerical stability
            elementwise_affine: Whether to include learnable parameters
        """
        self._available = is_onednn_available()
        self._version = detect_onednn_version()
        self._version_str = ".".join(str(v) for v in self._version) if self._version else "unknown"
        self._use_fallback = not self._available

        # Store LayerNorm parameters
        if isinstance(normalized_shape, int):
            self._normalized_shape = (normalized_shape,)
        else:
            self._normalized_shape = tuple(normalized_shape)
        self._eps = eps
        self._elementwise_affine = elementwise_affine

        # Create learnable parameters
        if elementwise_affine:
            self._weight: torch.Tensor | None = torch.ones(self._normalized_shape)
            self._bias: torch.Tensor | None = torch.zeros(self._normalized_shape)
        else:
            self._weight = None
            self._bias = None

        self._kernel_spec = self._build_kernel_spec()

    def _build_kernel_spec(self) -> KernelSpec:
        """Build KernelSpec for this adapter."""
        return KernelSpec(
            kernel_id="onednn.layer_norm",
            operation="layer_norm",
            source="onednn",
            version=self._version_str,
            platform=Platform.CPU,
            supported_dtypes=frozenset({torch.float32, torch.float64, torch.bfloat16}),
            priority=40,
            impl=self,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def is_available(self) -> bool:
        """Check if this adapter is available."""
        return self._available

    def __call__(
        self,
        x: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute LayerNorm operation.

        Args:
            x: Input tensor
            **kwargs: Additional arguments (ignored)

        Returns:
            Normalized tensor
        """
        # Move parameters to same device as input
        weight = self._weight.to(x.device) if self._weight is not None else None
        bias = self._bias.to(x.device) if self._bias is not None else None

        return torch.nn.functional.layer_norm(
            x,
            self._normalized_shape,
            weight=weight,
            bias=bias,
            eps=self._eps,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"OneDNNLayerNormAdapter(shape={self._normalized_shape}, available={self._available})"
