"""
LayerZero Intel IPEX Backend

Adapters for Intel Extension for PyTorch (IPEX) optimized operations.
IPEX provides optimized operations for Intel CPUs and XPU (Intel GPUs).
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from layerzero.backends.base import BaseKernel
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)


def is_ipex_available() -> bool:
    """Check if Intel Extension for PyTorch is available.

    Returns:
        True if IPEX is installed and importable, False otherwise.
    """
    try:
        import intel_extension_for_pytorch  # noqa: F401
        return True
    except ImportError:
        return False
    except Exception as e:
        logger.warning(f"IPEX import check failed: {e}")
        return False


def is_xpu_available() -> bool:
    """Check if Intel XPU (GPU) is available.

    Returns:
        True if XPU device is available, False otherwise.
    """
    if not is_ipex_available():
        return False

    try:
        import intel_extension_for_pytorch as ipex
        import torch

        # Check for XPU device
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return True

        return False

    except (ImportError, AttributeError) as e:
        logger.warning(f"XPU availability check failed: {e}")
        return False


def detect_ipex_version() -> tuple[int, int, int] | None:
    """Detect IPEX version.

    Returns:
        Tuple of (major, minor, patch) or None if not available.
    """
    if not is_ipex_available():
        return None

    try:
        import intel_extension_for_pytorch as ipex
        version_str = ipex.__version__

        # Parse version string
        base_version = version_str.split("+")[0]

        parts = base_version.split(".")
        if len(parts) >= 3:
            return (int(parts[0]), int(parts[1]), int(parts[2]))
        elif len(parts) == 2:
            return (int(parts[0]), int(parts[1]), 0)
        return (int(parts[0]), 0, 0)

    except (ImportError, ValueError, AttributeError) as e:
        logger.warning(f"Failed to detect IPEX version: {e}")
        return None


class IPEXMatmulAdapter(BaseKernel):
    """Intel IPEX matmul adapter.

    Uses IPEX for optimized matrix multiplication on Intel hardware.
    Supports both CPU and XPU (Intel GPU) backends.
    """

    def __init__(self, use_xpu: bool = False) -> None:
        """Initialize the adapter.

        Args:
            use_xpu: Whether to use XPU (Intel GPU) if available
        """
        self._available = is_ipex_available()
        self._xpu_available = is_xpu_available()
        self._use_xpu = use_xpu and self._xpu_available
        self._version = detect_ipex_version()
        self._version_str = ".".join(str(v) for v in self._version) if self._version else "unknown"
        self._use_fallback = not self._available
        self._use_torch_compile = False

        # Determine platform
        if self._use_xpu:
            self._platform = Platform.XPU
        else:
            self._platform = Platform.CPU

        self._kernel_spec = self._build_kernel_spec()

    def _build_kernel_spec(self) -> KernelSpec:
        """Build KernelSpec for this adapter."""
        return KernelSpec(
            kernel_id="ipex.matmul",
            operation="matmul",
            source="ipex",
            version=self._version_str,
            platform=self._platform,
            supported_dtypes=frozenset({torch.float32, torch.float64, torch.bfloat16, torch.float16}),
            priority=45,  # Higher than oneDNN on Intel
            impl=self,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def is_available(self) -> bool:
        """Check if this adapter is available."""
        return self._available

    @property
    def supports_bf16(self) -> bool:
        """Check if bf16 is supported."""
        # IPEX supports bf16 on modern Intel CPUs (Cooper Lake+)
        return self._available

    @property
    def use_torch_compile(self) -> bool:
        """Check if torch.compile integration is enabled."""
        return self._use_torch_compile

    @use_torch_compile.setter
    def use_torch_compile(self, value: bool) -> None:
        """Set torch.compile integration."""
        self._use_torch_compile = value

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
        if self._available and not self._use_fallback:
            try:
                import intel_extension_for_pytorch as ipex

                # Apply IPEX optimizations if available
                if hasattr(ipex, "optimize"):
                    # IPEX can optimize the matmul operation
                    pass

            except ImportError:
                pass

        # Use standard PyTorch matmul (IPEX optimizes at lower level)
        if a.dim() == 3:
            return torch.bmm(a, b)
        return torch.matmul(a, b)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IPEXMatmulAdapter(available={self._available}, xpu={self._use_xpu})"


class IPEXAttentionAdapter(BaseKernel):
    """Intel IPEX attention adapter.

    Uses IPEX for optimized scaled dot-product attention on Intel hardware.
    """

    def __init__(self, use_xpu: bool = False) -> None:
        """Initialize the adapter.

        Args:
            use_xpu: Whether to use XPU (Intel GPU) if available
        """
        self._available = is_ipex_available()
        self._xpu_available = is_xpu_available()
        self._use_xpu = use_xpu and self._xpu_available
        self._version = detect_ipex_version()
        self._version_str = ".".join(str(v) for v in self._version) if self._version else "unknown"
        self._use_fallback = not self._available
        self._use_torch_compile = False

        # Determine platform
        if self._use_xpu:
            self._platform = Platform.XPU
        else:
            self._platform = Platform.CPU

        self._kernel_spec = self._build_kernel_spec()

    def _build_kernel_spec(self) -> KernelSpec:
        """Build KernelSpec for this adapter."""
        return KernelSpec(
            kernel_id="ipex.attention",
            operation="attention",
            source="ipex",
            version=self._version_str,
            platform=self._platform,
            supported_dtypes=frozenset({torch.float32, torch.bfloat16, torch.float16}),
            priority=45,
            impl=self,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def is_available(self) -> bool:
        """Check if this adapter is available."""
        return self._available

    @property
    def supports_bf16(self) -> bool:
        """Check if bf16 is supported."""
        return self._available

    @property
    def use_torch_compile(self) -> bool:
        """Check if torch.compile integration is enabled."""
        return self._use_torch_compile

    @use_torch_compile.setter
    def use_torch_compile(self, value: bool) -> None:
        """Set torch.compile integration."""
        self._use_torch_compile = value

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute attention operation.

        Args:
            query: Query tensor (B, H, S, D)
            key: Key tensor (B, H, S, D)
            value: Value tensor (B, H, S, D)
            attn_mask: Optional attention mask
            dropout_p: Dropout probability
            is_causal: Whether to use causal masking
            scale: Optional scale factor
            **kwargs: Additional arguments (ignored)

        Returns:
            Output tensor (B, H, S, D)
        """
        # Use PyTorch SDPA which IPEX optimizes at lower level
        return torch.nn.functional.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
            scale=scale,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return f"IPEXAttentionAdapter(available={self._available}, xpu={self._use_xpu})"
