"""
LayerZero HuggingFace Kernel Hub Adapter

Adapter for kernels loaded from HuggingFace Hub.
"""
from __future__ import annotations

import logging
from typing import Any

import torch

from layerzero.backends.base import BaseKernel
from layerzero.backends.hf_kernels.loader import HFKernelLoader, LoadedKernel
from layerzero.backends.hf_kernels.validator import validate_torch_ops_namespace
from layerzero.backends.hf_kernels.version import is_hf_kernels_available
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)


class HFKernelAdapter(BaseKernel):
    """Adapter for HuggingFace Hub kernels.

    Loads and adapts kernels from the HuggingFace Kernel Hub
    for use in the LayerZero kernel selection system.

    Example:
        ```python
        adapter = HFKernelAdapter(
            kernel_name="flash_attn",
            version="2.6.0",
        )
        if adapter.is_available():
            result = adapter(query, key, value)
        ```
    """

    def __init__(
        self,
        kernel_name: str,
        version: str | None = None,
        operation: str | None = None,
        priority: int = 55,
    ) -> None:
        """Initialize the adapter.

        Args:
            kernel_name: Name of the kernel to load
            version: Optional version string
            operation: Optional operation name (auto-detected if None)
            priority: Selection priority (default: 55)
        """
        self._kernel_name = kernel_name
        self._version = version or "latest"
        self._operation = operation or self._detect_operation(kernel_name)
        self._priority = priority

        self._loader = HFKernelLoader()
        self._loaded_kernel: LoadedKernel | None = None
        self._available = False
        self._kernel_fn = None

        # Try to load the kernel
        self._try_load()

        self._kernel_spec = self._build_kernel_spec()

    def _detect_operation(self, kernel_name: str) -> str:
        """Detect operation from kernel name.

        Args:
            kernel_name: Name of the kernel

        Returns:
            Detected operation name.
        """
        # Common kernel name patterns
        if "attn" in kernel_name.lower() or "attention" in kernel_name.lower():
            return "attention"
        if "matmul" in kernel_name.lower() or "gemm" in kernel_name.lower():
            return "matmul"
        if "norm" in kernel_name.lower():
            return "layer_norm"
        if "rope" in kernel_name.lower():
            return "rope"
        if "softmax" in kernel_name.lower():
            return "softmax"

        return "custom"

    def _try_load(self) -> None:
        """Try to load the kernel from Hub."""
        if not is_hf_kernels_available():
            return

        try:
            kernel = self._loader.load(
                self._kernel_name,
                version=self._version if self._version != "latest" else None,
            )
            if kernel:
                self._loaded_kernel = kernel
                self._available = True
                logger.debug(f"Loaded HF kernel: {kernel.name} v{kernel.version}")

        except Exception as e:
            logger.debug(f"Failed to load HF kernel {self._kernel_name}: {e}")

    def _build_kernel_spec(self) -> KernelSpec:
        """Build KernelSpec for this adapter."""
        version_str = self._version
        if self._loaded_kernel:
            version_str = self._loaded_kernel.version

        return KernelSpec(
            kernel_id=f"hf.{self._kernel_name}.{version_str}",
            operation=self._operation,
            source="hf_kernels",
            version=version_str,
            platform=Platform.CUDA,
            supported_dtypes=frozenset({torch.float16, torch.bfloat16, torch.float32}),
            priority=self._priority,
            impl=self,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def is_available(self) -> bool:
        """Check if the kernel is available."""
        return self._available

    def check_namespace_clash(
        self,
        existing_namespaces: set[str],
    ) -> list[str]:
        """Check for torch.ops namespace clashes.

        Args:
            existing_namespaces: Set of existing namespace names

        Returns:
            List of clash messages. Empty if no clashes.
        """
        return validate_torch_ops_namespace(
            self._kernel_name,
            existing_namespaces,
        )

    def register(self) -> bool:
        """Register the kernel with torch.ops.

        Returns:
            True if registration successful, False otherwise.
        """
        if not self._available:
            return False

        # Registration would involve loading the .so file
        # and registering with torch.ops
        # For now, just return True if kernel is loaded
        return self._loaded_kernel is not None

    def __call__(
        self,
        *args: torch.Tensor,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute the kernel.

        Args:
            *args: Input tensors
            **kwargs: Additional arguments

        Returns:
            Output tensor

        Raises:
            RuntimeError: If kernel is not available
        """
        if not self._available:
            raise RuntimeError(
                f"HF kernel '{self._kernel_name}' is not available. "
                "Install huggingface_hub and ensure the kernel exists."
            )

        # For now, raise NotImplementedError as we need actual
        # kernel loading and execution logic
        raise NotImplementedError(
            f"Kernel execution for '{self._kernel_name}' not implemented. "
            "This requires dynamic loading of the kernel library."
        )

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"HFKernelAdapter("
            f"kernel_name={self._kernel_name!r}, "
            f"version={self._version!r}, "
            f"available={self._available})"
        )
