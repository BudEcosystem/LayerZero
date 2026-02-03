"""
LayerZero Triton Kernel Adapter

Adapter for custom Triton kernels that wraps user-defined kernels
and provides a consistent interface for the kernel selection system.
"""
from __future__ import annotations

import logging
from typing import Any, Callable, Set

import torch

from layerzero.backends.base import BaseKernel
from layerzero.backends.triton.version import (
    detect_triton_version,
    get_triton_backend,
    is_triton_available,
)
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec

logger = logging.getLogger(__name__)


class TritonKernelAdapter(BaseKernel):
    """Adapter for custom Triton kernels.

    Wraps a Triton kernel function and provides:
    - Automatic KernelSpec generation
    - Grid computation support
    - Cache key generation
    - Execution with proper device handling

    Example:
        ```python
        import triton
        import triton.language as tl

        @triton.jit
        def add_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
            pid = tl.program_id(0)
            block_start = pid * BLOCK_SIZE
            offsets = block_start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            tl.store(out_ptr + offsets, x + y, mask=mask)

        adapter = TritonKernelAdapter(
            name="my_add",
            kernel_fn=add_kernel,
            operation="math.add",
            supported_dtypes={torch.float16, torch.float32},
            grid_fn=lambda meta: (triton.cdiv(meta['n_elements'], meta['BLOCK_SIZE']),),
        )
        ```
    """

    def __init__(
        self,
        name: str,
        kernel_fn: Callable,
        operation: str,
        supported_dtypes: Set[torch.dtype],
        grid_fn: Callable | None = None,
        priority: int = 60,
        **kernel_spec_kwargs: Any,
    ) -> None:
        """Initialize adapter for a Triton kernel.

        Args:
            name: Unique name for this kernel.
            kernel_fn: Triton JIT-compiled kernel function.
            operation: Operation this kernel implements (e.g., "math.add").
            supported_dtypes: Set of supported input dtypes.
            grid_fn: Optional function to compute grid dimensions.
                     Takes metadata dict, returns grid tuple.
            priority: Selection priority (0-100, default: 60).
            **kernel_spec_kwargs: Additional KernelSpec parameters.

        Raises:
            RuntimeError: If Triton is not available.
        """
        if not is_triton_available():
            raise RuntimeError(
                "Triton is not available. Install with: pip install triton"
            )

        self._name = name
        self._kernel_fn = kernel_fn
        self._operation = operation
        self._supported_dtypes = frozenset(supported_dtypes)
        self._grid_fn = grid_fn
        self._priority = priority
        self._kernel_spec_kwargs = kernel_spec_kwargs

        # Get Triton version for spec
        self._triton_version = detect_triton_version()
        self._version_str = ".".join(str(v) for v in self._triton_version) if self._triton_version else "unknown"

        # Determine platform
        backend = get_triton_backend()
        if backend == "hip":
            self._platform = Platform.ROCM
        else:
            self._platform = Platform.CUDA

        # Cache the kernel spec
        self._kernel_spec = self._build_kernel_spec()

    def _build_kernel_spec(self) -> KernelSpec:
        """Build KernelSpec for this adapter.

        Returns:
            KernelSpec describing this kernel's capabilities.
        """
        return KernelSpec(
            kernel_id=f"triton.custom.{self._name}",
            operation=self._operation,
            source="triton.custom",
            version=self._version_str,
            platform=self._platform,
            supported_dtypes=self._supported_dtypes,
            priority=self._priority,
            impl=self,
            **self._kernel_spec_kwargs,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification.

        Returns:
            KernelSpec for this kernel.
        """
        return self._kernel_spec

    @property
    def name(self) -> str:
        """Get kernel name."""
        return self._name

    @property
    def kernel_fn(self) -> Callable:
        """Get underlying Triton kernel function."""
        return self._kernel_fn

    def __call__(
        self,
        *args: torch.Tensor,
        block_size: int = 256,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute the Triton kernel.

        This is a basic execution path that handles common patterns.
        For more complex kernels, subclass and override this method.

        Args:
            *args: Input tensors.
            block_size: Block size for kernel launch (default: 256).
            **kwargs: Additional kernel arguments.

        Returns:
            Output tensor.

        Raises:
            RuntimeError: If no input tensors provided.
        """
        if not args:
            raise RuntimeError("At least one input tensor is required")

        # Import triton for grid computation
        import triton

        # Get first tensor for shape/dtype reference
        first_tensor = args[0]
        n_elements = first_tensor.numel()

        # Allocate output tensor
        output = torch.empty_like(first_tensor)

        # Compute grid
        if self._grid_fn is not None:
            meta = {
                "n_elements": n_elements,
                "BLOCK_SIZE": block_size,
                **kwargs,
            }
            grid = self._grid_fn(meta)
        else:
            # Default grid: 1D based on number of elements
            grid = (triton.cdiv(n_elements, block_size),)

        # Handle different kernel signatures based on number of inputs
        if len(args) == 1:
            # Unary operation: x -> output
            x = args[0]
            # Handle scale parameter for unary ops
            scale = kwargs.get("scale", None)
            if scale is not None:
                self._kernel_fn[grid](
                    x,
                    output,
                    scale,
                    n_elements,
                    BLOCK_SIZE=block_size,
                )
            else:
                self._kernel_fn[grid](
                    x,
                    output,
                    n_elements,
                    BLOCK_SIZE=block_size,
                )
        elif len(args) == 2:
            # Binary operation: x, y -> output
            x, y = args
            self._kernel_fn[grid](
                x,
                y,
                output,
                n_elements,
                BLOCK_SIZE=block_size,
            )
        else:
            # Generic: pass all args plus output
            self._kernel_fn[grid](
                *args,
                output,
                n_elements,
                BLOCK_SIZE=block_size,
            )

        return output

    def __repr__(self) -> str:
        """Return string representation."""
        return (
            f"TritonKernelAdapter("
            f"name={self._name!r}, "
            f"operation={self._operation!r}, "
            f"platform={self._platform.value!r})"
        )
