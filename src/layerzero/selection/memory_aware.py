"""
Memory-aware kernel selection.

This module provides:
- MemoryConfig: Configuration for memory-aware selection
- MemoryRequirement: Memory requirements for a kernel
- MemoryEstimator: Estimates memory requirements
- MemoryFilter: Filters kernels by memory constraints
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from layerzero.reasons import MEMORY_HEADROOM_EXCEEDED, Reason, ReasonCategory

if TYPE_CHECKING:
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MemoryConfig:
    """Configuration for memory-aware selection.

    Attributes:
        headroom_bytes: Explicit headroom in bytes. If None, uses fraction.
        headroom_fraction: Fraction of available memory to reserve (0.0-1.0).
        min_headroom_mb: Minimum headroom in megabytes.
        include_workspace: Include workspace memory in estimate.
        include_temp_buffers: Include temporary buffers in estimate.
    """

    headroom_bytes: int | None = None
    headroom_fraction: float = 0.1
    min_headroom_mb: int = 256
    include_workspace: bool = True
    include_temp_buffers: bool = True


@dataclass
class MemoryRequirement:
    """Memory requirements for kernel execution.

    Attributes:
        workspace_bytes: Workspace memory required by kernel.
        temp_buffer_bytes: Temporary buffer allocations.
        output_bytes: Output tensor size.
    """

    workspace_bytes: int = 0
    temp_buffer_bytes: int = 0
    output_bytes: int = 0

    @property
    def total_bytes(self) -> int:
        """Total memory requirement in bytes."""
        return self.workspace_bytes + self.temp_buffer_bytes + self.output_bytes

    @property
    def total_mb(self) -> float:
        """Total memory requirement in megabytes."""
        return self.total_bytes / (1024 * 1024)


class MemoryEstimator:
    """Estimates memory requirements for kernel execution.

    Estimates workspace, temporary buffers, and output memory
    based on kernel specification and execution context.

    Example:
        estimator = MemoryEstimator()

        requirement = estimator.estimate(kernel, ctx)
        print(f"Total memory: {requirement.total_mb:.2f} MB")
    """

    def __init__(self) -> None:
        """Initialize memory estimator."""
        # Dtype to bytes mapping
        self._dtype_sizes: dict[torch.dtype, int] = {
            torch.float32: 4,
            torch.float16: 2,
            torch.bfloat16: 2,
            torch.float64: 8,
            torch.int32: 4,
            torch.int64: 8,
            torch.int8: 1,
            torch.uint8: 1,
            torch.float8_e4m3fn: 1,
            torch.float8_e5m2: 1,
        }

    def estimate(
        self,
        kernel: "KernelSpec",
        ctx: "SelectionContext",
    ) -> MemoryRequirement:
        """Estimate total memory requirement.

        Args:
            kernel: Kernel specification.
            ctx: Selection context with shape info.

        Returns:
            MemoryRequirement with all estimates.
        """
        workspace = self.estimate_workspace(kernel, ctx)
        temp_buffers = self.estimate_temp_buffers(kernel, ctx)
        output = self.estimate_output(kernel, ctx)

        return MemoryRequirement(
            workspace_bytes=workspace,
            temp_buffer_bytes=temp_buffers,
            output_bytes=output,
        )

    def estimate_workspace(
        self,
        kernel: "KernelSpec",
        ctx: "SelectionContext",
    ) -> int:
        """Estimate workspace memory.

        Args:
            kernel: Kernel specification.
            ctx: Selection context.

        Returns:
            Workspace bytes required.
        """
        # Check if kernel has explicit workspace size
        if hasattr(kernel, 'workspace_bytes') and kernel.workspace_bytes is not None:
            return kernel.workspace_bytes

        # Estimate based on operation
        operation = getattr(kernel, 'operation', 'unknown')

        if operation.startswith('attention'):
            # Attention workspace scales with seq_len^2 for softmax
            batch = getattr(ctx, 'batch_size', 1)
            heads = getattr(ctx, 'num_heads', 8)
            seq_len = getattr(ctx, 'seq_len', 512)
            dtype_size = self._get_dtype_size(getattr(ctx, 'dtype', torch.float16))

            # Softmax scratch space: batch * heads * seq * seq * dtype
            workspace = batch * heads * seq_len * seq_len * dtype_size

            # Cap at reasonable maximum (1GB)
            return min(workspace, 1024 * 1024 * 1024)

        elif operation.startswith('matmul') or operation.startswith('linear'):
            # Matmul typically needs minimal workspace
            return 1024 * 1024  # 1MB default

        else:
            # Default workspace estimate
            return 512 * 1024  # 512KB default

    def estimate_temp_buffers(
        self,
        kernel: "KernelSpec",
        ctx: "SelectionContext",
    ) -> int:
        """Estimate temporary buffer allocations.

        Args:
            kernel: Kernel specification.
            ctx: Selection context.

        Returns:
            Temporary buffer bytes.
        """
        # Check if kernel has explicit temp buffer ratio
        if hasattr(kernel, 'temp_buffer_ratio'):
            ratio = kernel.temp_buffer_ratio or 0.0
        else:
            ratio = 0.0

        if ratio <= 0:
            return 0

        # Temp buffers as fraction of output
        output = self.estimate_output(kernel, ctx)
        return int(output * ratio)

    def estimate_output(
        self,
        kernel: "KernelSpec",
        ctx: "SelectionContext",
    ) -> int:
        """Estimate output tensor size.

        Args:
            kernel: Kernel specification.
            ctx: Selection context.

        Returns:
            Output bytes.
        """
        batch = getattr(ctx, 'batch_size', 1)
        seq_len = getattr(ctx, 'seq_len', 512)
        heads = getattr(ctx, 'num_heads', 8)
        head_dim = getattr(ctx, 'head_dim', 64)
        dtype = getattr(ctx, 'dtype', torch.float16)

        dtype_size = self._get_dtype_size(dtype)

        # Standard attention output: batch * seq * heads * head_dim
        return batch * seq_len * heads * head_dim * dtype_size

    def _get_dtype_size(self, dtype: torch.dtype) -> int:
        """Get size of dtype in bytes.

        Args:
            dtype: PyTorch dtype.

        Returns:
            Size in bytes.
        """
        return self._dtype_sizes.get(dtype, 2)  # Default to fp16 size


class MemoryFilter:
    """Filters kernels by memory constraints.

    Rejects kernels that would exceed configured memory headroom.

    Example:
        config = MemoryConfig(headroom_bytes=512 * 1024 * 1024)
        filter = MemoryFilter(config)

        passed, reason = filter.check(kernel, ctx)
        if not passed:
            print(f"Rejected: {reason}")
    """

    def __init__(
        self,
        config: MemoryConfig | None = None,
        estimator: MemoryEstimator | None = None,
    ) -> None:
        """Initialize memory filter.

        Args:
            config: Memory configuration.
            estimator: Memory estimator. Creates default if None.
        """
        self._config = config or MemoryConfig()
        self._estimator = estimator or MemoryEstimator()

    @property
    def config(self) -> MemoryConfig:
        """Get current configuration."""
        return self._config

    def update_config(self, config: MemoryConfig) -> None:
        """Update configuration.

        Args:
            config: New configuration.
        """
        self._config = config

    def check(
        self,
        kernel: "KernelSpec",
        ctx: "SelectionContext",
    ) -> tuple[bool, Reason | None]:
        """Check if kernel meets memory constraints.

        Args:
            kernel: Kernel specification.
            ctx: Selection context.

        Returns:
            Tuple of (passed, reason). Reason is None if passed.
        """
        # Estimate memory requirement
        requirement = self._estimate_requirement(kernel, ctx)

        # Get available memory and headroom
        available = self._get_available_memory()
        headroom = self.get_effective_headroom(total_memory=available)

        # Check if kernel fits
        if requirement.total_bytes > available - headroom:
            reason = Reason(
                code=MEMORY_HEADROOM_EXCEEDED,
                message=(
                    f"Kernel {getattr(kernel, 'kernel_id', 'unknown')} requires "
                    f"{requirement.total_mb:.2f}MB but only "
                    f"{(available - headroom) / (1024 * 1024):.2f}MB available "
                    f"(headroom: {headroom / (1024 * 1024):.2f}MB)"
                ),
                category=ReasonCategory.MEMORY,
            )
            logger.debug(
                "Memory filter rejected kernel %s: %s",
                getattr(kernel, 'kernel_id', 'unknown'),
                reason.message,
            )
            return False, reason

        return True, None

    def get_effective_headroom(self, total_memory: int) -> int:
        """Calculate effective memory headroom.

        Args:
            total_memory: Total available memory in bytes.

        Returns:
            Effective headroom in bytes.
        """
        # Use explicit headroom if configured
        if self._config.headroom_bytes is not None:
            return self._config.headroom_bytes

        # Calculate from fraction
        headroom_from_fraction = int(total_memory * self._config.headroom_fraction)

        # Enforce minimum
        min_headroom = self._config.min_headroom_mb * 1024 * 1024

        return max(headroom_from_fraction, min_headroom)

    def _estimate_requirement(
        self,
        kernel: "KernelSpec",
        ctx: "SelectionContext",
    ) -> MemoryRequirement:
        """Estimate memory requirement for kernel.

        Args:
            kernel: Kernel specification.
            ctx: Selection context.

        Returns:
            MemoryRequirement.
        """
        full_req = self._estimator.estimate(kernel, ctx)

        # Apply config filters
        workspace = full_req.workspace_bytes if self._config.include_workspace else 0
        temp = full_req.temp_buffer_bytes if self._config.include_temp_buffers else 0
        output = full_req.output_bytes

        return MemoryRequirement(
            workspace_bytes=workspace,
            temp_buffer_bytes=temp,
            output_bytes=output,
        )

    def _get_available_memory(self) -> int:
        """Get available GPU memory.

        Returns:
            Available memory in bytes.
        """
        if not torch.cuda.is_available():
            # Return a large value for CPU (effectively no limit)
            return 64 * 1024 * 1024 * 1024  # 64GB

        try:
            # Get free memory on current device
            free, total = torch.cuda.mem_get_info()
            return free
        except Exception as e:
            logger.warning("Failed to get CUDA memory info: %s", e)
            # Return conservative estimate
            return 4 * 1024 * 1024 * 1024  # 4GB


# Global memory filter instance
_global_memory_filter: MemoryFilter | None = None


def get_global_memory_filter() -> MemoryFilter:
    """Get global memory filter instance.

    Returns:
        Global MemoryFilter instance.
    """
    global _global_memory_filter
    if _global_memory_filter is None:
        _global_memory_filter = MemoryFilter()
    return _global_memory_filter


def check_memory_constraint(
    kernel: "KernelSpec",
    ctx: "SelectionContext",
) -> tuple[bool, Reason | None]:
    """Check memory constraint using global filter.

    Args:
        kernel: Kernel specification.
        ctx: Selection context.

    Returns:
        Tuple of (passed, reason).
    """
    return get_global_memory_filter().check(kernel, ctx)
