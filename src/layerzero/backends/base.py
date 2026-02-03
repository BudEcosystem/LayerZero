"""
LayerZero Base Kernel

Abstract base class for all kernel adapters.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    from layerzero.models.kernel_spec import KernelSpec
    from layerzero.models.selection_context import SelectionContext
    from layerzero.reasons import Reason


class BaseKernel(ABC):
    """Abstract base class for kernel adapters.

    All kernel adapters must implement this interface.
    Provides a consistent API for kernel selection and execution.
    """

    @abstractmethod
    def get_kernel_spec(self) -> "KernelSpec":
        """Return kernel specification.

        The KernelSpec describes the kernel's capabilities, constraints,
        and metadata for the selection engine.

        Returns:
            KernelSpec for this kernel.
        """

    @abstractmethod
    def __call__(self, **kwargs) -> "torch.Tensor":
        """Execute the kernel.

        Args:
            **kwargs: Kernel-specific arguments.

        Returns:
            Output tensor.
        """

    def check_constraints(self, ctx: "SelectionContext") -> "list[Reason]":
        """Check if kernel can handle selection context.

        Default implementation delegates to KernelSpec.check().
        Subclasses can override for additional runtime checks.

        Args:
            ctx: Selection context to validate.

        Returns:
            Empty list if valid, else list of failure reasons.
        """
        return self.get_kernel_spec().check(ctx)

    @property
    def kernel_id(self) -> str:
        """Get kernel identifier."""
        return self.get_kernel_spec().kernel_id

    @property
    def operation(self) -> str:
        """Get operation this kernel implements."""
        return self.get_kernel_spec().operation

    @property
    def source(self) -> str:
        """Get source library."""
        return self.get_kernel_spec().source
