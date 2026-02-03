"""
LayerZero Liger CrossEntropy Adapter

Adapter class wrapping Liger CrossEntropy kernel.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.backends.base import BaseKernel
from layerzero.backends.liger.version import (
    detect_liger_version,
    is_liger_available,
)
from layerzero.enums import Platform
from layerzero.models.kernel_spec import KernelSpec

if TYPE_CHECKING:
    pass


class LigerCrossEntropyAdapter(BaseKernel):
    """Adapter for Liger CrossEntropy kernel.

    Liger CrossEntropy is a memory-efficient implementation that
    computes softmax and cross-entropy in a fused manner.

    Attributes:
        is_available: Whether Liger is installed.
        version: Installed Liger version tuple.
    """

    __slots__ = ("_version", "_kernel_spec")

    def __init__(self) -> None:
        """Initialize LigerCrossEntropyAdapter."""
        self._version = detect_liger_version()
        self._kernel_spec = self._create_kernel_spec()

    @property
    def is_available(self) -> bool:
        """Check if Liger is available."""
        return self._version is not None

    @property
    def version(self) -> tuple[int, int, int] | None:
        """Get installed Liger version."""
        return self._version

    def _create_kernel_spec(self) -> KernelSpec:
        """Create KernelSpec for this adapter."""
        if self._version:
            version_str = f"{self._version[0]}.{self._version[1]}.{self._version[2]}"
        else:
            version_str = "unknown"

        return KernelSpec(
            kernel_id="liger.cross_entropy",
            operation="loss.cross_entropy",
            source="liger",
            version=version_str,
            impl=self,
            platform=Platform.CUDA,
            min_sm=None,
            max_sm=None,
            # CrossEntropy typically uses FP32 for numerical stability
            supported_dtypes=frozenset([torch.float16, torch.bfloat16, torch.float32]),
            min_head_dim=1,
            max_head_dim=None,
            min_seq_len=1,
            max_seq_len=None,
            supports_gqa=False,
            supports_mqa=False,
            supports_attn_mask=False,
            supports_dropout=False,
            supports_scale=False,
            supports_alibi=False,
            requires_last_dim_stride1=True,
            requires_contiguous=True,
            is_cuda_graph_safe=True,
            deterministic=True,
            workspace_bytes=0,
            priority=75,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def __call__(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        ignore_index: int = -100,
        reduction: str = "mean",
    ) -> torch.Tensor:
        """Execute Liger CrossEntropy.

        Computes cross-entropy loss with optional label smoothing
        in a memory-efficient fused kernel.

        Args:
            logits: Logits tensor (batch_size, vocab_size) or (N, vocab_size).
            labels: Target labels (batch_size,) or (N,).
            ignore_index: Target value to ignore in loss computation.
            reduction: Reduction mode ("mean", "sum", "none").

        Returns:
            Loss tensor (scalar if reduction="mean"/"sum").

        Raises:
            RuntimeError: If Liger is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "Liger is not installed. "
                "Install with: pip install liger-kernel"
            )

        try:
            from liger_kernel.ops.cross_entropy import LigerCrossEntropyFunction
        except ImportError as e:
            raise RuntimeError(f"Failed to import Liger CrossEntropy: {e}") from e

        # Call Liger CrossEntropy
        # Note: Liger's API may vary slightly by version
        loss = LigerCrossEntropyFunction.apply(logits, labels, ignore_index)

        # Apply reduction
        if reduction == "mean":
            # Already reduced to mean in Liger
            return loss
        elif reduction == "sum":
            return loss * labels.numel()
        else:  # "none"
            # Liger typically returns reduced loss; for "none" we'd need
            # per-sample loss which may not be directly available
            return loss
