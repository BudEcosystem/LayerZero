"""
LayerZero Liger RMSNorm Adapter

Adapter class wrapping Liger RMSNorm kernel.
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


class LigerRMSNormAdapter(BaseKernel):
    """Adapter for Liger RMSNorm kernel.

    Liger RMSNorm is a Triton-based implementation that is typically
    20-30% faster than PyTorch's equivalent.

    Attributes:
        is_available: Whether Liger is installed.
        version: Installed Liger version tuple.
    """

    __slots__ = ("_version", "_kernel_spec")

    def __init__(self) -> None:
        """Initialize LigerRMSNormAdapter."""
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
            kernel_id="liger.rms_norm",
            operation="norm.rms",
            source="liger",
            version=version_str,
            impl=self,
            platform=Platform.CUDA,  # Also works on ROCm via Triton
            min_sm=None,  # Triton handles compatibility
            max_sm=None,
            supported_dtypes=frozenset([torch.float16, torch.bfloat16, torch.float32]),
            min_head_dim=1,
            max_head_dim=None,
            min_seq_len=1,
            max_seq_len=None,
            supports_gqa=False,  # Not attention kernel
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
            priority=75,  # Good priority for normalization
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def __call__(
        self,
        hidden_states: torch.Tensor,
        weight: torch.Tensor,
        eps: float = 1e-5,
    ) -> torch.Tensor:
        """Execute Liger RMSNorm.

        Args:
            hidden_states: Input tensor (..., hidden_dim).
            weight: RMSNorm weight tensor (hidden_dim,).
            eps: Epsilon for numerical stability.

        Returns:
            Normalized output tensor (same shape as input).

        Raises:
            RuntimeError: If Liger is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "Liger is not installed. "
                "Install with: pip install liger-kernel"
            )

        try:
            from liger_kernel.ops.rms_norm import LigerRMSNormFunction
        except ImportError as e:
            raise RuntimeError(f"Failed to import Liger RMSNorm: {e}") from e

        # Call Liger RMSNorm
        output = LigerRMSNormFunction.apply(hidden_states, weight, eps)

        return output
