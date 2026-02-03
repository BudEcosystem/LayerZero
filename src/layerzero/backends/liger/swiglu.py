"""
LayerZero Liger SwiGLU Adapter

Adapter class wrapping Liger SwiGLU (SiLU * x) kernel.
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


class LigerSwiGLUAdapter(BaseKernel):
    """Adapter for Liger SwiGLU kernel.

    SwiGLU is the activation function used in LLaMA and many modern LLMs:
    SwiGLU(gate, up) = SiLU(gate) * up

    Liger's implementation is fused for better performance.

    Attributes:
        is_available: Whether Liger is installed.
        version: Installed Liger version tuple.
    """

    __slots__ = ("_version", "_kernel_spec")

    def __init__(self) -> None:
        """Initialize LigerSwiGLUAdapter."""
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
            kernel_id="liger.swiglu",
            operation="activation.swiglu",
            source="liger",
            version=version_str,
            impl=self,
            platform=Platform.CUDA,
            min_sm=None,
            max_sm=None,
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
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> torch.Tensor:
        """Execute Liger SwiGLU.

        Computes SwiGLU(gate, up) = SiLU(gate) * up in a fused kernel.

        Args:
            gate: Gate tensor from first linear projection.
            up: Up tensor from second linear projection.

        Returns:
            Output tensor (same shape as inputs).

        Raises:
            RuntimeError: If Liger is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "Liger is not installed. "
                "Install with: pip install liger-kernel"
            )

        try:
            from liger_kernel.ops.swiglu import LigerSiLUMulFunction
        except ImportError as e:
            raise RuntimeError(f"Failed to import Liger SwiGLU: {e}") from e

        # Call Liger SwiGLU (SiLU * x)
        output = LigerSiLUMulFunction.apply(gate, up)

        return output
