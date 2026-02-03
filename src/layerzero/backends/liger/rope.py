"""
LayerZero Liger RoPE Adapter

Adapter class wrapping Liger Rotary Position Embedding kernel.
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


class LigerRoPEAdapter(BaseKernel):
    """Adapter for Liger RoPE (Rotary Position Embedding) kernel.

    Liger RoPE is a Triton-based implementation optimized for
    LLaMA-style rotary embeddings.

    Attributes:
        is_available: Whether Liger is installed.
        version: Installed Liger version tuple.
    """

    __slots__ = ("_version", "_kernel_spec")

    def __init__(self) -> None:
        """Initialize LigerRoPEAdapter."""
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
            kernel_id="liger.rope",
            operation="embedding.rope",
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
            supports_gqa=True,  # RoPE works with GQA
            supports_mqa=True,
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
        q: torch.Tensor,
        k: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        unsqueeze_dim: int = 1,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Execute Liger RoPE.

        Applies rotary position embedding to query and key tensors.

        Args:
            q: Query tensor (batch, seq, heads, dim).
            k: Key tensor (batch, seq, heads_k, dim).
            cos: Cosine frequencies (seq, dim) or (batch, seq, dim).
            sin: Sine frequencies (seq, dim) or (batch, seq, dim).
            position_ids: Optional position indices (batch, seq).
            unsqueeze_dim: Dimension to unsqueeze cos/sin for broadcasting.

        Returns:
            Tuple of (q_rotated, k_rotated) tensors.

        Raises:
            RuntimeError: If Liger is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "Liger is not installed. "
                "Install with: pip install liger-kernel"
            )

        try:
            from liger_kernel.ops.rope import LigerRopeFunction
        except ImportError as e:
            raise RuntimeError(f"Failed to import Liger RoPE: {e}") from e

        # Call Liger RoPE
        q_out, k_out = LigerRopeFunction.apply(
            q, k, cos, sin, position_ids, unsqueeze_dim
        )

        return q_out, k_out
