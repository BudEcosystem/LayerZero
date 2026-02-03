"""
LayerZero FlashAttention Adapter

Adapter class wrapping FlashAttention library.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.backends.base import BaseKernel
from layerzero.backends.flash_attn.version import (
    detect_flash_attn_version,
    is_flash_attn_available,
    select_fa_variant,
    FAVariant,
)
from layerzero.enums import Layout, MaskType, Platform
from layerzero.models.kernel_spec import KernelSpec

if TYPE_CHECKING:
    pass


class FlashAttnAdapter(BaseKernel):
    """Adapter for FlashAttention library.

    Automatically selects FA2/FA3/FA4 based on GPU SM version.
    Handles layout conversion between BHSD (PyTorch) and BSHD (Flash).

    Attributes:
        is_available: Whether flash_attn is installed.
        version: Installed flash_attn version tuple.
    """

    __slots__ = ("_version", "_kernel_spec")

    def __init__(self) -> None:
        """Initialize FlashAttnAdapter."""
        self._version = detect_flash_attn_version()
        self._kernel_spec = self._create_kernel_spec()

    @property
    def is_available(self) -> bool:
        """Check if FlashAttention is available."""
        return self._version is not None

    @property
    def version(self) -> tuple[int, int, int] | None:
        """Get installed FlashAttention version."""
        return self._version

    def _create_kernel_spec(self) -> KernelSpec:
        """Create KernelSpec for this adapter.

        Returns:
            KernelSpec describing FlashAttention capabilities.
        """
        # Determine version string
        if self._version:
            version_str = f"{self._version[0]}.{self._version[1]}.{self._version[2]}"
        else:
            version_str = "unknown"

        return KernelSpec(
            kernel_id="flash_attn.auto",
            operation="attention",
            source="flash_attn",
            version=version_str,
            impl=self,
            platform=Platform.CUDA,
            # Flash requires SM 8.0+
            min_sm=(8, 0),
            max_sm=None,  # No upper limit (FA4 for future)
            # Only fp16/bf16 supported
            supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
            # Head dim constraints
            min_head_dim=8,
            max_head_dim=256,
            head_dim_multiple=8,
            # Sequence constraints
            min_seq_len=1,
            max_seq_len=None,  # Flash handles variable seq
            # Feature support
            supports_gqa=True,
            supports_mqa=True,
            supports_attn_mask=False,  # Flash has different mask API
            supported_attn_mask_types=frozenset([MaskType.NONE]),
            supports_dropout=True,
            supports_scale=True,
            supports_alibi=True,
            # Layout support - Flash uses BSHD
            requires_last_dim_stride1=True,
            requires_contiguous=True,
            requires_layouts=frozenset([Layout.BSHD]),
            produces_layout=Layout.BSHD,
            # Execution properties
            is_cuda_graph_safe=True,
            deterministic=False,
            workspace_bytes=0,
            # High priority - Flash is preferred
            priority=90,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification.

        Returns:
            KernelSpec for this FlashAttention adapter.
        """
        return self._kernel_spec

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
        softmax_scale: float | None = None,
        dropout_p: float = 0.0,
        window_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        """Execute FlashAttention.

        Expected input layout: BSHD (batch, seqlen, nheads, headdim)

        Args:
            query: Query tensor (batch, seqlen_q, nheads, headdim).
            key: Key tensor (batch, seqlen_k, nheads_k, headdim).
            value: Value tensor (batch, seqlen_k, nheads_k, headdim).
            is_causal: Use causal attention masking.
            softmax_scale: Attention scale (default: 1/sqrt(headdim)).
            dropout_p: Dropout probability.
            window_size: Sliding window attention size.

        Returns:
            Attention output tensor (batch, seqlen_q, nheads, headdim).

        Raises:
            RuntimeError: If FlashAttention is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "FlashAttention is not installed. "
                "Install with: pip install flash-attn --no-build-isolation"
            )

        try:
            from flash_attn import flash_attn_func
        except ImportError as e:
            raise RuntimeError(f"Failed to import flash_attn: {e}") from e

        # Build kwargs
        kwargs = {
            "causal": is_causal,
            "dropout_p": dropout_p,
        }

        if softmax_scale is not None:
            kwargs["softmax_scale"] = softmax_scale

        if window_size is not None:
            kwargs["window_size"] = window_size

        # Call flash_attn_func
        return flash_attn_func(query, key, value, **kwargs)
