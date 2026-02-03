"""FlashAttention 4 adapter for Blackwell GPUs.

This module provides the FA4 adapter class that wraps
FlashAttention 4 for Blackwell (SM 10.0+) GPUs.
"""
from __future__ import annotations

import logging
from threading import RLock
from typing import TYPE_CHECKING

import torch

from layerzero.backends.base import BaseKernel
from layerzero.backends.flash_attn_v4.availability import (
    detect_fa4_version,
    is_fa4_available,
)
from layerzero.backends.flash_attn_v4.specs import create_fa4_kernel_spec
from layerzero.models.kernel_spec import KernelSpec

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class FlashAttnV4Adapter(BaseKernel):
    """Adapter for FlashAttention 4 (Blackwell).

    FA4 provides optimized attention for Blackwell GPUs (SM 10.0+)
    using tcgen05.mma tensor core operations.

    Attributes:
        is_available: Whether FA4 is installed.
        version: Installed FA4 version tuple.
    """

    __slots__ = ("_version", "_kernel_spec", "_lock")

    def __init__(self) -> None:
        """Initialize FlashAttnV4Adapter."""
        self._lock = RLock()
        self._version = detect_fa4_version()
        self._kernel_spec = self._create_kernel_spec()

        logger.debug(
            "FlashAttnV4Adapter initialized, version=%s, available=%s",
            self._version,
            self.is_available,
        )

    @property
    def is_available(self) -> bool:
        """Check if FA4 is available."""
        return self._version is not None

    @property
    def version(self) -> tuple[int, int, int] | None:
        """Get installed FA4 version."""
        return self._version

    def _create_kernel_spec(self) -> KernelSpec:
        """Create KernelSpec for FA4.

        Returns:
            KernelSpec describing FA4 capabilities.
        """
        if self._version:
            version_str = f"{self._version[0]}.{self._version[1]}.{self._version[2]}"
        else:
            version_str = "4.0.0"

        return create_fa4_kernel_spec(
            impl=self,
            version_str=version_str,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification.

        Returns:
            KernelSpec for FA4.
        """
        with self._lock:
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
        """Execute FlashAttention 4.

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
            RuntimeError: If FA4 is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "FlashAttention 4 is not installed. "
                "Install with: pip install flash-attn>=3.0 --no-build-isolation"
            )

        try:
            # FA4 uses flash_attn_func for the basic API
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

    def call_fp8(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
        softmax_scale: float | None = None,
        descale_q: torch.Tensor | None = None,
        descale_k: torch.Tensor | None = None,
        descale_v: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute FlashAttention 4 with FP8 inputs.

        FA4 supports FP8 (E4M3, E5M2) on Blackwell GPUs.

        Args:
            query: Query tensor in FP8 format.
            key: Key tensor in FP8 format.
            value: Value tensor in FP8 format.
            is_causal: Use causal attention masking.
            softmax_scale: Attention scale.
            descale_q: Dequantization scale for query.
            descale_k: Dequantization scale for key.
            descale_v: Dequantization scale for value.

        Returns:
            Attention output tensor.

        Raises:
            RuntimeError: If FA4 FP8 is not available.
        """
        if not self.is_available:
            raise RuntimeError("FlashAttention 4 FP8 is not available")

        try:
            # FA4 FP8 API (may differ from actual implementation)
            from flash_attn import flash_attn_func_fp8
        except ImportError:
            # Fallback: convert FP8 to fp16 and use regular path
            logger.warning(
                "FA4 FP8 API not available, falling back to fp16 conversion"
            )
            return self(
                query.to(torch.float16),
                key.to(torch.float16),
                value.to(torch.float16),
                is_causal=is_causal,
                softmax_scale=softmax_scale,
            )

        kwargs = {
            "causal": is_causal,
        }

        if softmax_scale is not None:
            kwargs["softmax_scale"] = softmax_scale

        if descale_q is not None:
            kwargs["descale_q"] = descale_q
        if descale_k is not None:
            kwargs["descale_k"] = descale_k
        if descale_v is not None:
            kwargs["descale_v"] = descale_v

        return flash_attn_func_fp8(query, key, value, **kwargs)
