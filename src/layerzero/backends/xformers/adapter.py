"""
LayerZero xFormers Adapter

Adapter class wrapping xFormers memory_efficient_attention for attention operations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.backends.base import BaseKernel
from layerzero.backends.xformers.version import (
    detect_xformers_version,
    is_xformers_available,
)
from layerzero.backends.xformers.constraints import (
    XFORMERS_MIN_SM,
    XFORMERS_MIN_HEAD_DIM,
    XFORMERS_MAX_HEAD_DIM,
    XFORMERS_HEAD_DIM_MULTIPLE,
)
from layerzero.backends.xformers.bias import expand_attn_bias
from layerzero.enums import Layout, MaskType, Platform
from layerzero.models.kernel_spec import KernelSpec

if TYPE_CHECKING:
    pass


class XFormersAdapter(BaseKernel):
    """Adapter for xFormers memory_efficient_attention.

    xFormers provides memory efficient attention implementations from
    Facebook Research. This adapter handles:
    - BSHD layout requirement
    - attn_bias validation and expansion
    - 5D GQA inputs (experimental)
    - LowerTriangularMask for causal attention

    Attributes:
        is_available: Whether xFormers is installed.
        version: Installed xFormers version tuple.
    """

    __slots__ = ("_version", "_kernel_spec")

    def __init__(self) -> None:
        """Initialize XFormersAdapter."""
        self._version = detect_xformers_version()
        self._kernel_spec = self._create_kernel_spec()

    @property
    def is_available(self) -> bool:
        """Check if xFormers is available."""
        return self._version is not None

    @property
    def version(self) -> tuple[int, int, int] | None:
        """Get installed xFormers version."""
        return self._version

    def _create_kernel_spec(self) -> KernelSpec:
        """Create KernelSpec for this adapter."""
        if self._version:
            version_str = f"{self._version[0]}.{self._version[1]}.{self._version[2]}"
        else:
            version_str = "unknown"

        return KernelSpec(
            kernel_id="xformers.memory_efficient",
            operation="attention.causal",
            source="xformers",
            version=version_str,
            impl=self,
            platform=Platform.CUDA,
            min_sm=XFORMERS_MIN_SM,
            max_sm=None,  # No upper limit
            supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
            min_head_dim=XFORMERS_MIN_HEAD_DIM,
            max_head_dim=XFORMERS_MAX_HEAD_DIM,
            head_dim_multiple=XFORMERS_HEAD_DIM_MULTIPLE,
            min_seq_len=1,
            max_seq_len=None,  # No hard limit
            supports_gqa=True,  # Via 5D inputs
            supports_mqa=True,  # Via 5D inputs
            supports_attn_mask=True,  # Via attn_bias
            supported_attn_mask_types=frozenset([MaskType.NONE, MaskType.FLOAT]),
            supports_dropout=True,
            supports_scale=True,
            supports_alibi=False,  # Not directly supported
            requires_last_dim_stride1=True,
            requires_contiguous=False,
            requires_layouts=frozenset([Layout.BSHD]),
            produces_layout=Layout.BSHD,
            is_cuda_graph_safe=True,
            deterministic=False,  # May vary between runs
            workspace_bytes=0,
            priority=70,  # Lower than FlashAttn (90) and FlashInfer (85)
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_bias: torch.Tensor | None = None,
        is_causal: bool = False,
        dropout_p: float = 0.0,
        softmax_scale: float | None = None,
        auto_expand_bias: bool = False,
        use_5d_gqa: bool = False,
    ) -> torch.Tensor:
        """Execute xFormers memory_efficient_attention.

        Expected input layout: BSHD (batch, seqlen, nheads, headdim).

        Args:
            query: Query tensor (batch, seqlen_q, nheads, headdim).
            key: Key tensor.
                 - Standard: (batch, seqlen_k, nheads_k, headdim).
                 - 5D GQA: (batch, seqlen_k, groups, nheads_kv, headdim).
            value: Value tensor (same shape as key).
            attn_bias: Additive attention bias (batch, heads, seq_q, seq_k).
                       Must be on same device as query.
                       Cannot use broadcast dimensions unless auto_expand_bias=True.
            is_causal: Use causal attention masking via LowerTriangularMask.
            dropout_p: Dropout probability (0.0 = no dropout).
            softmax_scale: Attention scale (default: 1/sqrt(headdim)).
            auto_expand_bias: If True, automatically expand attn_bias to
                              remove broadcast dimensions.
            use_5d_gqa: If True, K/V are in 5D format for GQA/MQA.

        Returns:
            Attention output tensor (batch, seqlen_q, nheads, headdim).

        Raises:
            RuntimeError: If xFormers is not available or CUDA not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "xFormers is not installed. "
                "Install with: pip install xformers"
            )

        if not query.is_cuda:
            raise RuntimeError("xFormers requires CUDA tensors")

        try:
            from xformers.ops import memory_efficient_attention, LowerTriangularMask
        except ImportError as e:
            raise RuntimeError(f"Failed to import xformers.ops: {e}") from e

        batch, seq_q, heads_q, dim = query.shape

        # Handle 5D GQA inputs
        if use_5d_gqa and key.ndim == 5:
            # K/V shape: (B, S, G, Hkv, D) -> pass as-is to xFormers
            # xFormers handles 5D internally for GQA
            pass
        elif use_5d_gqa and key.ndim == 4:
            # Reshape 4D to 5D if requested
            _, seq_k, heads_k, _ = key.shape
            groups = heads_q // heads_k
            key = key.view(batch, seq_k, groups, heads_k, dim)
            value = value.view(batch, seq_k, groups, heads_k, dim)

        # Build attn_bias argument
        op_attn_bias = None

        if is_causal and attn_bias is None:
            # Use efficient LowerTriangularMask for causal
            op_attn_bias = LowerTriangularMask()
        elif attn_bias is not None:
            # Validate and optionally expand bias
            if auto_expand_bias:
                heads = heads_q
                attn_bias = expand_attn_bias(attn_bias, batch, heads)
            op_attn_bias = attn_bias

        # Build kwargs
        kwargs: dict = {
            "attn_bias": op_attn_bias,
        }

        if dropout_p > 0.0:
            kwargs["p"] = dropout_p

        if softmax_scale is not None:
            kwargs["scale"] = softmax_scale

        # Call xFormers memory_efficient_attention
        output = memory_efficient_attention(query, key, value, **kwargs)

        return output
