"""
LayerZero Torch SDPA Adapter

Adapter class wrapping torch.nn.functional.scaled_dot_product_attention.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.backends.base import BaseKernel
from layerzero.backends.torch_sdpa.kernel import sdpa_forward, SDPAConfig
from layerzero.enums import Layout, MaskType, Platform
from layerzero.models.kernel_spec import KernelSpec

if TYPE_CHECKING:
    pass


class TorchSDPAAdapter(BaseKernel):
    """Adapter for torch.nn.functional.scaled_dot_product_attention.

    Provides a LayerZero-compatible interface to PyTorch's built-in
    scaled dot product attention implementation.

    Supports multiple backends:
    - FlashAttention (SM 8.0+, fp16/bf16)
    - Memory Efficient Attention (SM 5.0+, any dtype)
    - cuDNN Attention (SM 8.0+, fp16/bf16, head_dim <= 128)
    - Math (reference implementation, always works)

    Attributes:
        backend_hint: Optional hint for backend selection.
    """

    __slots__ = ("_backend_hint", "_kernel_spec")

    def __init__(self, backend_hint: str | None = None) -> None:
        """Initialize TorchSDPAAdapter.

        Args:
            backend_hint: Optional backend hint ("flash", "efficient", "cudnn", "math").
                         None lets PyTorch auto-select.
        """
        self._backend_hint = backend_hint
        self._kernel_spec = self._create_kernel_spec()

    @property
    def backend_hint(self) -> str | None:
        """Get backend hint."""
        return self._backend_hint

    def _create_kernel_spec(self) -> KernelSpec:
        """Create KernelSpec for this adapter.

        Returns:
            KernelSpec describing SDPA capabilities.
        """
        # Determine kernel_id based on backend hint
        if self._backend_hint:
            kernel_id = f"torch.sdpa.{self._backend_hint}"
        else:
            kernel_id = "torch.sdpa.auto"

        return KernelSpec(
            kernel_id=kernel_id,
            operation="attention",
            source="torch",
            version=torch.__version__,
            impl=self,
            platform=Platform.CUDA,
            # SM 5.0+ for efficient backend (most compatible)
            min_sm=(5, 0),
            # No upper limit
            max_sm=None,
            # Support common float dtypes
            supported_dtypes=frozenset([
                torch.float16,
                torch.bfloat16,
                torch.float32,
            ]),
            # Head dimension constraints (conservative)
            min_head_dim=1,
            max_head_dim=256,
            head_dim_multiple=1,
            # Sequence constraints
            min_seq_len=1,
            max_seq_len=None,  # No hard limit
            # Feature support
            supports_gqa=True,
            supports_mqa=True,
            supports_attn_mask=True,
            supported_attn_mask_types=frozenset([
                MaskType.NONE,
                MaskType.BOOL,
                MaskType.FLOAT,
            ]),
            supports_dropout=True,
            supports_scale=True,
            supports_alibi=False,  # SDPA doesn't support ALiBi directly
            # Layout support
            requires_last_dim_stride1=False,  # Math backend handles any stride
            requires_contiguous=False,  # Math backend handles non-contiguous
            requires_layouts=frozenset([Layout.BHSD]),  # PyTorch SDPA uses (B, H, S, D)
            produces_layout=Layout.BHSD,
            # Execution properties
            is_cuda_graph_safe=True,
            deterministic=False,  # Flash/efficient are non-deterministic
            workspace_bytes=0,
            # Priority (high for native PyTorch)
            priority=70,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification.

        Returns:
            KernelSpec for this SDPA adapter.
        """
        return self._kernel_spec

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        enable_gqa: bool = False,
        training: bool = False,
    ) -> torch.Tensor:
        """Execute scaled dot product attention.

        Args:
            query: Query tensor (batch, heads, seq_q, head_dim).
            key: Key tensor (batch, heads_kv, seq_k, head_dim).
            value: Value tensor (batch, heads_kv, seq_k, head_dim).
            attn_mask: Optional attention mask.
            dropout_p: Dropout probability.
            is_causal: Use causal attention masking.
            scale: Attention scale (default: 1/sqrt(head_dim)).
            enable_gqa: Enable grouped query attention.
            training: Whether in training mode.

        Returns:
            Attention output tensor (batch, heads, seq_q, head_dim).
        """
        return sdpa_forward(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=scale,
            enable_gqa=enable_gqa,
            training=training,
            backend_hint=self._backend_hint,
        )
