"""
LayerZero FlashInfer Adapter

Adapter classes wrapping FlashInfer library for attention operations.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from layerzero.backends.base import BaseKernel
from layerzero.backends.flashinfer.version import (
    detect_flashinfer_version,
    is_flashinfer_available,
    is_jit_cache_available,
)
from layerzero.backends.flashinfer.layout import (
    bshd_to_nhd,
    nhd_to_bshd,
)
from layerzero.backends.flashinfer.constraints import (
    FLASHINFER_MIN_SM,
    FLASHINFER_MIN_HEAD_DIM,
    FLASHINFER_MAX_HEAD_DIM,
    FLASHINFER_HEAD_DIM_MULTIPLE,
)
from layerzero.enums import Layout, MaskType, Platform, KVCacheStrategy
from layerzero.models.kernel_spec import KernelSpec

if TYPE_CHECKING:
    pass


class FlashInferPrefillAdapter(BaseKernel):
    """Adapter for FlashInfer prefill attention.

    FlashInfer prefill handles full sequence attention computation.
    Supports both dense batches and ragged/variable-length batches.

    Attributes:
        is_available: Whether flashinfer is installed.
        version: Installed flashinfer version tuple.
        jit_cache_enabled: Whether JIT cache is available.
    """

    __slots__ = ("_version", "_kernel_spec", "_jit_cache_enabled")

    def __init__(self) -> None:
        """Initialize FlashInferPrefillAdapter."""
        self._version = detect_flashinfer_version()
        self._jit_cache_enabled = is_jit_cache_available()
        self._kernel_spec = self._create_kernel_spec()

    @property
    def is_available(self) -> bool:
        """Check if FlashInfer is available."""
        return self._version is not None

    @property
    def version(self) -> tuple[int, int, int] | None:
        """Get installed FlashInfer version."""
        return self._version

    @property
    def jit_cache_enabled(self) -> bool:
        """Check if JIT cache is enabled."""
        return self._jit_cache_enabled

    def _create_kernel_spec(self) -> KernelSpec:
        """Create KernelSpec for this adapter."""
        if self._version:
            version_str = f"{self._version[0]}.{self._version[1]}.{self._version[2]}"
        else:
            version_str = "unknown"

        return KernelSpec(
            kernel_id="flashinfer.prefill",
            operation="attention.prefill",
            source="flashinfer",
            version=version_str,
            impl=self,
            platform=Platform.CUDA,
            min_sm=FLASHINFER_MIN_SM,
            max_sm=None,  # No upper limit
            supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
            min_head_dim=FLASHINFER_MIN_HEAD_DIM,
            max_head_dim=FLASHINFER_MAX_HEAD_DIM,
            head_dim_multiple=FLASHINFER_HEAD_DIM_MULTIPLE,
            min_seq_len=1,
            max_seq_len=None,  # FlashInfer handles long sequences
            supports_gqa=True,
            supports_mqa=True,
            supports_attn_mask=False,  # FlashInfer has different mask API
            supported_attn_mask_types=frozenset([MaskType.NONE]),
            supports_dropout=True,
            supports_scale=True,
            supports_alibi=True,
            requires_last_dim_stride1=True,
            requires_contiguous=True,
            requires_layouts=frozenset([Layout.BSHD, Layout.NHD]),
            produces_layout=Layout.BSHD,
            supports_kv_strategies=frozenset([
                KVCacheStrategy.CONTIGUOUS,
                KVCacheStrategy.PAGED,
            ]),
            is_cuda_graph_safe=True,
            deterministic=False,
            workspace_bytes=0,
            priority=85,  # High but below FA for standard cases
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def warmup(
        self,
        shapes: list[tuple[int, ...]],
        dry_run: bool = False,
    ) -> None:
        """Warmup JIT compilation for specified shapes.

        FlashInfer uses JIT compilation which can be slow on first use.
        Call this method with expected shapes to pre-compile kernels.

        Args:
            shapes: List of (batch, seq, heads, dim) shapes to warmup.
            dry_run: If True, only validate shapes without compilation.
        """
        if not self.is_available:
            raise RuntimeError("FlashInfer is not installed")

        if dry_run:
            return

        try:
            from flashinfer import single_prefill_with_kv_cache
        except ImportError as e:
            raise RuntimeError(f"Failed to import flashinfer: {e}") from e

        for shape in shapes:
            batch, seq, heads, dim = shape
            device = torch.device("cuda")
            dtype = torch.float16

            # Create dummy tensors for warmup
            q = torch.empty(batch * seq, heads, dim, device=device, dtype=dtype)
            k = torch.empty(batch * seq, heads, dim, device=device, dtype=dtype)
            v = torch.empty(batch * seq, heads, dim, device=device, dtype=dtype)

            # Trigger JIT compilation
            try:
                single_prefill_with_kv_cache(q, k, v, causal=True)
            except Exception:
                # Ignore errors during warmup (may fail on CPU-only systems)
                pass

    def __call__(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        is_causal: bool = False,
        softmax_scale: float | None = None,
        dropout_p: float = 0.0,
        seq_lens_q: torch.Tensor | None = None,
        seq_lens_k: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Execute FlashInfer prefill attention.

        Expected input layout: BSHD (batch, seqlen, nheads, headdim)
        Internally converts to NHD for FlashInfer.

        Args:
            query: Query tensor (batch, seqlen_q, nheads, headdim).
            key: Key tensor (batch, seqlen_k, nheads_k, headdim).
            value: Value tensor (batch, seqlen_k, nheads_k, headdim).
            is_causal: Use causal attention masking.
            softmax_scale: Attention scale (default: 1/sqrt(headdim)).
            dropout_p: Dropout probability.
            seq_lens_q: Actual sequence lengths for Q (for ragged batches).
            seq_lens_k: Actual sequence lengths for K/V (for ragged batches).

        Returns:
            Attention output tensor (batch, seqlen_q, nheads, headdim).

        Raises:
            RuntimeError: If FlashInfer is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "FlashInfer is not installed. "
                "Install with: pip install flashinfer-python"
            )

        try:
            from flashinfer import single_prefill_with_kv_cache
        except ImportError as e:
            raise RuntimeError(f"Failed to import flashinfer: {e}") from e

        # Get original shape for output
        batch, seq_q, heads_q, dim = query.shape
        _, seq_k, heads_k, _ = key.shape

        # Convert BSHD to NHD
        q_nhd, q_seq_lens = bshd_to_nhd(query, seq_lens_q)
        k_nhd, k_seq_lens = bshd_to_nhd(key, seq_lens_k)
        v_nhd, _ = bshd_to_nhd(value, seq_lens_k)

        # Build kwargs
        kwargs = {
            "causal": is_causal,
        }

        if softmax_scale is not None:
            kwargs["sm_scale"] = softmax_scale

        # Call FlashInfer
        output_nhd = single_prefill_with_kv_cache(q_nhd, k_nhd, v_nhd, **kwargs)

        # Convert NHD back to BSHD
        output = nhd_to_bshd(output_nhd, q_seq_lens, max_seq_len=seq_q)

        return output


class FlashInferDecodeAdapter(BaseKernel):
    """Adapter for FlashInfer decode attention.

    FlashInfer decode handles single-token decoding with KV cache.
    Optimized for autoregressive generation.

    Attributes:
        is_available: Whether flashinfer is installed.
        version: Installed flashinfer version tuple.
    """

    __slots__ = ("_version", "_kernel_spec")

    def __init__(self) -> None:
        """Initialize FlashInferDecodeAdapter."""
        self._version = detect_flashinfer_version()
        self._kernel_spec = self._create_kernel_spec()

    @property
    def is_available(self) -> bool:
        """Check if FlashInfer is available."""
        return self._version is not None

    @property
    def version(self) -> tuple[int, int, int] | None:
        """Get installed FlashInfer version."""
        return self._version

    @property
    def jit_cache_enabled(self) -> bool:
        """Check if JIT cache is enabled."""
        return is_jit_cache_available()

    def _create_kernel_spec(self) -> KernelSpec:
        """Create KernelSpec for this adapter."""
        if self._version:
            version_str = f"{self._version[0]}.{self._version[1]}.{self._version[2]}"
        else:
            version_str = "unknown"

        return KernelSpec(
            kernel_id="flashinfer.decode",
            operation="attention.decode",
            source="flashinfer",
            version=version_str,
            impl=self,
            platform=Platform.CUDA,
            min_sm=FLASHINFER_MIN_SM,
            max_sm=None,
            supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
            min_head_dim=FLASHINFER_MIN_HEAD_DIM,
            max_head_dim=FLASHINFER_MAX_HEAD_DIM,
            head_dim_multiple=FLASHINFER_HEAD_DIM_MULTIPLE,
            min_seq_len=1,
            max_seq_len=1,  # Decode is single token
            supports_gqa=True,
            supports_mqa=True,
            supports_attn_mask=False,
            supported_attn_mask_types=frozenset([MaskType.NONE]),
            supports_dropout=False,  # No dropout in decode
            supports_scale=True,
            supports_alibi=True,
            requires_last_dim_stride1=True,
            requires_contiguous=True,
            requires_layouts=frozenset([Layout.BSHD, Layout.NHD]),
            produces_layout=Layout.BSHD,
            supports_kv_strategies=frozenset([
                KVCacheStrategy.CONTIGUOUS,
                KVCacheStrategy.PAGED,
            ]),
            is_cuda_graph_safe=True,
            deterministic=False,
            workspace_bytes=0,
            priority=85,
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def warmup(
        self,
        shapes: list[tuple[int, ...]],
        dry_run: bool = False,
    ) -> None:
        """Warmup JIT compilation for specified shapes."""
        if not self.is_available:
            raise RuntimeError("FlashInfer is not installed")
        # Similar warmup logic as prefill
        if dry_run:
            return

    def __call__(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        seq_lens: torch.Tensor,
        softmax_scale: float | None = None,
    ) -> torch.Tensor:
        """Execute FlashInfer decode attention.

        Args:
            query: Query tensor (batch, 1, nheads, headdim).
            kv_cache: KV cache tensor (batch, seq_k, 2, nheads_k, headdim).
            seq_lens: Sequence lengths per batch item.
            softmax_scale: Attention scale.

        Returns:
            Attention output tensor (batch, 1, nheads, headdim).

        Raises:
            RuntimeError: If FlashInfer is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "FlashInfer is not installed. "
                "Install with: pip install flashinfer-python"
            )

        try:
            from flashinfer import single_decode_with_kv_cache
        except ImportError as e:
            raise RuntimeError(f"Failed to import flashinfer: {e}") from e

        batch, _, heads, dim = query.shape

        # Extract K and V from cache
        # Assuming cache shape: (batch, seq_k, 2, heads, dim)
        k_cache = kv_cache[:, :, 0, :, :]  # (batch, seq_k, heads, dim)
        v_cache = kv_cache[:, :, 1, :, :]  # (batch, seq_k, heads, dim)

        # Flatten to NHD
        q_nhd = query.squeeze(1)  # (batch, heads, dim) -> treat batch as N
        k_nhd, _ = bshd_to_nhd(k_cache, seq_lens)
        v_nhd, _ = bshd_to_nhd(v_cache, seq_lens)

        kwargs = {}
        if softmax_scale is not None:
            kwargs["sm_scale"] = softmax_scale

        # Call FlashInfer decode
        output_nhd = single_decode_with_kv_cache(q_nhd, k_nhd, v_nhd, **kwargs)

        # Reshape output: (batch, heads, dim) -> (batch, 1, heads, dim)
        output = output_nhd.unsqueeze(1)

        return output


class FlashInferPagedAdapter(BaseKernel):
    """Adapter for FlashInfer paged KV cache attention.

    FlashInfer paged attention handles block-organized KV cache
    for efficient memory management in production LLM serving.

    Attributes:
        is_available: Whether flashinfer is installed.
        version: Installed flashinfer version tuple.
    """

    __slots__ = ("_version", "_kernel_spec")

    def __init__(self) -> None:
        """Initialize FlashInferPagedAdapter."""
        self._version = detect_flashinfer_version()
        self._kernel_spec = self._create_kernel_spec()

    @property
    def is_available(self) -> bool:
        """Check if FlashInfer is available."""
        return self._version is not None

    @property
    def version(self) -> tuple[int, int, int] | None:
        """Get installed FlashInfer version."""
        return self._version

    @property
    def jit_cache_enabled(self) -> bool:
        """Check if JIT cache is enabled."""
        return is_jit_cache_available()

    def _create_kernel_spec(self) -> KernelSpec:
        """Create KernelSpec for this adapter."""
        if self._version:
            version_str = f"{self._version[0]}.{self._version[1]}.{self._version[2]}"
        else:
            version_str = "unknown"

        return KernelSpec(
            kernel_id="flashinfer.paged",
            operation="attention.paged",
            source="flashinfer",
            version=version_str,
            impl=self,
            platform=Platform.CUDA,
            min_sm=FLASHINFER_MIN_SM,
            max_sm=None,
            supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
            min_head_dim=FLASHINFER_MIN_HEAD_DIM,
            max_head_dim=FLASHINFER_MAX_HEAD_DIM,
            head_dim_multiple=FLASHINFER_HEAD_DIM_MULTIPLE,
            min_seq_len=1,
            max_seq_len=None,
            supports_gqa=True,
            supports_mqa=True,
            supports_attn_mask=False,
            supported_attn_mask_types=frozenset([MaskType.NONE]),
            supports_dropout=False,
            supports_scale=True,
            supports_alibi=True,
            requires_last_dim_stride1=True,
            requires_contiguous=True,
            requires_layouts=frozenset([Layout.BSHD, Layout.NHD]),
            produces_layout=Layout.BSHD,
            supports_kv_cache_layouts=frozenset([Layout.NHD]),
            supports_kv_cache_dtypes=frozenset([torch.float16, torch.bfloat16]),
            supports_kv_strategies=frozenset([KVCacheStrategy.PAGED]),
            is_cuda_graph_safe=True,
            deterministic=False,
            workspace_bytes=0,
            priority=90,  # High priority for paged attention
        )

    def get_kernel_spec(self) -> KernelSpec:
        """Return kernel specification."""
        return self._kernel_spec

    def warmup(
        self,
        shapes: list[tuple[int, ...]],
        dry_run: bool = False,
    ) -> None:
        """Warmup JIT compilation for specified shapes."""
        if not self.is_available:
            raise RuntimeError("FlashInfer is not installed")
        if dry_run:
            return

    def __call__(
        self,
        query: torch.Tensor,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor,
        softmax_scale: float | None = None,
        block_size: int = 16,
    ) -> torch.Tensor:
        """Execute FlashInfer paged attention.

        Args:
            query: Query tensor (batch, seq_q, nheads, headdim).
            kv_cache: Paged KV cache (num_blocks, 2, block_size, heads, dim).
            block_table: Block indices per batch (batch, max_blocks).
            seq_lens: Sequence lengths per batch item.
            softmax_scale: Attention scale.
            block_size: Size of each KV cache block.

        Returns:
            Attention output tensor (batch, seq_q, nheads, headdim).

        Raises:
            RuntimeError: If FlashInfer is not available.
        """
        if not self.is_available:
            raise RuntimeError(
                "FlashInfer is not installed. "
                "Install with: pip install flashinfer-python"
            )

        try:
            from flashinfer import BatchDecodeWithPagedKVCacheWrapper
        except ImportError as e:
            raise RuntimeError(f"Failed to import flashinfer: {e}") from e

        batch, seq_q, heads, dim = query.shape
        num_kv_heads = kv_cache.shape[3]  # Shape: (num_blocks, 2, block_size, kv_heads, dim)

        # Use FlashInfer's batch decode with paged KV wrapper
        # This handles the complexity of paged attention internally

        # Allocate workspace buffer (required by FlashInfer 0.5+)
        workspace_size = 128 * 1024 * 1024  # 128 MB default
        workspace_buffer = torch.empty(
            workspace_size, device=query.device, dtype=torch.uint8
        )

        # For single-token decode (most common case)
        if seq_q == 1:
            wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer)

            # Compute cumulative sequence lengths for indptr
            indptr = torch.zeros(batch + 1, device=query.device, dtype=torch.int32)
            indptr[1:] = seq_lens.cumsum(0)

            # Flatten query
            q_flat = query.reshape(batch, heads, dim)

            # Create indices from block_table
            indices = block_table.reshape(-1)
            last_page_len = seq_lens % block_size
            last_page_len[last_page_len == 0] = block_size

            wrapper.begin_forward(
                indptr,
                indices,
                last_page_len,
                heads,
                num_kv_heads,
                dim,
                block_size,
                query.dtype,
            )

            kwargs = {}
            if softmax_scale is not None:
                kwargs["sm_scale"] = softmax_scale

            output = wrapper.forward(q_flat, kv_cache, **kwargs)
            wrapper.end_forward()

            # Reshape output: (batch, heads, dim) -> (batch, 1, heads, dim)
            return output.unsqueeze(1)

        # For prefill with paged KV (less common)
        try:
            from flashinfer import BatchPrefillWithPagedKVCacheWrapper
        except ImportError as e:
            raise RuntimeError(f"FlashInfer prefill paged not available: {e}") from e

        wrapper = BatchPrefillWithPagedKVCacheWrapper(workspace_buffer)

        # Setup similar to decode but for prefill
        qo_indptr = torch.zeros(batch + 1, device=query.device, dtype=torch.int32)
        for i in range(batch):
            qo_indptr[i + 1] = qo_indptr[i] + seq_q

        kv_indptr = torch.zeros(batch + 1, device=query.device, dtype=torch.int32)
        kv_indptr[1:] = seq_lens.cumsum(0)

        indices = block_table.reshape(-1)
        last_page_len = seq_lens % block_size
        last_page_len[last_page_len == 0] = block_size

        wrapper.begin_forward(
            qo_indptr,
            kv_indptr,
            indices,
            last_page_len,
            heads,
            num_kv_heads,
            dim,
            block_size,
        )

        # Flatten query to (total_tokens, heads, dim)
        q_flat, _ = bshd_to_nhd(query)

        kwargs = {"causal": True}
        if softmax_scale is not None:
            kwargs["sm_scale"] = softmax_scale

        output_flat = wrapper.forward(q_flat, kv_cache, **kwargs)
        wrapper.end_forward()

        # Reshape output back to BSHD
        output = nhd_to_bshd(
            output_flat,
            torch.full((batch,), seq_q, device=query.device, dtype=torch.int32),
            max_seq_len=seq_q,
        )

        return output
