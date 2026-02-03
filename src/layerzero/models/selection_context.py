"""
LayerZero Selection Context

Dataclass representing runtime context for kernel selection.
Built from input tensors and operation parameters.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, TYPE_CHECKING

from layerzero.enums import (
    KVCacheStrategy,
    Layout,
    MaskType,
    OpKind,
    QuantFormat,
)

if TYPE_CHECKING:
    import torch
    from layerzero.models.device_spec import DeviceSpec


# Map torch dtype names to strings for serialization
DTYPE_TO_STR = {
    "float16": "float16",
    "float32": "float32",
    "float64": "float64",
    "bfloat16": "bfloat16",
    "int8": "int8",
    "int16": "int16",
    "int32": "int32",
    "int64": "int64",
    "uint8": "uint8",
    "bool": "bool",
}


def _dtype_to_str(dtype: Any) -> str:
    """Convert torch dtype to string."""
    dtype_str = str(dtype).replace("torch.", "")
    return DTYPE_TO_STR.get(dtype_str, dtype_str)


def _str_to_dtype(dtype_str: str) -> Any:
    """Convert string to torch dtype."""
    import torch
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "float64": torch.float64,
        "bfloat16": torch.bfloat16,
        "int8": torch.int8,
        "int16": torch.int16,
        "int32": torch.int32,
        "int64": torch.int64,
        "uint8": torch.uint8,
        "bool": torch.bool,
    }
    return dtype_map.get(dtype_str, torch.float32)


@dataclass(frozen=True, slots=True)
class SelectionContext:
    """Runtime context for kernel selection.

    Contains all information needed to filter and score kernel candidates.
    Immutable (frozen) for hashability and thread safety.

    Attributes:
        device: Device specification
        op_kind: Operation kind (tensor, tokenization, etc.)
        operation: Operation identifier (e.g., "attention.causal")
        dtype: Data type of inputs
        batch_size: Batch size

        # Attention-specific
        seq_len_q: Query sequence length
        seq_len_k: Key/value sequence length
        num_heads: Number of attention heads
        num_kv_heads: Number of KV heads (for GQA)
        head_dim: Head dimension

        # Layout and contiguity
        layout: Tensor layout (BSHD, BHSD, etc.)
        stride_last_dim: Stride of last dimension
        is_contiguous: Whether tensor is contiguous

        # Attention parameters
        attn_mask_type: Type of attention mask
        is_causal: Whether attention is causal
        dropout_p: Dropout probability
        scale: Attention scale factor
        enable_gqa: Whether GQA is enabled

        # KV cache
        kv_cache_layout: KV cache layout
        kv_cache_dtype: KV cache dtype
        kv_strategy: KV cache strategy

        # Tokenizer context
        tokenizer_id: Tokenizer identifier
        vocab_hash: Vocabulary hash
        merges_hash: BPE merges hash
        added_tokens_hash: Added tokens hash
        normalizer_id: Normalizer identifier
        pretokenizer_id: Pre-tokenizer identifier
        special_tokens_hash: Special tokens hash
        return_offsets: Whether to return token offsets

        # Quantization
        quant_format: Quantization format
        packed_weights_id: Packed weights identifier

        # Runtime context
        is_cuda_graph_capturing: Whether CUDA graph is being captured
        requires_deterministic: Whether determinism is required

        # Distributed context
        tp_size: Tensor parallel size
        pp_size: Pipeline parallel size
        rank: Current rank
    """

    # Required fields
    device: "DeviceSpec"
    op_kind: OpKind
    operation: str
    dtype: "torch.dtype"
    batch_size: int

    # Attention-specific (optional)
    seq_len_q: int | None = None
    seq_len_k: int | None = None
    num_heads: int | None = None
    num_kv_heads: int | None = None
    head_dim: int | None = None

    # Layout
    layout: Layout = Layout.BSHD
    stride_last_dim: int = 1
    is_contiguous: bool = True

    # Attention mask
    attn_mask_type: MaskType = MaskType.NONE
    is_causal: bool = False

    # Optional parameters
    dropout_p: float = 0.0
    scale: float | None = None
    enable_gqa: bool = False

    # KV cache
    kv_cache_layout: Layout | None = None
    kv_cache_dtype: "torch.dtype | None" = None
    kv_strategy: KVCacheStrategy | None = None

    # Tokenizer context
    tokenizer_id: str | None = None
    vocab_hash: str | None = None
    merges_hash: str | None = None
    added_tokens_hash: str | None = None
    normalizer_id: str | None = None
    pretokenizer_id: str | None = None
    special_tokens_hash: str | None = None
    return_offsets: bool = False

    # Quantization
    quant_format: QuantFormat | None = None
    packed_weights_id: str | None = None

    # Runtime context
    is_cuda_graph_capturing: bool = False
    requires_deterministic: bool = False

    # Distributed context
    tp_size: int = 1
    pp_size: int = 1
    rank: int = 0

    def cache_key(self) -> str:
        """Generate cache key for selection caching.

        Returns:
            String cache key based on context properties.
        """
        key_parts = [
            self.operation,
            _dtype_to_str(self.dtype),
            str(self.batch_size),
            str(self.seq_len_q),
            str(self.seq_len_k),
            str(self.num_heads),
            str(self.num_kv_heads),
            str(self.head_dim),
            self.layout.value,
            str(self.is_causal),
            str(self.is_cuda_graph_capturing),
            self.device.cache_key(),
        ]
        key_str = ":".join(key_parts)
        return hashlib.md5(key_str.encode()).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON compatibility.

        Returns:
            Dict with context fields.
        """
        return {
            "device": self.device.to_dict(),
            "op_kind": self.op_kind.value,
            "operation": self.operation,
            "dtype": _dtype_to_str(self.dtype),
            "batch_size": self.batch_size,
            "seq_len_q": self.seq_len_q,
            "seq_len_k": self.seq_len_k,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "head_dim": self.head_dim,
            "layout": self.layout.value,
            "stride_last_dim": self.stride_last_dim,
            "is_contiguous": self.is_contiguous,
            "attn_mask_type": self.attn_mask_type.value,
            "is_causal": self.is_causal,
            "dropout_p": self.dropout_p,
            "scale": self.scale,
            "enable_gqa": self.enable_gqa,
            "kv_cache_layout": self.kv_cache_layout.value if self.kv_cache_layout else None,
            "kv_cache_dtype": _dtype_to_str(self.kv_cache_dtype) if self.kv_cache_dtype else None,
            "kv_strategy": self.kv_strategy.value if self.kv_strategy else None,
            "tokenizer_id": self.tokenizer_id,
            "vocab_hash": self.vocab_hash,
            "merges_hash": self.merges_hash,
            "added_tokens_hash": self.added_tokens_hash,
            "normalizer_id": self.normalizer_id,
            "pretokenizer_id": self.pretokenizer_id,
            "special_tokens_hash": self.special_tokens_hash,
            "return_offsets": self.return_offsets,
            "quant_format": self.quant_format.value if self.quant_format else None,
            "packed_weights_id": self.packed_weights_id,
            "is_cuda_graph_capturing": self.is_cuda_graph_capturing,
            "requires_deterministic": self.requires_deterministic,
            "tp_size": self.tp_size,
            "pp_size": self.pp_size,
            "rank": self.rank,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SelectionContext":
        """Deserialize from dictionary.

        Args:
            d: Dict with context fields.

        Returns:
            New SelectionContext instance.
        """
        from layerzero.models.device_spec import DeviceSpec

        return cls(
            device=DeviceSpec.from_dict(d["device"]),
            op_kind=OpKind(d["op_kind"]),
            operation=d["operation"],
            dtype=_str_to_dtype(d["dtype"]),
            batch_size=d["batch_size"],
            seq_len_q=d.get("seq_len_q"),
            seq_len_k=d.get("seq_len_k"),
            num_heads=d.get("num_heads"),
            num_kv_heads=d.get("num_kv_heads"),
            head_dim=d.get("head_dim"),
            layout=Layout(d.get("layout", "BSHD")),
            stride_last_dim=d.get("stride_last_dim", 1),
            is_contiguous=d.get("is_contiguous", True),
            attn_mask_type=MaskType(d.get("attn_mask_type", "none")),
            is_causal=d.get("is_causal", False),
            dropout_p=d.get("dropout_p", 0.0),
            scale=d.get("scale"),
            enable_gqa=d.get("enable_gqa", False),
            kv_cache_layout=Layout(d["kv_cache_layout"]) if d.get("kv_cache_layout") else None,
            kv_cache_dtype=_str_to_dtype(d["kv_cache_dtype"]) if d.get("kv_cache_dtype") else None,
            kv_strategy=KVCacheStrategy(d["kv_strategy"]) if d.get("kv_strategy") else None,
            tokenizer_id=d.get("tokenizer_id"),
            vocab_hash=d.get("vocab_hash"),
            merges_hash=d.get("merges_hash"),
            added_tokens_hash=d.get("added_tokens_hash"),
            normalizer_id=d.get("normalizer_id"),
            pretokenizer_id=d.get("pretokenizer_id"),
            special_tokens_hash=d.get("special_tokens_hash"),
            return_offsets=d.get("return_offsets", False),
            quant_format=QuantFormat(d["quant_format"]) if d.get("quant_format") else None,
            packed_weights_id=d.get("packed_weights_id"),
            is_cuda_graph_capturing=d.get("is_cuda_graph_capturing", False),
            requires_deterministic=d.get("requires_deterministic", False),
            tp_size=d.get("tp_size", 1),
            pp_size=d.get("pp_size", 1),
            rank=d.get("rank", 0),
        )

    @classmethod
    def from_tensors(
        cls,
        q: "torch.Tensor",
        k: "torch.Tensor | None" = None,
        v: "torch.Tensor | None" = None,
        *,
        is_causal: bool = False,
        dropout_p: float = 0.0,
        scale: float | None = None,
        attn_mask: "torch.Tensor | None" = None,
        layout: Layout = Layout.BSHD,
        device: "DeviceSpec | None" = None,
        **kwargs: Any,
    ) -> "SelectionContext":
        """Build context from attention tensors.

        Args:
            q: Query tensor
            k: Key tensor (optional, defaults to q shape)
            v: Value tensor (optional, defaults to k shape)
            is_causal: Whether attention is causal
            dropout_p: Dropout probability
            scale: Attention scale factor
            attn_mask: Attention mask tensor
            layout: Tensor layout
            device: Device specification (auto-detected if None)
            **kwargs: Additional context fields

        Returns:
            SelectionContext for the attention operation.
        """
        import torch
        from layerzero.models.device_spec import DeviceSpec

        # Use k=q, v=k defaults if not provided
        if k is None:
            k = q
        if v is None:
            v = k

        # Detect device if not provided
        if device is None:
            device = DeviceSpec.detect(str(q.device))

        # Extract shape based on layout
        if layout == Layout.BSHD:
            # (batch, seq, heads, dim)
            batch_size = q.shape[0]
            seq_len_q = q.shape[1]
            num_heads = q.shape[2]
            head_dim = q.shape[3]
            seq_len_k = k.shape[1]
            num_kv_heads = k.shape[2]
        elif layout == Layout.BHSD:
            # (batch, heads, seq, dim)
            batch_size = q.shape[0]
            num_heads = q.shape[1]
            seq_len_q = q.shape[2]
            head_dim = q.shape[3]
            seq_len_k = k.shape[2]
            num_kv_heads = k.shape[1]
        else:
            # NHD, HND - flattened batch
            batch_size = 1
            seq_len_q = q.shape[0]
            num_heads = q.shape[1]
            head_dim = q.shape[2]
            seq_len_k = k.shape[0]
            num_kv_heads = k.shape[1]

        # Detect GQA
        enable_gqa = num_heads != num_kv_heads

        # Detect contiguity
        is_contiguous = q.is_contiguous()
        stride_last_dim = q.stride(-1) if q.ndim > 0 else 1

        # Detect mask type
        attn_mask_type = MaskType.NONE
        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                attn_mask_type = MaskType.BOOL
            else:
                attn_mask_type = MaskType.FLOAT

        return cls(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal" if is_causal else "attention.full",
            dtype=q.dtype,
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_k=seq_len_k,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            layout=layout,
            stride_last_dim=stride_last_dim,
            is_contiguous=is_contiguous,
            attn_mask_type=attn_mask_type,
            is_causal=is_causal,
            dropout_p=dropout_p,
            scale=scale,
            enable_gqa=enable_gqa,
            **kwargs,
        )

    @classmethod
    def for_norm(
        cls,
        x: "torch.Tensor",
        *,
        operation: str = "norm.rms",
        device: "DeviceSpec | None" = None,
        **kwargs: Any,
    ) -> "SelectionContext":
        """Build context for normalization operations.

        Args:
            x: Input tensor
            operation: Operation identifier
            device: Device specification (auto-detected if None)
            **kwargs: Additional context fields

        Returns:
            SelectionContext for the normalization operation.
        """
        from layerzero.models.device_spec import DeviceSpec

        # Detect device if not provided
        if device is None:
            device = DeviceSpec.detect(str(x.device))

        batch_size = x.shape[0] if x.ndim > 0 else 1

        return cls(
            device=device,
            op_kind=OpKind.TENSOR,
            operation=operation,
            dtype=x.dtype,
            batch_size=batch_size,
            is_contiguous=x.is_contiguous(),
            stride_last_dim=x.stride(-1) if x.ndim > 0 else 1,
            **kwargs,
        )
