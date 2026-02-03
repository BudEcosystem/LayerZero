"""
LayerZero Kernel Specification

Dataclass describing a kernel's capabilities and constraints.
Used by selection engine to filter and score candidates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from layerzero.device import GPUGeneration
from layerzero.enums import (
    KVCacheStrategy,
    Layout,
    MaskType,
    Platform,
    QuantFormat,
)
from layerzero.reasons import (
    CUDA_GRAPH_UNSAFE,
    DTYPE_UNSUPPORTED,
    GQA_UNSUPPORTED,
    HEAD_DIM_ALIGNMENT,
    HEAD_DIM_TOO_LARGE,
    HEAD_DIM_TOO_SMALL,
    LAYOUT_UNSUPPORTED,
    NON_DETERMINISTIC,
    NOT_CONTIGUOUS,
    PLATFORM_MISMATCH,
    SEQ_TOO_LONG,
    SEQ_TOO_SHORT,
    SM_TOO_NEW,
    SM_TOO_OLD,
    STRIDE_LAST_DIM,
    Reason,
    ReasonCategory,
    make_reason,
)

if TYPE_CHECKING:
    import torch
    from layerzero.models.selection_context import SelectionContext


def _dtype_to_str(dtype: Any) -> str:
    """Convert torch dtype to string."""
    return str(dtype).replace("torch.", "")


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
    return dtype_map.get(dtype_str)


@dataclass(frozen=True, slots=True)
class KernelSpec:
    """Kernel specification.

    Describes a kernel's capabilities, constraints, and implementation.
    Immutable (frozen) for hashability and thread safety.

    Attributes:
        kernel_id: Unique kernel identifier (e.g., "flash_attn.v3.causal")
        operation: Operation this kernel implements (e.g., "attention.causal")
        source: Source library (e.g., "flash_attn")
        version: Kernel version string

        impl: Implementation callable (not serialized)

        platform: Hardware platform requirement
        min_sm: Minimum SM version (major, minor)
        max_sm: Maximum SM version (major, minor)
        supported_generations: Supported GPU generations
        min_tensor_core_gen: Minimum tensor core generation

        supported_dtypes: Supported input dtypes
        requires_dtype: Required input dtype
        produces_dtype: Output dtype (if different from input)

        min_head_dim: Minimum head dimension
        max_head_dim: Maximum head dimension
        head_dim_multiple: Head dimension must be multiple of this
        min_seq_len: Minimum sequence length
        max_seq_len: Maximum sequence length
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size

        supports_gqa: Whether GQA is supported
        supports_mqa: Whether MQA is supported
        supports_attn_mask: Whether attention mask is supported
        supported_attn_mask_types: Supported attention mask types
        supports_dropout: Whether dropout is supported
        supports_scale: Whether scale parameter is supported
        supports_alibi: Whether ALiBi is supported

        requires_last_dim_stride1: Whether last dim must have stride 1
        requires_contiguous: Whether tensor must be contiguous
        requires_layouts: Supported input layouts
        produces_layout: Output layout (if different from input)

        supports_kv_cache_layouts: Supported KV cache layouts
        supports_kv_cache_dtypes: Supported KV cache dtypes
        supports_kv_strategies: Supported KV cache strategies

        supports_quant_formats: Supported quantization formats
        requires_packed_weights: Whether packed weights are required
        supports_prepack: Whether prepacking is supported

        is_cuda_graph_safe: Whether kernel is CUDA graph safe
        deterministic: Whether kernel is deterministic
        workspace_bytes: Workspace memory requirement

        priority: Selection priority (0-100, higher = preferred)

        fuses_ops: List of fused operations
        transform_cost_hint: Cost hint for layout/dtype transforms
    """

    # Identity
    kernel_id: str
    operation: str
    source: str
    version: str

    # Implementation (not serialized)
    impl: Callable | None = None

    # Hardware requirements
    platform: Platform = Platform.CUDA
    min_sm: tuple[int, int] | None = None
    max_sm: tuple[int, int] | None = None
    supported_generations: frozenset[GPUGeneration] = field(
        default_factory=frozenset
    )
    min_tensor_core_gen: int = 0

    # Dtype support
    supported_dtypes: frozenset["torch.dtype"] = field(
        default_factory=frozenset
    )
    requires_dtype: "torch.dtype | None" = None
    produces_dtype: "torch.dtype | None" = None

    # Shape constraints
    min_head_dim: int = 1
    max_head_dim: int = 256
    head_dim_multiple: int = 1
    min_seq_len: int = 1
    max_seq_len: int | None = None
    min_batch_size: int = 1
    max_batch_size: int | None = None

    # Feature support
    supports_gqa: bool = True
    supports_mqa: bool = True
    supports_attn_mask: bool = True
    supported_attn_mask_types: frozenset[MaskType] = field(
        default_factory=lambda: frozenset([MaskType.NONE, MaskType.BOOL, MaskType.FLOAT])
    )
    supports_dropout: bool = True
    supports_scale: bool = True
    supports_alibi: bool = False

    # Layout requirements
    requires_last_dim_stride1: bool = True
    requires_contiguous: bool = False
    requires_layouts: frozenset[Layout] = field(
        default_factory=lambda: frozenset([Layout.BSHD, Layout.BHSD])
    )
    produces_layout: Layout | None = None

    # KV cache support
    supports_kv_cache_layouts: frozenset[Layout] = field(
        default_factory=frozenset
    )
    supports_kv_cache_dtypes: frozenset["torch.dtype"] = field(
        default_factory=frozenset
    )
    supports_kv_strategies: frozenset[KVCacheStrategy] = field(
        default_factory=frozenset
    )

    # Quantization
    supports_quant_formats: frozenset[QuantFormat] = field(
        default_factory=frozenset
    )
    requires_packed_weights: bool = False
    supports_prepack: bool = False

    # Execution properties
    is_cuda_graph_safe: bool = True
    deterministic: bool = False
    workspace_bytes: int = 0

    # Selection priority
    priority: int = 50

    # Fusion
    fuses_ops: tuple[str, ...] = ()
    transform_cost_hint: int = 0

    def check(self, ctx: "SelectionContext") -> list[Reason]:
        """Check if kernel is valid for context.

        Args:
            ctx: Selection context to validate against.

        Returns:
            Empty list if valid, else list of failure reasons.
        """
        reasons: list[Reason] = []

        # Platform check
        if ctx.device.platform != self.platform:
            reasons.append(make_reason(
                PLATFORM_MISMATCH,
                f"Kernel requires {self.platform.value}, got {ctx.device.platform.value}"
            ))
            return reasons  # No point checking further

        # SM version checks (only for CUDA)
        if self.platform == Platform.CUDA and ctx.device.sm_version:
            if self.min_sm and ctx.device.sm_version < self.min_sm:
                reasons.append(make_reason(
                    SM_TOO_OLD,
                    f"Kernel requires SM {self.min_sm}, got {ctx.device.sm_version}"
                ))

            if self.max_sm and ctx.device.sm_version > self.max_sm:
                reasons.append(make_reason(
                    SM_TOO_NEW,
                    f"Kernel requires SM <= {self.max_sm}, got {ctx.device.sm_version}"
                ))

        # Dtype check
        if self.supported_dtypes and ctx.dtype not in self.supported_dtypes:
            reasons.append(make_reason(
                DTYPE_UNSUPPORTED,
                f"Kernel does not support dtype {_dtype_to_str(ctx.dtype)}"
            ))

        # Head dimension checks
        if ctx.head_dim is not None:
            if ctx.head_dim < self.min_head_dim:
                reasons.append(make_reason(
                    HEAD_DIM_TOO_SMALL,
                    f"head_dim {ctx.head_dim} < min {self.min_head_dim}"
                ))

            if ctx.head_dim > self.max_head_dim:
                reasons.append(make_reason(
                    HEAD_DIM_TOO_LARGE,
                    f"head_dim {ctx.head_dim} > max {self.max_head_dim}"
                ))

            if self.head_dim_multiple > 1 and ctx.head_dim % self.head_dim_multiple != 0:
                reasons.append(make_reason(
                    HEAD_DIM_ALIGNMENT,
                    f"head_dim {ctx.head_dim} not multiple of {self.head_dim_multiple}"
                ))

        # Sequence length checks
        if ctx.seq_len_q is not None:
            if ctx.seq_len_q < self.min_seq_len:
                reasons.append(make_reason(
                    SEQ_TOO_SHORT,
                    f"seq_len {ctx.seq_len_q} < min {self.min_seq_len}"
                ))

            if self.max_seq_len is not None and ctx.seq_len_q > self.max_seq_len:
                reasons.append(make_reason(
                    SEQ_TOO_LONG,
                    f"seq_len {ctx.seq_len_q} > max {self.max_seq_len}"
                ))

        # GQA check
        if ctx.enable_gqa and not self.supports_gqa:
            reasons.append(make_reason(
                GQA_UNSUPPORTED,
                "Kernel does not support grouped query attention"
            ))

        # Layout check
        if self.requires_layouts and ctx.layout not in self.requires_layouts:
            reasons.append(make_reason(
                LAYOUT_UNSUPPORTED,
                f"Kernel requires layouts {self.requires_layouts}, got {ctx.layout}"
            ))

        # Contiguity check
        if self.requires_contiguous and not ctx.is_contiguous:
            reasons.append(make_reason(
                NOT_CONTIGUOUS,
                "Kernel requires contiguous tensor"
            ))

        # Stride check
        if self.requires_last_dim_stride1 and ctx.stride_last_dim != 1:
            reasons.append(make_reason(
                STRIDE_LAST_DIM,
                f"Kernel requires stride[-1]=1, got {ctx.stride_last_dim}"
            ))

        # CUDA graph safety check
        if ctx.is_cuda_graph_capturing and not self.is_cuda_graph_safe:
            reasons.append(make_reason(
                CUDA_GRAPH_UNSAFE,
                "Kernel is not safe for CUDA graph capture"
            ))

        # Determinism check
        if ctx.requires_deterministic and not self.deterministic:
            reasons.append(make_reason(
                NON_DETERMINISTIC,
                "Kernel is non-deterministic"
            ))

        return reasons

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excludes impl callable).

        Returns:
            Dict with kernel spec fields.
        """
        return {
            "kernel_id": self.kernel_id,
            "operation": self.operation,
            "source": self.source,
            "version": self.version,
            "platform": self.platform.value,
            "min_sm": list(self.min_sm) if self.min_sm else None,
            "max_sm": list(self.max_sm) if self.max_sm else None,
            "supported_generations": [g.value for g in self.supported_generations],
            "min_tensor_core_gen": self.min_tensor_core_gen,
            "supported_dtypes": [_dtype_to_str(d) for d in self.supported_dtypes],
            "requires_dtype": _dtype_to_str(self.requires_dtype) if self.requires_dtype else None,
            "produces_dtype": _dtype_to_str(self.produces_dtype) if self.produces_dtype else None,
            "min_head_dim": self.min_head_dim,
            "max_head_dim": self.max_head_dim,
            "head_dim_multiple": self.head_dim_multiple,
            "min_seq_len": self.min_seq_len,
            "max_seq_len": self.max_seq_len,
            "min_batch_size": self.min_batch_size,
            "max_batch_size": self.max_batch_size,
            "supports_gqa": self.supports_gqa,
            "supports_mqa": self.supports_mqa,
            "supports_attn_mask": self.supports_attn_mask,
            "supported_attn_mask_types": [m.value for m in self.supported_attn_mask_types],
            "supports_dropout": self.supports_dropout,
            "supports_scale": self.supports_scale,
            "supports_alibi": self.supports_alibi,
            "requires_last_dim_stride1": self.requires_last_dim_stride1,
            "requires_contiguous": self.requires_contiguous,
            "requires_layouts": [l.value for l in self.requires_layouts],
            "produces_layout": self.produces_layout.value if self.produces_layout else None,
            "supports_kv_cache_layouts": [l.value for l in self.supports_kv_cache_layouts],
            "supports_kv_cache_dtypes": [_dtype_to_str(d) for d in self.supports_kv_cache_dtypes],
            "supports_kv_strategies": [s.value for s in self.supports_kv_strategies],
            "supports_quant_formats": [q.value for q in self.supports_quant_formats],
            "requires_packed_weights": self.requires_packed_weights,
            "supports_prepack": self.supports_prepack,
            "is_cuda_graph_safe": self.is_cuda_graph_safe,
            "deterministic": self.deterministic,
            "workspace_bytes": self.workspace_bytes,
            "priority": self.priority,
            "fuses_ops": list(self.fuses_ops),
            "transform_cost_hint": self.transform_cost_hint,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "KernelSpec":
        """Deserialize from dictionary (impl will be None).

        Args:
            d: Dict with kernel spec fields.

        Returns:
            New KernelSpec instance with impl=None.
        """
        return cls(
            kernel_id=d["kernel_id"],
            operation=d["operation"],
            source=d["source"],
            version=d["version"],
            impl=None,  # Callable not serialized
            platform=Platform(d.get("platform", "cuda")),
            min_sm=tuple(d["min_sm"]) if d.get("min_sm") else None,
            max_sm=tuple(d["max_sm"]) if d.get("max_sm") else None,
            supported_generations=frozenset(
                GPUGeneration(g) for g in d.get("supported_generations", [])
            ),
            min_tensor_core_gen=d.get("min_tensor_core_gen", 0),
            supported_dtypes=frozenset(
                _str_to_dtype(dt) for dt in d.get("supported_dtypes", [])
                if _str_to_dtype(dt) is not None
            ),
            requires_dtype=_str_to_dtype(d["requires_dtype"]) if d.get("requires_dtype") else None,
            produces_dtype=_str_to_dtype(d["produces_dtype"]) if d.get("produces_dtype") else None,
            min_head_dim=d.get("min_head_dim", 1),
            max_head_dim=d.get("max_head_dim", 256),
            head_dim_multiple=d.get("head_dim_multiple", 1),
            min_seq_len=d.get("min_seq_len", 1),
            max_seq_len=d.get("max_seq_len"),
            min_batch_size=d.get("min_batch_size", 1),
            max_batch_size=d.get("max_batch_size"),
            supports_gqa=d.get("supports_gqa", True),
            supports_mqa=d.get("supports_mqa", True),
            supports_attn_mask=d.get("supports_attn_mask", True),
            supported_attn_mask_types=frozenset(
                MaskType(m) for m in d.get("supported_attn_mask_types", ["none", "bool", "float"])
            ),
            supports_dropout=d.get("supports_dropout", True),
            supports_scale=d.get("supports_scale", True),
            supports_alibi=d.get("supports_alibi", False),
            requires_last_dim_stride1=d.get("requires_last_dim_stride1", True),
            requires_contiguous=d.get("requires_contiguous", False),
            requires_layouts=frozenset(
                Layout(l) for l in d.get("requires_layouts", ["BSHD", "BHSD"])
            ),
            produces_layout=Layout(d["produces_layout"]) if d.get("produces_layout") else None,
            supports_kv_cache_layouts=frozenset(
                Layout(l) for l in d.get("supports_kv_cache_layouts", [])
            ),
            supports_kv_cache_dtypes=frozenset(
                _str_to_dtype(dt) for dt in d.get("supports_kv_cache_dtypes", [])
                if _str_to_dtype(dt) is not None
            ),
            supports_kv_strategies=frozenset(
                KVCacheStrategy(s) for s in d.get("supports_kv_strategies", [])
            ),
            supports_quant_formats=frozenset(
                QuantFormat(q) for q in d.get("supports_quant_formats", [])
            ),
            requires_packed_weights=d.get("requires_packed_weights", False),
            supports_prepack=d.get("supports_prepack", False),
            is_cuda_graph_safe=d.get("is_cuda_graph_safe", True),
            deterministic=d.get("deterministic", False),
            workspace_bytes=d.get("workspace_bytes", 0),
            priority=d.get("priority", 50),
            fuses_ops=tuple(d.get("fuses_ops", [])),
            transform_cost_hint=d.get("transform_cost_hint", 0),
        )
