"""
LayerZero Operation Specification

Dataclass describing an operation type and its requirements.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

from layerzero.enums import OpKind
from layerzero.reasons import (
    HEAD_DIM_INVALID,
    VOCAB_HASH_MISMATCH,
    TOKENIZER_ID_MISMATCH,
    Reason,
    ReasonCategory,
    make_reason,
)

if TYPE_CHECKING:
    import torch
    from layerzero.models.selection_context import SelectionContext


@dataclass(frozen=True, slots=True)
class OperationSpec:
    """Operation specification.

    Describes an operation type, its requirements, and fallback.
    Immutable (frozen) for hashability and thread safety.

    Attributes:
        op_id: Operation identifier (e.g., "attention.causal")
        op_kind: Operation kind (tensor, tokenization, etc.)
        required_fields: Set of required context fields
        has_fallback: Whether a fallback implementation exists
        fallback_impl: Fallback implementation callable
        tolerances: Dict of dtype to (rtol, atol) tolerance tuples
    """

    op_id: str
    op_kind: OpKind
    required_fields: frozenset[str]
    has_fallback: bool
    fallback_impl: Callable | None
    tolerances: dict["torch.dtype", tuple[float, float]]

    def validate_context(self, ctx: "SelectionContext") -> list[Reason]:
        """Validate context has required fields.

        Args:
            ctx: Selection context to validate.

        Returns:
            Empty list if valid, else list of failure reasons.
        """
        reasons: list[Reason] = []

        for field_name in self.required_fields:
            value = getattr(ctx, field_name, None)

            if value is None:
                # Map field names to appropriate reason codes
                if field_name == "head_dim":
                    reasons.append(make_reason(
                        HEAD_DIM_INVALID,
                        f"Required field '{field_name}' is missing"
                    ))
                elif field_name == "vocab_hash":
                    reasons.append(make_reason(
                        VOCAB_HASH_MISMATCH,
                        f"Required field '{field_name}' is missing"
                    ))
                elif field_name == "tokenizer_id":
                    reasons.append(make_reason(
                        TOKENIZER_ID_MISMATCH,
                        f"Required field '{field_name}' is missing"
                    ))
                else:
                    # Generic validation failure
                    reasons.append(Reason(
                        code=f"MISSING_{field_name.upper()}",
                        message=f"Required field '{field_name}' is missing",
                        category=ReasonCategory.SCHEMA,
                    ))

        return reasons

    def get_tolerance(self, dtype: "torch.dtype") -> tuple[float, float]:
        """Get tolerance for a dtype.

        Args:
            dtype: Data type to get tolerance for.

        Returns:
            Tuple of (rtol, atol), defaults to (1e-5, 1e-5).
        """
        return self.tolerances.get(dtype, (1e-5, 1e-5))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary (excludes fallback_impl).

        Returns:
            Dict with operation spec fields.
        """
        # Convert tolerances - dtype keys need special handling
        tolerances_dict = {}
        for dtype, (rtol, atol) in self.tolerances.items():
            dtype_str = str(dtype).replace("torch.", "")
            tolerances_dict[dtype_str] = [rtol, atol]

        return {
            "op_id": self.op_id,
            "op_kind": self.op_kind.value,
            "required_fields": list(self.required_fields),
            "has_fallback": self.has_fallback,
            "tolerances": tolerances_dict,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OperationSpec":
        """Deserialize from dictionary (fallback_impl will be None).

        Args:
            d: Dict with operation spec fields.

        Returns:
            New OperationSpec instance with fallback_impl=None.
        """
        # Convert tolerances back to dtype keys
        tolerances: dict[Any, tuple[float, float]] = {}
        if "tolerances" in d:
            import torch
            dtype_map = {
                "float16": torch.float16,
                "float32": torch.float32,
                "float64": torch.float64,
                "bfloat16": torch.bfloat16,
            }
            for dtype_str, (rtol, atol) in d["tolerances"].items():
                if dtype_str in dtype_map:
                    tolerances[dtype_map[dtype_str]] = (rtol, atol)

        return cls(
            op_id=d["op_id"],
            op_kind=OpKind(d["op_kind"]),
            required_fields=frozenset(d.get("required_fields", [])),
            has_fallback=d.get("has_fallback", True),
            fallback_impl=None,  # Callable not serialized
            tolerances=tolerances,
        )


# Pre-defined operation specs for common operations
def attention_causal_spec() -> OperationSpec:
    """Create OperationSpec for causal attention."""
    import torch
    return OperationSpec(
        op_id="attention.causal",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["head_dim", "num_heads", "seq_len_q", "seq_len_k"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
            torch.float32: (1e-5, 1e-5),
        },
    )


def attention_full_spec() -> OperationSpec:
    """Create OperationSpec for full (bidirectional) attention."""
    import torch
    return OperationSpec(
        op_id="attention.full",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["head_dim", "num_heads", "seq_len_q", "seq_len_k"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
            torch.float32: (1e-5, 1e-5),
        },
    )


def rms_norm_spec() -> OperationSpec:
    """Create OperationSpec for RMSNorm."""
    import torch
    return OperationSpec(
        op_id="norm.rms",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["batch_size"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
            torch.float32: (1e-5, 1e-5),
        },
    )


def layer_norm_spec() -> OperationSpec:
    """Create OperationSpec for LayerNorm."""
    import torch
    return OperationSpec(
        op_id="norm.layer",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["batch_size"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
            torch.float32: (1e-5, 1e-5),
        },
    )


def tokenize_encode_spec() -> OperationSpec:
    """Create OperationSpec for tokenization encoding."""
    return OperationSpec(
        op_id="tokenize.encode",
        op_kind=OpKind.TOKENIZATION,
        required_fields=frozenset(["tokenizer_id", "vocab_hash"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={},
    )


def sampling_topk_spec() -> OperationSpec:
    """Create OperationSpec for top-k sampling."""
    return OperationSpec(
        op_id="sampling.topk",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["batch_size"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={},  # Sampling is stochastic, no numerical tolerance
    )


def sampling_topp_spec() -> OperationSpec:
    """Create OperationSpec for top-p (nucleus) sampling."""
    return OperationSpec(
        op_id="sampling.topp",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["batch_size"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={},  # Sampling is stochastic, no numerical tolerance
    )


def embedding_lookup_spec() -> OperationSpec:
    """Create OperationSpec for embedding lookup."""
    import torch
    return OperationSpec(
        op_id="embedding.lookup",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["batch_size"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={
            torch.float16: (1e-3, 1e-3),
            torch.float32: (1e-5, 1e-5),
        },
    )


def posenc_alibi_spec() -> OperationSpec:
    """Create OperationSpec for ALiBi positional encoding."""
    import torch
    return OperationSpec(
        op_id="posenc.alibi",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["num_heads", "seq_len_q"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={
            torch.float16: (1e-3, 1e-3),
            torch.float32: (1e-5, 1e-5),
        },
    )


def mlp_fused_spec() -> OperationSpec:
    """Create OperationSpec for fused MLP operation."""
    import torch
    return OperationSpec(
        op_id="mlp.fused",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["batch_size"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
            torch.float32: (1e-5, 1e-5),
        },
    )


def mlp_linear_spec() -> OperationSpec:
    """Create OperationSpec for linear/GEMM operation."""
    import torch
    return OperationSpec(
        op_id="mlp.linear",
        op_kind=OpKind.TENSOR,
        required_fields=frozenset(["batch_size"]),
        has_fallback=True,
        fallback_impl=None,
        tolerances={
            torch.float16: (1e-3, 1e-3),
            torch.bfloat16: (1e-2, 1e-2),
            torch.float32: (1e-5, 1e-5),
        },
    )
