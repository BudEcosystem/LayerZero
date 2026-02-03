"""
Test suite for OperationSpec dataclass.

Tests operation specification and context validation.
Following TDD methodology - tests define expected behavior.
"""
import json
import pytest


class TestOperationSpecCreation:
    """Test OperationSpec construction."""

    def test_operation_spec_required_fields(self):
        """OperationSpec must have op_id, op_kind, required_fields."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.enums import OpKind

        spec = OperationSpec(
            op_id="attention.causal",
            op_kind=OpKind.TENSOR,
            required_fields=frozenset(["head_dim", "num_heads", "seq_len_q", "seq_len_k"]),
            has_fallback=True,
            fallback_impl=None,
            tolerances={},
        )

        assert spec.op_id == "attention.causal"
        assert spec.op_kind == OpKind.TENSOR
        assert "head_dim" in spec.required_fields

    def test_operation_spec_tokenization_required_fields(self):
        """Tokenization ops require tokenizer_id and vocab_hash."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.enums import OpKind

        spec = OperationSpec(
            op_id="tokenize.encode",
            op_kind=OpKind.TOKENIZATION,
            required_fields=frozenset(["tokenizer_id", "vocab_hash"]),
            has_fallback=True,
            fallback_impl=None,
            tolerances={},
        )

        assert spec.op_kind == OpKind.TOKENIZATION
        assert "tokenizer_id" in spec.required_fields
        assert "vocab_hash" in spec.required_fields

    def test_operation_spec_is_frozen(self):
        """OperationSpec is immutable (frozen)."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.enums import OpKind

        spec = OperationSpec(
            op_id="norm.rms",
            op_kind=OpKind.TENSOR,
            required_fields=frozenset(["batch_size"]),
            has_fallback=True,
            fallback_impl=None,
            tolerances={},
        )

        with pytest.raises(AttributeError):
            spec.op_id = "norm.layer"  # type: ignore


class TestOperationSpecValidation:
    """Test OperationSpec.validate_context() method."""

    def test_validate_context_returns_empty_for_valid(self):
        """validate_context() returns empty list when all required fields present."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        import torch

        spec = OperationSpec(
            op_id="attention.causal",
            op_kind=OpKind.TENSOR,
            required_fields=frozenset(["head_dim", "num_heads"]),
            has_fallback=True,
            fallback_impl=None,
            tolerances={},
        )

        device = DeviceSpec.cpu()
        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            head_dim=64,
            num_heads=16,
            num_kv_heads=16,
            seq_len_q=1024,
            seq_len_k=1024,
        )

        reasons = spec.validate_context(ctx)
        assert reasons == []

    def test_validate_context_returns_reasons_for_missing_head_dim(self):
        """validate_context() returns reason when head_dim missing."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        from layerzero.reasons import HEAD_DIM_INVALID
        import torch

        spec = OperationSpec(
            op_id="attention.causal",
            op_kind=OpKind.TENSOR,
            required_fields=frozenset(["head_dim", "num_heads"]),
            has_fallback=True,
            fallback_impl=None,
            tolerances={},
        )

        device = DeviceSpec.cpu()
        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            head_dim=None,  # Missing
            num_heads=16,
        )

        reasons = spec.validate_context(ctx)
        assert len(reasons) > 0
        assert any(r.code == HEAD_DIM_INVALID for r in reasons)

    def test_validate_context_tokenizer_missing_vocab_hash(self):
        """validate_context() returns reason when vocab_hash missing for tokenization."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        from layerzero.reasons import VOCAB_HASH_MISMATCH
        import torch

        spec = OperationSpec(
            op_id="tokenize.encode",
            op_kind=OpKind.TOKENIZATION,
            required_fields=frozenset(["tokenizer_id", "vocab_hash"]),
            has_fallback=True,
            fallback_impl=None,
            tolerances={},
        )

        device = DeviceSpec.cpu()
        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TOKENIZATION,
            operation="tokenize.encode",
            dtype=torch.int64,
            batch_size=1,
            tokenizer_id="gpt2",
            vocab_hash=None,  # Missing
        )

        reasons = spec.validate_context(ctx)
        assert len(reasons) > 0


class TestOperationSpecWithFallback:
    """Test OperationSpec with fallback implementation."""

    def test_operation_spec_has_fallback_true(self):
        """OperationSpec can indicate it has a fallback."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.enums import OpKind

        def fallback_attention(q, k, v):
            # Reference implementation
            import torch.nn.functional as F
            return F.scaled_dot_product_attention(q, k, v)

        spec = OperationSpec(
            op_id="attention.causal",
            op_kind=OpKind.TENSOR,
            required_fields=frozenset(["head_dim"]),
            has_fallback=True,
            fallback_impl=fallback_attention,
            tolerances={},
        )

        assert spec.has_fallback is True
        assert spec.fallback_impl is not None
        assert callable(spec.fallback_impl)

    def test_operation_spec_tolerances(self):
        """OperationSpec can specify tolerances per dtype."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.enums import OpKind
        import torch

        spec = OperationSpec(
            op_id="attention.causal",
            op_kind=OpKind.TENSOR,
            required_fields=frozenset(),
            has_fallback=True,
            fallback_impl=None,
            tolerances={
                torch.float16: (1e-3, 1e-3),  # (rtol, atol)
                torch.bfloat16: (1e-2, 1e-2),
                torch.float32: (1e-5, 1e-5),
            },
        )

        assert spec.tolerances[torch.float16] == (1e-3, 1e-3)
        assert spec.tolerances[torch.bfloat16] == (1e-2, 1e-2)


class TestOperationSpecSerialization:
    """Test OperationSpec JSON serialization."""

    def test_operation_spec_to_dict(self):
        """OperationSpec can be serialized to dict."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.enums import OpKind

        spec = OperationSpec(
            op_id="norm.rms",
            op_kind=OpKind.TENSOR,
            required_fields=frozenset(["batch_size"]),
            has_fallback=True,
            fallback_impl=None,
            tolerances={},
        )

        d = spec.to_dict()
        assert d["op_id"] == "norm.rms"
        assert d["op_kind"] == "tensor"
        assert "batch_size" in d["required_fields"]

    def test_operation_spec_json_roundtrip(self):
        """OperationSpec serialize/deserialize preserves key fields."""
        from layerzero.models.operation_spec import OperationSpec
        from layerzero.enums import OpKind

        original = OperationSpec(
            op_id="attention.full",
            op_kind=OpKind.TENSOR,
            required_fields=frozenset(["head_dim", "num_heads"]),
            has_fallback=True,
            fallback_impl=None,
            tolerances={},
        )

        json_str = json.dumps(original.to_dict())
        d = json.loads(json_str)
        restored = OperationSpec.from_dict(d)

        assert restored.op_id == original.op_id
        assert restored.op_kind == original.op_kind
        assert restored.has_fallback == original.has_fallback
