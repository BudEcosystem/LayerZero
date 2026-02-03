"""
Test suite for SelectionContext dataclass.

Tests context building from tensors and validation.
Following TDD methodology - tests define expected behavior.
"""
import json
import pytest


class TestSelectionContextCreation:
    """Test SelectionContext construction."""

    def test_selection_context_required_fields(self):
        """SelectionContext must have device, op_kind, operation, dtype, batch_size."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform
        from layerzero.device import GPUGeneration
        import torch

        device = DeviceSpec.cpu()
        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
        )

        assert ctx.device == device
        assert ctx.op_kind == OpKind.TENSOR
        assert ctx.operation == "attention.causal"
        assert ctx.dtype == torch.float16
        assert ctx.batch_size == 32

    def test_selection_context_defaults(self):
        """SelectionContext has sensible defaults for optional fields."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Layout, MaskType
        import torch

        device = DeviceSpec.cpu()
        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="norm.rms",
            dtype=torch.float32,
            batch_size=16,
        )

        # Check defaults
        assert ctx.layout == Layout.BSHD
        assert ctx.attn_mask_type == MaskType.NONE
        assert ctx.is_causal is False
        assert ctx.dropout_p == 0.0
        assert ctx.is_cuda_graph_capturing is False
        assert ctx.tp_size == 1
        assert ctx.pp_size == 1
        assert ctx.rank == 0

    def test_selection_context_is_frozen(self):
        """SelectionContext is immutable (frozen)."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        import torch

        device = DeviceSpec.cpu()
        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
        )

        with pytest.raises(AttributeError):
            ctx.batch_size = 64  # type: ignore


class TestSelectionContextFromTensors:
    """Test SelectionContext.from_tensors() class method."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not available"),
        reason="torch not available"
    )
    def test_from_tensors_basic(self):
        """SelectionContext.from_tensors() extracts tensor properties."""
        import torch
        from layerzero.models.selection_context import SelectionContext

        q = torch.randn(2, 1024, 32, 128, dtype=torch.float16)
        k = torch.randn(2, 1024, 8, 128, dtype=torch.float16)
        v = torch.randn(2, 1024, 8, 128, dtype=torch.float16)

        ctx = SelectionContext.from_tensors(q, k, v, is_causal=True)

        assert ctx.dtype == torch.float16
        assert ctx.batch_size == 2
        assert ctx.seq_len_q == 1024
        assert ctx.seq_len_k == 1024
        assert ctx.num_heads == 32
        assert ctx.num_kv_heads == 8
        assert ctx.head_dim == 128
        assert ctx.is_causal is True

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not available"),
        reason="torch not available"
    )
    def test_from_tensors_detects_gqa(self):
        """from_tensors() sets enable_gqa when num_heads != num_kv_heads."""
        import torch
        from layerzero.models.selection_context import SelectionContext

        q = torch.randn(1, 512, 32, 64, dtype=torch.bfloat16)
        k = torch.randn(1, 512, 8, 64, dtype=torch.bfloat16)
        v = torch.randn(1, 512, 8, 64, dtype=torch.bfloat16)

        ctx = SelectionContext.from_tensors(q, k, v)

        assert ctx.num_heads == 32
        assert ctx.num_kv_heads == 8
        assert ctx.enable_gqa is True

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not available"),
        reason="torch not available"
    )
    def test_from_tensors_detects_layout_bshd(self):
        """from_tensors() detects BSHD layout (batch, seq, heads, dim)."""
        import torch
        from layerzero.models.selection_context import SelectionContext
        from layerzero.enums import Layout

        # BSHD: (batch, seq, heads, dim) - dim should be last
        q = torch.randn(2, 1024, 16, 64, dtype=torch.float16)
        k = torch.randn(2, 1024, 16, 64, dtype=torch.float16)
        v = torch.randn(2, 1024, 16, 64, dtype=torch.float16)

        ctx = SelectionContext.from_tensors(q, k, v, layout=Layout.BSHD)

        assert ctx.layout == Layout.BSHD

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not available"),
        reason="torch not available"
    )
    def test_from_tensors_detects_contiguity(self):
        """from_tensors() detects tensor contiguity."""
        import torch
        from layerzero.models.selection_context import SelectionContext

        q = torch.randn(2, 1024, 16, 64, dtype=torch.float16)
        k = torch.randn(2, 1024, 16, 64, dtype=torch.float16)
        v = torch.randn(2, 1024, 16, 64, dtype=torch.float16)

        ctx = SelectionContext.from_tensors(q, k, v)
        assert ctx.is_contiguous is True

        # Non-contiguous via transpose
        q_nc = q.transpose(1, 2)
        ctx_nc = SelectionContext.from_tensors(q_nc, k.transpose(1, 2), v.transpose(1, 2))
        assert ctx_nc.is_contiguous is False


class TestSelectionContextForNorm:
    """Test SelectionContext.for_norm() class method."""

    @pytest.mark.skipif(
        not pytest.importorskip("torch", reason="torch not available"),
        reason="torch not available"
    )
    def test_for_norm_basic(self):
        """SelectionContext.for_norm() builds context for normalization."""
        import torch
        from layerzero.models.selection_context import SelectionContext
        from layerzero.enums import OpKind

        x = torch.randn(32, 1024, 4096, dtype=torch.float32)
        ctx = SelectionContext.for_norm(x, operation="norm.rms")

        assert ctx.op_kind == OpKind.TENSOR
        assert ctx.operation == "norm.rms"
        assert ctx.dtype == torch.float32
        assert ctx.batch_size == 32


class TestSelectionContextCacheKey:
    """Test SelectionContext.cache_key() method."""

    def test_cache_key_is_string(self):
        """cache_key() returns a string."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        import torch

        device = DeviceSpec.cpu()
        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            seq_len_q=1024,
            seq_len_k=1024,
            num_heads=16,
            num_kv_heads=16,
            head_dim=64,
        )

        key = ctx.cache_key()
        assert isinstance(key, str)
        assert len(key) > 0

    def test_cache_key_deterministic(self):
        """Same context produces same cache_key()."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        import torch

        device = DeviceSpec.cpu()

        ctx1 = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            seq_len_q=1024,
            seq_len_k=1024,
            num_heads=16,
            num_kv_heads=16,
            head_dim=64,
        )

        ctx2 = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            seq_len_q=1024,
            seq_len_k=1024,
            num_heads=16,
            num_kv_heads=16,
            head_dim=64,
        )

        assert ctx1.cache_key() == ctx2.cache_key()

    def test_cache_key_differs_for_different_shapes(self):
        """Different shapes produce different cache_key()."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        import torch

        device = DeviceSpec.cpu()

        ctx1 = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            seq_len_q=1024,
            seq_len_k=1024,
            num_heads=16,
            num_kv_heads=16,
            head_dim=64,
        )

        ctx2 = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            seq_len_q=2048,  # Different seq_len
            seq_len_k=2048,
            num_heads=16,
            num_kv_heads=16,
            head_dim=64,
        )

        assert ctx1.cache_key() != ctx2.cache_key()


class TestSelectionContextSerialization:
    """Test SelectionContext JSON serialization."""

    def test_selection_context_to_dict(self):
        """SelectionContext can be serialized to dict."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        import torch

        device = DeviceSpec.cpu()
        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            seq_len_q=1024,
            seq_len_k=1024,
            num_heads=16,
            num_kv_heads=16,
            head_dim=64,
            is_causal=True,
        )

        d = ctx.to_dict()
        assert d["operation"] == "attention.causal"
        assert d["batch_size"] == 32
        assert d["is_causal"] is True

    def test_selection_context_json_roundtrip(self):
        """SelectionContext serialize/deserialize preserves key fields."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        import torch

        device = DeviceSpec.cpu()
        original = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="norm.rms",
            dtype=torch.bfloat16,
            batch_size=64,
        )

        json_str = json.dumps(original.to_dict())
        d = json.loads(json_str)
        restored = SelectionContext.from_dict(d)

        assert restored.operation == original.operation
        assert restored.batch_size == original.batch_size


class TestSelectionContextTokenization:
    """Test SelectionContext for tokenization operations."""

    def test_tokenization_context_required_fields(self):
        """Tokenization ops should have tokenizer_id and vocab_hash."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind
        import torch

        device = DeviceSpec.cpu()
        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TOKENIZATION,
            operation="tokenize.encode",
            dtype=torch.int64,
            batch_size=1,
            tokenizer_id="gpt2",
            vocab_hash="abc123",
            merges_hash="def456",
        )

        assert ctx.op_kind == OpKind.TOKENIZATION
        assert ctx.tokenizer_id == "gpt2"
        assert ctx.vocab_hash == "abc123"
