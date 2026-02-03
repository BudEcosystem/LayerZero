"""
Test suite for KernelSpec dataclass.

Tests kernel specification and constraint checking.
Following TDD methodology - tests define expected behavior.
"""
import json
import pytest


class TestKernelSpecCreation:
    """Test KernelSpec construction."""

    def test_kernel_spec_required_fields(self):
        """KernelSpec must have kernel_id, operation, source, version."""
        from layerzero.models.kernel_spec import KernelSpec

        spec = KernelSpec(
            kernel_id="flash_attn.v3.causal",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
        )

        assert spec.kernel_id == "flash_attn.v3.causal"
        assert spec.operation == "attention.causal"
        assert spec.source == "flash_attn"
        assert spec.version == "3.0.0"

    def test_kernel_spec_defaults(self):
        """KernelSpec has sensible defaults for optional fields."""
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.enums import Platform, Layout, MaskType

        spec = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0.0",
        )

        # Check defaults
        assert spec.platform == Platform.CUDA
        assert spec.min_sm is None
        assert spec.max_sm is None
        assert spec.min_head_dim == 1
        assert spec.max_head_dim == 256
        assert spec.head_dim_multiple == 1
        assert spec.supports_gqa is True
        assert spec.is_cuda_graph_safe is True
        assert spec.priority == 50

    def test_kernel_spec_is_frozen(self):
        """KernelSpec is immutable (frozen)."""
        from layerzero.models.kernel_spec import KernelSpec

        spec = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0.0",
        )

        with pytest.raises(AttributeError):
            spec.priority = 100  # type: ignore

    def test_kernel_spec_is_hashable(self):
        """KernelSpec is hashable (can be used in sets)."""
        from layerzero.models.kernel_spec import KernelSpec

        spec1 = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0.0",
        )
        spec2 = KernelSpec(
            kernel_id="test.kernel",
            operation="attention.causal",
            source="test",
            version="1.0.0",
        )

        # Should be hashable
        kernel_set = {spec1, spec2}
        assert len(kernel_set) == 1


class TestKernelSpecCheck:
    """Test KernelSpec.check() method for constraint validation."""

    def test_check_returns_empty_for_valid_context(self):
        """check() returns empty list when kernel is valid for context."""
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform
        from layerzero.device import GPUGeneration
        import torch

        spec = KernelSpec(
            kernel_id="torch.sdpa.causal",
            operation="attention.causal",
            source="torch",
            version="2.0.0",
            platform=Platform.CUDA,
            supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
            max_head_dim=256,
            min_head_dim=16,
        )

        device = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 6),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=10 * 1024**3,
            available_memory_bytes=8 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.0",
            driver_version="525.0",
        )

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

        reasons = spec.check(ctx)
        assert reasons == []

    def test_check_platform_mismatch(self):
        """check() returns PLATFORM_MISMATCH when platforms differ."""
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform
        from layerzero.device import GPUGeneration
        from layerzero.reasons import PLATFORM_MISMATCH
        import torch

        spec = KernelSpec(
            kernel_id="flash_attn.v3",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
            platform=Platform.CUDA,
        )

        # CPU device
        device = DeviceSpec.cpu()

        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
        )

        reasons = spec.check(ctx)
        assert len(reasons) > 0
        assert any(r.code == PLATFORM_MISMATCH for r in reasons)

    def test_check_dtype_unsupported(self):
        """check() returns DTYPE_UNSUPPORTED when dtype not supported."""
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform
        from layerzero.device import GPUGeneration
        from layerzero.reasons import DTYPE_UNSUPPORTED
        import torch

        spec = KernelSpec(
            kernel_id="flash_attn.v3",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
            supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
        )

        device = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 6),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=10 * 1024**3,
            available_memory_bytes=8 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.0",
            driver_version="525.0",
        )

        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float32,  # Not supported
            batch_size=32,
        )

        reasons = spec.check(ctx)
        assert any(r.code == DTYPE_UNSUPPORTED for r in reasons)

    def test_check_head_dim_too_large(self):
        """check() returns HEAD_DIM_TOO_LARGE when head_dim exceeds max."""
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform
        from layerzero.device import GPUGeneration
        from layerzero.reasons import HEAD_DIM_TOO_LARGE
        import torch

        spec = KernelSpec(
            kernel_id="flash_attn.v2",
            operation="attention.causal",
            source="flash_attn",
            version="2.5.0",
            max_head_dim=128,  # Limited to 128
        )

        device = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 6),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=10 * 1024**3,
            available_memory_bytes=8 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.0",
            driver_version="525.0",
        )

        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            head_dim=256,  # Too large
            num_heads=16,
            num_kv_heads=16,
            seq_len_q=1024,
            seq_len_k=1024,
        )

        reasons = spec.check(ctx)
        assert any(r.code == HEAD_DIM_TOO_LARGE for r in reasons)

    def test_check_sm_too_old(self):
        """check() returns SM_TOO_OLD when GPU SM < min_sm."""
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform
        from layerzero.device import GPUGeneration
        from layerzero.reasons import SM_TOO_OLD
        import torch

        spec = KernelSpec(
            kernel_id="flash_attn.v3",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
            min_sm=(9, 0),  # Requires Hopper
        )

        device = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="RTX 3080",
            sm_version=(8, 6),  # Ampere - too old
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=10 * 1024**3,
            available_memory_bytes=8 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.0",
            driver_version="525.0",
        )

        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
        )

        reasons = spec.check(ctx)
        assert any(r.code == SM_TOO_OLD for r in reasons)

    def test_check_cuda_graph_unsafe(self):
        """check() returns CUDA_GRAPH_UNSAFE when capturing but kernel unsafe."""
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform
        from layerzero.device import GPUGeneration
        from layerzero.reasons import CUDA_GRAPH_UNSAFE
        import torch

        spec = KernelSpec(
            kernel_id="triton.custom",
            operation="attention.causal",
            source="triton",
            version="1.0.0",
            is_cuda_graph_safe=False,  # Not graph safe
        )

        device = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 6),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=10 * 1024**3,
            available_memory_bytes=8 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.0",
            driver_version="525.0",
        )

        ctx = SelectionContext(
            device=device,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=32,
            is_cuda_graph_capturing=True,  # Capturing graphs
        )

        reasons = spec.check(ctx)
        assert any(r.code == CUDA_GRAPH_UNSAFE for r in reasons)


class TestKernelSpecSerialization:
    """Test KernelSpec JSON serialization."""

    def test_kernel_spec_to_dict(self):
        """KernelSpec can be serialized to dict."""
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.enums import Platform
        import torch

        spec = KernelSpec(
            kernel_id="flash_attn.v3.causal",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
            platform=Platform.CUDA,
            min_sm=(9, 0),
            supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
            priority=100,
        )

        d = spec.to_dict()
        assert d["kernel_id"] == "flash_attn.v3.causal"
        assert d["platform"] == "cuda"
        assert d["min_sm"] == [9, 0]
        assert d["priority"] == 100

    def test_kernel_spec_json_roundtrip(self):
        """KernelSpec serialize/deserialize preserves fields (except impl)."""
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.enums import Platform
        import torch

        original = KernelSpec(
            kernel_id="xformers.causal",
            operation="attention.causal",
            source="xformers",
            version="0.0.25",
            platform=Platform.CUDA,
            min_sm=(7, 5),
            supported_dtypes=frozenset([torch.float16, torch.bfloat16, torch.float32]),
            max_head_dim=128,
            priority=75,
        )

        json_str = json.dumps(original.to_dict())
        d = json.loads(json_str)
        restored = KernelSpec.from_dict(d)

        assert restored.kernel_id == original.kernel_id
        assert restored.source == original.source
        assert restored.min_sm == original.min_sm
        assert restored.max_head_dim == original.max_head_dim
        assert restored.impl is None  # Callable not serialized


class TestKernelSpecWithImpl:
    """Test KernelSpec with implementation callable."""

    def test_kernel_spec_with_impl_callable(self):
        """KernelSpec can store a callable implementation."""
        from layerzero.models.kernel_spec import KernelSpec

        def my_kernel(q, k, v):
            return q @ k.transpose(-2, -1) @ v

        spec = KernelSpec(
            kernel_id="custom.kernel",
            operation="attention.causal",
            source="custom",
            version="1.0.0",
            impl=my_kernel,
        )

        assert spec.impl is not None
        assert callable(spec.impl)
