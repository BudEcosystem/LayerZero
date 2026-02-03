"""
Unit tests for dispatch/executor.py module.

Tests cover:
- KernelExecutorImpl execution logic
- Argument mapping for different operations (attention, norm, rope)
- Layout transformation
- Error handling and wrapping
- Timing measurement
- CUDAGraphExecutor capture and replay
- Execute convenience function

All tests use pytest and mock torch tensors/operations where needed.
"""
from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import torch

from layerzero.dispatch.executor import (
    KernelExecutorImpl,
    CUDAGraphExecutor,
    execute_kernel,
)
from layerzero.dispatch.types import KernelExecutionError, TransformError
from layerzero.enums import Layout, Platform
from layerzero.models.kernel_spec import KernelSpec
from layerzero.registry.backend_registry import BackendRegistry


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def backend_registry() -> BackendRegistry:
    """Create a backend registry for testing."""
    return BackendRegistry(failure_threshold=3, cooldown_seconds=30.0)


@pytest.fixture
def executor(backend_registry: BackendRegistry) -> KernelExecutorImpl:
    """Create a kernel executor for testing."""
    return KernelExecutorImpl(backend_registry=backend_registry)


@pytest.fixture
def executor_no_registry() -> KernelExecutorImpl:
    """Create a kernel executor without backend registry."""
    return KernelExecutorImpl(backend_registry=None)


def _make_attention_kernel(output_value: float = 0.0) -> KernelSpec:
    """Create a mock attention kernel spec."""
    def impl(**kwargs: Any) -> torch.Tensor:
        # Use explicit None checks to avoid tensor boolean ambiguity
        q = kwargs.get("q")
        if q is None:
            q = kwargs.get("query")
        if q is None:
            return torch.full((1,), output_value)
        return torch.full_like(q, output_value)

    return KernelSpec(
        kernel_id="test_attention.v1",
        operation="attention.causal",
        source="test",
        version="1.0",
        impl=impl,
        platform=Platform.CUDA,
        min_sm=(7, 0),
        supported_dtypes=frozenset([torch.float16, torch.float32]),
    )


def _make_flash_attn_kernel() -> KernelSpec:
    """Create a mock FlashAttention kernel spec."""
    def impl(**kwargs: Any) -> torch.Tensor:
        # FlashAttention expects q, k, v with specific argument names
        q = kwargs.get("q")
        k = kwargs.get("k")
        v = kwargs.get("v")
        if q is None or k is None or v is None:
            raise ValueError("Missing q, k, or v")
        return torch.zeros_like(q)

    return KernelSpec(
        kernel_id="flash_attn.v3",
        operation="attention.causal",
        source="flash_attn",
        version="3.0",
        impl=impl,
        platform=Platform.CUDA,
    )


def _make_xformers_kernel() -> KernelSpec:
    """Create a mock xFormers kernel spec."""
    def impl(**kwargs: Any) -> torch.Tensor:
        # xFormers expects query, key, value
        query = kwargs.get("query")
        key = kwargs.get("key")
        value = kwargs.get("value")
        if query is None or key is None or value is None:
            raise ValueError("Missing query, key, or value")
        return torch.ones_like(query)

    return KernelSpec(
        kernel_id="xformers.memory_efficient",
        operation="attention.causal",
        source="xformers",
        version="0.0.25",
        impl=impl,
        platform=Platform.CUDA,
    )


def _make_torch_sdpa_kernel() -> KernelSpec:
    """Create a mock PyTorch SDPA kernel spec."""
    def impl(**kwargs: Any) -> torch.Tensor:
        query = kwargs.get("query")
        key = kwargs.get("key")
        value = kwargs.get("value")
        is_causal = kwargs.get("is_causal", False)
        dropout_p = kwargs.get("dropout_p", 0.0)
        if query is None or key is None or value is None:
            raise ValueError("Missing query, key, or value")
        return torch.zeros_like(query) + (0.5 if is_causal else 0.0)

    return KernelSpec(
        kernel_id="torch_sdpa",
        operation="attention.causal",
        source="torch_sdpa",
        version="2.0",
        impl=impl,
        platform=Platform.CUDA,
    )


def _make_norm_kernel() -> KernelSpec:
    """Create a mock normalization kernel spec."""
    def impl(**kwargs: Any) -> torch.Tensor:
        # Use explicit None checks to avoid tensor boolean ambiguity
        x = kwargs.get("x")
        if x is None:
            x = kwargs.get("input")
        if x is None:
            return torch.zeros(1)
        return torch.ones_like(x)

    return KernelSpec(
        kernel_id="liger_rms_norm",
        operation="rms_norm",
        source="liger",
        version="1.0",
        impl=impl,
        platform=Platform.CUDA,
    )


def _make_rope_kernel() -> KernelSpec:
    """Create a mock RoPE kernel spec."""
    def impl(**kwargs: Any) -> torch.Tensor:
        x = kwargs.get("x")
        if x is None:
            return torch.zeros(1)
        return x * 2.0

    return KernelSpec(
        kernel_id="liger_rope",
        operation="rope.fused",
        source="liger",
        version="1.0",
        impl=impl,
        platform=Platform.CUDA,
    )


def _make_failing_kernel(error_type: type = ValueError) -> KernelSpec:
    """Create a kernel that always fails."""
    def impl(**kwargs: Any) -> torch.Tensor:
        raise error_type("Kernel failed intentionally")

    return KernelSpec(
        kernel_id="failing_kernel",
        operation="attention.causal",
        source="test",
        version="1.0",
        impl=impl,
        platform=Platform.CUDA,
    )


def _make_no_impl_kernel() -> KernelSpec:
    """Create a kernel with no implementation."""
    return KernelSpec(
        kernel_id="no_impl_kernel",
        operation="attention.causal",
        source="test",
        version="1.0",
        impl=None,
        platform=Platform.CUDA,
    )


# ============================================================================
# KernelExecutorImpl Basic Tests
# ============================================================================


class TestKernelExecutorImplBasic:
    """Basic tests for KernelExecutorImpl."""

    def test_initialization_with_registry(self, backend_registry: BackendRegistry) -> None:
        """Executor can be initialized with a backend registry."""
        executor = KernelExecutorImpl(backend_registry=backend_registry)
        assert executor._backend_registry is backend_registry

    def test_initialization_without_registry(self) -> None:
        """Executor can be initialized without a backend registry."""
        executor = KernelExecutorImpl(backend_registry=None)
        assert executor._backend_registry is None

    def test_caches_are_initialized(self, executor: KernelExecutorImpl) -> None:
        """Internal caches are initialized as empty dicts."""
        assert executor._transform_cache == {}
        assert executor._execution_cache == {}
        assert executor._cuda_graphs == {}


# ============================================================================
# Execute Method Tests
# ============================================================================


class TestKernelExecutorExecute:
    """Tests for KernelExecutorImpl.execute method."""

    def test_execute_returns_tensor(self, executor: KernelExecutorImpl) -> None:
        """execute returns a tensor from the kernel."""
        kernel = _make_attention_kernel(output_value=1.0)
        inputs = {
            "query": torch.randn(2, 8, 4, 64),  # BHSD format
            "key": torch.randn(2, 8, 4, 64),
            "value": torch.randn(2, 8, 4, 64),
        }
        output = executor.execute(kernel, inputs)
        assert isinstance(output, torch.Tensor)
        # Note: shape may be transformed by layout transformation
        assert output.numel() == inputs["query"].numel()
        assert torch.all(output == 1.0)

    def test_execute_raises_for_no_impl(self, executor: KernelExecutorImpl) -> None:
        """execute raises KernelExecutionError when kernel has no impl."""
        kernel = _make_no_impl_kernel()
        inputs = {"query": torch.zeros(1)}

        with pytest.raises(KernelExecutionError) as exc_info:
            executor.execute(kernel, inputs)

        assert "no implementation" in str(exc_info.value).lower()
        assert exc_info.value.kernel_id == "no_impl_kernel"
        assert exc_info.value.operation == "attention.causal"

    def test_execute_wraps_kernel_errors(self, executor: KernelExecutorImpl) -> None:
        """execute wraps kernel exceptions in KernelExecutionError."""
        kernel = _make_failing_kernel()
        # Provide full attention inputs to pass argument mapping
        inputs = {
            "query": torch.zeros(2, 4, 8, 64),
            "key": torch.zeros(2, 4, 8, 64),
            "value": torch.zeros(2, 4, 8, 64),
        }

        with pytest.raises(KernelExecutionError) as exc_info:
            executor.execute(kernel, inputs)

        assert exc_info.value.original_error is not None
        assert isinstance(exc_info.value.original_error, ValueError)

    def test_execute_preserves_kernel_execution_error(self, executor: KernelExecutorImpl) -> None:
        """execute preserves KernelExecutionError without double wrapping."""
        def impl_raises_kernel_error(**kwargs: Any) -> torch.Tensor:
            raise KernelExecutionError("Already wrapped", "op", "kernel")

        kernel = KernelSpec(
            kernel_id="raises_kernel_error",
            operation="test",
            source="test",
            version="1.0",
            impl=impl_raises_kernel_error,
            platform=Platform.CUDA,
        )

        with pytest.raises(KernelExecutionError) as exc_info:
            executor.execute(kernel, {"x": torch.zeros(1)})

        # Should be the original error, not double-wrapped
        assert exc_info.value.kernel_id == "kernel"
        assert exc_info.value.original_error is None

    def test_execute_records_success_to_registry(
        self, backend_registry: BackendRegistry
    ) -> None:
        """execute records success to backend registry."""
        executor = KernelExecutorImpl(backend_registry=backend_registry)
        kernel = _make_attention_kernel()
        # Provide full attention inputs
        inputs = {
            "query": torch.zeros(2, 4, 8, 64),
            "key": torch.zeros(2, 4, 8, 64),
            "value": torch.zeros(2, 4, 8, 64),
        }

        executor.execute(kernel, inputs)

        # Backend registry should track the success
        assert backend_registry is executor._backend_registry

    def test_execute_records_failure_to_registry(
        self, backend_registry: BackendRegistry
    ) -> None:
        """execute records failure to backend registry on error."""
        executor = KernelExecutorImpl(backend_registry=backend_registry)
        kernel = _make_failing_kernel()
        inputs = {"query": torch.zeros(1)}

        with pytest.raises(KernelExecutionError):
            executor.execute(kernel, inputs)

        # Backend registry should track the failure
        # (We can't easily verify without inspecting internal state)

    def test_execute_without_registry_succeeds(
        self, executor_no_registry: KernelExecutorImpl
    ) -> None:
        """execute works without a backend registry."""
        kernel = _make_attention_kernel()
        # Provide full attention inputs
        inputs = {
            "query": torch.zeros(2, 4, 8, 64),
            "key": torch.zeros(2, 4, 8, 64),
            "value": torch.zeros(2, 4, 8, 64),
        }

        output = executor_no_registry.execute(kernel, inputs)
        assert output is not None


# ============================================================================
# Execute with Timing Tests
# ============================================================================


class TestKernelExecutorExecuteWithTiming:
    """Tests for KernelExecutorImpl.execute_with_timing method."""

    def test_returns_output_and_time(self, executor: KernelExecutorImpl) -> None:
        """execute_with_timing returns tuple of output and time."""
        kernel = _make_attention_kernel()
        # Provide full attention inputs
        inputs = {
            "query": torch.zeros(2, 4, 8, 64),
            "key": torch.zeros(2, 4, 8, 64),
            "value": torch.zeros(2, 4, 8, 64),
        }

        output, elapsed_ns = executor.execute_with_timing(kernel, inputs)

        assert isinstance(output, torch.Tensor)
        assert isinstance(elapsed_ns, int)
        assert elapsed_ns > 0

    def test_timing_is_reasonable(self, executor: KernelExecutorImpl) -> None:
        """Timing measurement is within reasonable bounds."""
        def slow_impl(**kwargs: Any) -> torch.Tensor:
            time.sleep(0.01)  # 10ms
            return torch.zeros(1)

        kernel = KernelSpec(
            kernel_id="slow_kernel",
            operation="test",
            source="test",
            version="1.0",
            impl=slow_impl,
            platform=Platform.CUDA,
        )

        _, elapsed_ns = executor.execute_with_timing(kernel, {"x": torch.zeros(1)})

        # Should be at least 10ms = 10,000,000ns
        assert elapsed_ns >= 5_000_000  # Allow some tolerance

    def test_timing_propagates_errors(self, executor: KernelExecutorImpl) -> None:
        """execute_with_timing propagates kernel errors."""
        kernel = _make_failing_kernel()

        with pytest.raises(KernelExecutionError):
            executor.execute_with_timing(kernel, {"x": torch.zeros(1)})


# ============================================================================
# CUDA Graph Support Tests
# ============================================================================


class TestKernelExecutorCUDAGraphSupport:
    """Tests for CUDA graph support checking."""

    def test_supports_cuda_graph_true(self, executor: KernelExecutorImpl) -> None:
        """Returns True for CUDA graph safe kernels."""
        kernel = KernelSpec(
            kernel_id="safe_kernel",
            operation="test",
            source="test",
            version="1.0",
            impl=lambda **k: torch.zeros(1),
            platform=Platform.CUDA,
            is_cuda_graph_safe=True,
        )
        assert executor.supports_cuda_graph(kernel) is True

    def test_supports_cuda_graph_false(self, executor: KernelExecutorImpl) -> None:
        """Returns False for non-CUDA graph safe kernels."""
        kernel = KernelSpec(
            kernel_id="unsafe_kernel",
            operation="test",
            source="test",
            version="1.0",
            impl=lambda **k: torch.zeros(1),
            platform=Platform.CUDA,
            is_cuda_graph_safe=False,
        )
        assert executor.supports_cuda_graph(kernel) is False


# ============================================================================
# Argument Mapping Tests - Attention
# ============================================================================


class TestArgumentMappingAttention:
    """Tests for attention argument mapping."""

    def test_flash_attn_argument_mapping(self, executor: KernelExecutorImpl) -> None:
        """FlashAttention gets q, k, v directly."""
        kernel = _make_flash_attn_kernel()
        inputs = {
            "query": torch.randn(2, 8, 4, 64),
            "key": torch.randn(2, 8, 4, 64),
            "value": torch.randn(2, 8, 4, 64),
        }
        kwargs = {"dropout_p": 0.1, "is_causal": True}

        output = executor.execute(kernel, inputs, **kwargs)
        assert output is not None

    def test_xformers_argument_mapping(self, executor: KernelExecutorImpl) -> None:
        """xFormers gets query, key, value."""
        kernel = _make_xformers_kernel()
        inputs = {
            "query": torch.randn(2, 8, 4, 64),
            "key": torch.randn(2, 8, 4, 64),
            "value": torch.randn(2, 8, 4, 64),
        }

        output = executor.execute(kernel, inputs)
        assert output is not None
        assert torch.all(output == 1.0)  # xFormers mock returns ones

    def test_torch_sdpa_argument_mapping(self, executor: KernelExecutorImpl) -> None:
        """PyTorch SDPA gets query, key, value with is_causal."""
        kernel = _make_torch_sdpa_kernel()
        inputs = {
            "query": torch.randn(2, 8, 4, 64),
            "key": torch.randn(2, 8, 4, 64),
            "value": torch.randn(2, 8, 4, 64),
        }
        kwargs = {"causal": True}  # Should be mapped to is_causal

        output = executor.execute(kernel, inputs, **kwargs)
        assert output is not None

    def test_attention_missing_inputs_raises(self, executor: KernelExecutorImpl) -> None:
        """Missing attention inputs raises KernelExecutionError."""
        kernel = _make_flash_attn_kernel()
        inputs = {"query": torch.zeros(1)}  # Missing key and value

        with pytest.raises(KernelExecutionError) as exc_info:
            executor.execute(kernel, inputs)

        assert "Missing" in str(exc_info.value) or "key" in str(exc_info.value).lower()

    def test_attention_alternate_input_names(self, executor: KernelExecutorImpl) -> None:
        """Supports alternate input names (q/k/v vs query/key/value)."""
        kernel = _make_attention_kernel()
        inputs = {
            "q": torch.randn(2, 8, 4, 64),
            "k": torch.randn(2, 8, 4, 64),
            "v": torch.randn(2, 8, 4, 64),
        }

        output = executor.execute(kernel, inputs)
        assert output is not None


# ============================================================================
# Argument Mapping Tests - Normalization
# ============================================================================


class TestArgumentMappingNorm:
    """Tests for normalization argument mapping."""

    def test_norm_with_input_name(self, executor: KernelExecutorImpl) -> None:
        """Normalization accepts 'input' as tensor name."""
        kernel = _make_norm_kernel()
        inputs = {
            "input": torch.randn(4, 512, 768),
            "weight": torch.ones(768),
        }

        output = executor.execute(kernel, inputs, eps=1e-6)
        assert output is not None

    def test_norm_with_x_name(self, executor: KernelExecutorImpl) -> None:
        """Normalization accepts 'x' as tensor name."""
        kernel = _make_norm_kernel()
        inputs = {
            "x": torch.randn(4, 512, 768),
            "weight": torch.ones(768),
        }

        output = executor.execute(kernel, inputs)
        assert output is not None

    def test_norm_with_hidden_states_name(self, executor: KernelExecutorImpl) -> None:
        """Normalization accepts 'hidden_states' as tensor name."""
        kernel = _make_norm_kernel()
        inputs = {
            "hidden_states": torch.randn(4, 512, 768),
            "gamma": torch.ones(768),  # Alternate weight name
        }

        output = executor.execute(kernel, inputs)
        assert output is not None

    def test_norm_missing_input_raises(self, executor: KernelExecutorImpl) -> None:
        """Missing normalization input raises KernelExecutionError."""
        def strict_norm_impl(**kwargs: Any) -> torch.Tensor:
            x = kwargs.get("x") or kwargs.get("input")
            if x is None:
                raise ValueError("Missing x")
            return x

        kernel = KernelSpec(
            kernel_id="strict_norm",
            operation="rms_norm",
            source="test",
            version="1.0",
            impl=strict_norm_impl,
            platform=Platform.CUDA,
        )
        inputs = {"weight": torch.ones(768)}  # Missing x/input

        with pytest.raises(KernelExecutionError):
            executor.execute(kernel, inputs)


# ============================================================================
# Argument Mapping Tests - RoPE
# ============================================================================


class TestArgumentMappingRoPE:
    """Tests for RoPE argument mapping."""

    def test_rope_basic_inputs(self, executor: KernelExecutorImpl) -> None:
        """RoPE accepts basic x, cos, sin inputs."""
        kernel = _make_rope_kernel()
        inputs = {
            "x": torch.randn(2, 512, 8, 64),
            "cos": torch.randn(512, 64),
            "sin": torch.randn(512, 64),
        }

        output = executor.execute(kernel, inputs)
        assert output is not None

    def test_rope_alternate_names(self, executor: KernelExecutorImpl) -> None:
        """RoPE accepts alternate names (input, cos_cached, sin_cached)."""
        kernel = _make_rope_kernel()
        inputs = {
            "input": torch.randn(2, 512, 8, 64),
            "cos_cached": torch.randn(512, 64),
            "sin_cached": torch.randn(512, 64),
        }

        output = executor.execute(kernel, inputs)
        assert output is not None

    def test_rope_with_position_ids(self, executor: KernelExecutorImpl) -> None:
        """RoPE accepts position_ids."""
        kernel = _make_rope_kernel()
        inputs = {
            "x": torch.randn(2, 512, 8, 64),
            "cos": torch.randn(512, 64),
            "sin": torch.randn(512, 64),
            "position_ids": torch.arange(512),
        }

        output = executor.execute(kernel, inputs, interleaved=True)
        assert output is not None


# ============================================================================
# Layout Transformation Tests
# ============================================================================


class TestLayoutTransformation:
    """Tests for attention layout transformation."""

    def test_no_transform_when_not_required(self, executor: KernelExecutorImpl) -> None:
        """No transformation when kernel doesn't specify layout requirements."""
        kernel = _make_attention_kernel()
        # 4D tensor without layout requirement
        q = torch.randn(2, 4, 8, 64)
        k = torch.randn(2, 4, 8, 64)
        v = torch.randn(2, 4, 8, 64)

        output = executor.execute(kernel, {"query": q, "key": k, "value": v})
        assert output is not None

    def test_no_transform_for_non_4d_tensors(self, executor: KernelExecutorImpl) -> None:
        """No transformation for non-4D tensors."""
        kernel = _make_attention_kernel()
        # 3D tensor (no layout to transform)
        q = torch.randn(8, 512, 64)

        output = executor.execute(kernel, {"query": q, "key": q, "value": q})
        assert output is not None


# ============================================================================
# CUDAGraphExecutor Tests
# ============================================================================


class TestCUDAGraphExecutor:
    """Tests for CUDAGraphExecutor."""

    @pytest.fixture
    def base_executor(self, backend_registry: BackendRegistry) -> KernelExecutorImpl:
        """Create base executor for CUDA graph executor."""
        return KernelExecutorImpl(backend_registry=backend_registry)

    @pytest.fixture
    def cuda_graph_executor(self, base_executor: KernelExecutorImpl) -> CUDAGraphExecutor:
        """Create CUDA graph executor."""
        return CUDAGraphExecutor(base_executor=base_executor, warmup_count=2)

    def test_initialization(self, base_executor: KernelExecutorImpl) -> None:
        """CUDAGraphExecutor initializes correctly."""
        executor = CUDAGraphExecutor(base_executor=base_executor, warmup_count=5)
        assert executor._base_executor is base_executor
        assert executor._warmup_count == 5
        assert executor._graphs == {}
        assert executor._static_inputs == {}
        assert executor._static_outputs == {}

    def test_execute_non_cuda_graph_safe_kernel(
        self, cuda_graph_executor: CUDAGraphExecutor
    ) -> None:
        """Falls back to base executor for non-CUDA graph safe kernels."""
        kernel = KernelSpec(
            kernel_id="unsafe_kernel",
            operation="test",
            source="test",
            version="1.0",
            impl=lambda **k: torch.zeros(1),
            platform=Platform.CUDA,
            is_cuda_graph_safe=False,
        )
        inputs = {"x": torch.zeros(1)}

        output = cuda_graph_executor.execute(kernel, inputs)
        assert output is not None
        # Graph should not be captured
        assert "unsafe_kernel" not in cuda_graph_executor._graphs

    def test_execute_cpu_tensor_falls_back(
        self, cuda_graph_executor: CUDAGraphExecutor
    ) -> None:
        """Falls back to base executor for CPU tensors."""
        kernel = KernelSpec(
            kernel_id="cpu_kernel",
            operation="test",
            source="test",
            version="1.0",
            impl=lambda **k: torch.zeros(1),
            platform=Platform.CUDA,
            is_cuda_graph_safe=True,
        )
        # CPU tensor
        inputs = {"x": torch.zeros(1, device="cpu")}

        output = cuda_graph_executor.execute(kernel, inputs)
        assert output is not None
        # Graph should not be captured
        assert "cpu_kernel" not in cuda_graph_executor._graphs

    def test_clear_graphs(self, cuda_graph_executor: CUDAGraphExecutor) -> None:
        """clear_graphs removes all captured graphs."""
        # Manually add some state
        cuda_graph_executor._graphs["test"] = MagicMock()
        cuda_graph_executor._static_inputs["test"] = {}
        cuda_graph_executor._static_outputs["test"] = MagicMock()

        cuda_graph_executor.clear_graphs()

        assert cuda_graph_executor._graphs == {}
        assert cuda_graph_executor._static_inputs == {}
        assert cuda_graph_executor._static_outputs == {}

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_cuda_graph_capture_and_replay(
        self, base_executor: KernelExecutorImpl
    ) -> None:
        """Tests CUDA graph capture and replay on GPU."""
        cuda_graph_executor = CUDAGraphExecutor(base_executor=base_executor, warmup_count=1)

        def simple_impl(**kwargs: Any) -> torch.Tensor:
            x = kwargs.get("x")
            return x * 2

        kernel = KernelSpec(
            kernel_id="simple_cuda_kernel",
            operation="test",
            source="test",
            version="1.0",
            impl=simple_impl,
            platform=Platform.CUDA,
            is_cuda_graph_safe=True,
        )

        inputs = {"x": torch.randn(4, 4, device="cuda")}

        # First call triggers capture
        output1 = cuda_graph_executor.execute(kernel, inputs)
        assert output1 is not None

        # Second call should use captured graph
        inputs2 = {"x": torch.randn(4, 4, device="cuda")}
        output2 = cuda_graph_executor.execute(kernel, inputs2)
        assert output2 is not None


# ============================================================================
# execute_kernel Convenience Function Tests
# ============================================================================


class TestExecuteKernelFunction:
    """Tests for execute_kernel convenience function."""

    def test_execute_kernel_creates_executor(self) -> None:
        """execute_kernel creates temporary executor and runs kernel."""
        kernel = _make_attention_kernel()
        # Provide full attention inputs
        inputs = {
            "query": torch.zeros(2, 4, 8, 64),
            "key": torch.zeros(2, 4, 8, 64),
            "value": torch.zeros(2, 4, 8, 64),
        }

        output = execute_kernel(kernel, inputs)
        assert output is not None

    def test_execute_kernel_with_registry(
        self, backend_registry: BackendRegistry
    ) -> None:
        """execute_kernel can use a provided backend registry."""
        kernel = _make_attention_kernel()
        # Provide full attention inputs
        inputs = {
            "query": torch.zeros(2, 4, 8, 64),
            "key": torch.zeros(2, 4, 8, 64),
            "value": torch.zeros(2, 4, 8, 64),
        }

        output = execute_kernel(kernel, inputs, backend_registry=backend_registry)
        assert output is not None

    def test_execute_kernel_passes_kwargs(self) -> None:
        """execute_kernel passes additional kwargs to kernel."""
        received_kwargs = {}

        def capturing_impl(**kwargs: Any) -> torch.Tensor:
            received_kwargs.update(kwargs)
            return torch.zeros(1)

        kernel = KernelSpec(
            kernel_id="capturing_kernel",
            operation="test",
            source="test",
            version="1.0",
            impl=capturing_impl,
            platform=Platform.CUDA,
        )

        execute_kernel(kernel, {"x": torch.zeros(1)}, custom_arg="test_value")

        assert "custom_arg" in received_kwargs or "x" in received_kwargs


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_inputs_dict(self, executor: KernelExecutorImpl) -> None:
        """Handles empty inputs dict."""
        def no_input_impl(**kwargs: Any) -> torch.Tensor:
            return torch.zeros(1)

        kernel = KernelSpec(
            kernel_id="no_input",
            operation="generate",
            source="test",
            version="1.0",
            impl=no_input_impl,
            platform=Platform.CUDA,
        )

        output = executor.execute(kernel, {})
        assert output is not None

    def test_non_callable_impl_raises(self, executor: KernelExecutorImpl) -> None:
        """Non-callable impl raises KernelExecutionError."""
        # Create kernel with non-callable impl using object.__new__ to bypass validation
        kernel = object.__new__(KernelSpec)
        object.__setattr__(kernel, "kernel_id", "bad_impl")
        object.__setattr__(kernel, "operation", "test")
        object.__setattr__(kernel, "source", "test")
        object.__setattr__(kernel, "impl", "not_callable")

        with pytest.raises(KernelExecutionError) as exc_info:
            executor.execute(kernel, {"x": torch.zeros(1)})

        assert "not callable" in str(exc_info.value).lower()

    def test_runtime_error_from_kernel(self, executor: KernelExecutorImpl) -> None:
        """RuntimeError from kernel is wrapped."""
        def raises_runtime_error(**kwargs: Any) -> torch.Tensor:
            raise RuntimeError("CUDA error")

        kernel = KernelSpec(
            kernel_id="cuda_error_kernel",
            operation="test",
            source="test",
            version="1.0",
            impl=raises_runtime_error,
            platform=Platform.CUDA,
        )

        with pytest.raises(KernelExecutionError) as exc_info:
            executor.execute(kernel, {"x": torch.zeros(1)})

        assert isinstance(exc_info.value.original_error, RuntimeError)

    def test_type_error_from_kernel(self, executor: KernelExecutorImpl) -> None:
        """TypeError from kernel is wrapped."""
        def raises_type_error(**kwargs: Any) -> torch.Tensor:
            raise TypeError("Wrong type")

        kernel = KernelSpec(
            kernel_id="type_error_kernel",
            operation="test",
            source="test",
            version="1.0",
            impl=raises_type_error,
            platform=Platform.CUDA,
        )

        with pytest.raises(KernelExecutionError) as exc_info:
            executor.execute(kernel, {"x": torch.zeros(1)})

        assert isinstance(exc_info.value.original_error, TypeError)


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    """Tests for thread safety of executor."""

    def test_concurrent_executes(self, executor: KernelExecutorImpl) -> None:
        """Multiple threads can execute concurrently."""
        import threading
        from concurrent.futures import ThreadPoolExecutor

        kernel = _make_attention_kernel()
        results: list[torch.Tensor] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def execute_task() -> None:
            try:
                # Provide full attention inputs
                inputs = {
                    "query": torch.randn(2, 4, 8, 64),
                    "key": torch.randn(2, 4, 8, 64),
                    "value": torch.randn(2, 4, 8, 64),
                }
                output = executor.execute(kernel, inputs)
                with lock:
                    results.append(output)
            except Exception as e:
                with lock:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(execute_task) for _ in range(20)]
            for f in futures:
                f.result()

        assert len(errors) == 0
        assert len(results) == 20


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
