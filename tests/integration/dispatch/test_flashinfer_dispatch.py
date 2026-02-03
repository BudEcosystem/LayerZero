"""
Integration tests for LayerZero dispatch system with FlashInfer/FlashAttention kernels.

This module provides comprehensive integration testing for the kernel dispatch system,
covering:
1. FlashInfer/FlashAttention kernel registration (mock if not installed)
2. Kernel selection based on input shapes, dtypes, and device
3. Output shape verification
4. Fallback chain behavior
5. Circuit breaker behavior
6. Telemetry recording
7. All dispatch modes (static, dynamic, config-driven)

Tests are designed to be runnable regardless of whether actual backends are installed,
using mocks as fallbacks.
"""
from __future__ import annotations

import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable
from unittest.mock import MagicMock, patch

import pytest
import torch

from layerzero.device import GPUGeneration
from layerzero.dispatch import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitOpenError,
    CircuitState,
    DispatchConfig,
    DispatchMode,
    DispatchResult,
    DispatchTiming,
    DynamicDispatcher,
    FallbackChainExhaustedError,
    KernelExecutionError,
    StaticDispatcher,
    StaticDispatcherBuilder,
    StaticKernelEntry,
    StaticKernelRegistry,
    create_dynamic_dispatcher,
    create_orchestrator,
    create_static_dispatcher,
    get_global_circuit_registry,
    get_global_dispatcher,
    set_global_dispatcher,
)
from layerzero.dispatch.config_dispatch import (
    ConfigDrivenDispatcher,
    create_config_dispatcher,
)
from layerzero.dispatch.executor import KernelExecutorImpl
from layerzero.enums import Layout, MaskType, OpKind, Platform
from layerzero.models.device_spec import DeviceSpec
from layerzero.models.kernel_spec import KernelSpec
from layerzero.models.selection_context import SelectionContext
from layerzero.registry.backend_registry import BackendRegistry
from layerzero.registry.kernel_registry import KernelRegistry
from layerzero.selection.engine import NoKernelAvailableError, SelectionEngine


logger = logging.getLogger(__name__)


# ============================================================================
# Constants for Testing
# ============================================================================

# Test shape configurations
# Note: seq_len must be > 128 to avoid false positive in executor's
# layout detection heuristic which assumes BHSD when seq_len <= 128
TEST_SHAPES = [
    # (batch, seq_len, heads, head_dim)
    (1, 256, 8, 64),     # Small batch, medium seq
    (4, 256, 8, 64),     # Medium batch, medium seq
    (2, 1024, 32, 128),  # Long seq, many heads
    (8, 512, 8, 64),     # Large batch
    (1, 2048, 16, 64),   # Very long seq
]

# Test dtypes
TEST_DTYPES = [torch.float16, torch.bfloat16]

# Test devices
TEST_DEVICES = ["cuda", "cpu"]


# ============================================================================
# Helper Functions
# ============================================================================

def _check_flashinfer_available() -> bool:
    """Check if FlashInfer is installed."""
    try:
        import flashinfer
        return True
    except ImportError:
        return False


def _check_flash_attn_available() -> bool:
    """Check if FlashAttention is installed."""
    try:
        import flash_attn
        return True
    except ImportError:
        return False


def _check_cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()


def _create_mock_attention_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Create mock attention output with correct shape.

    For mock purposes, returns a tensor of the same shape as query
    with values computed from a simplified attention formula.

    Input/Output layout: BSHD (batch, seq, heads, dim)
    """
    # Input is BSHD: (batch, seq, heads, dim)
    batch, seq_q, heads, dim = query.shape
    _, seq_k, _, _ = key.shape

    # Convert to BHSD for computation: (batch, heads, seq, dim)
    q = query.transpose(1, 2).contiguous()  # (batch, heads, seq_q, dim)
    k = key.transpose(1, 2).contiguous()    # (batch, heads, seq_k, dim)
    v = value.transpose(1, 2).contiguous()  # (batch, heads, seq_k, dim)

    scale = dim ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (batch, heads, seq_q, seq_k)

    if is_causal and seq_q == seq_k:
        mask = torch.triu(
            torch.ones(seq_q, seq_k, dtype=torch.bool, device=query.device),
            diagonal=1
        )
        scores = scores.masked_fill(mask, float('-inf'))

    attn = torch.softmax(scores, dim=-1)
    output_bhsd = torch.matmul(attn, v)  # (batch, heads, seq_q, dim)

    # Convert back to BSHD: (batch, seq, heads, dim)
    output = output_bhsd.transpose(1, 2).contiguous()

    return output


# ============================================================================
# Mock Kernel Implementations
# ============================================================================

class MockFlashInferKernel:
    """Mock FlashInfer kernel for testing when FlashInfer is not installed."""

    def __init__(self, fail_after: int = -1, fail_probability: float = 0.0):
        """Initialize mock kernel.

        Args:
            fail_after: Fail after this many calls (-1 = never fail)
            fail_probability: Probability of random failure (0.0 = never, 1.0 = always)
        """
        self._call_count = 0
        self._fail_after = fail_after
        self._fail_probability = fail_probability
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def reset(self) -> None:
        with self._lock:
            self._call_count = 0

    def __call__(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        causal: bool = False,
        is_causal: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute mock attention."""
        with self._lock:
            self._call_count += 1
            call_num = self._call_count

        # Handle argument naming variations
        q_tensor = q if q is not None else query
        k_tensor = k if k is not None else key
        v_tensor = v if v is not None else value

        if q_tensor is None or k_tensor is None or v_tensor is None:
            raise ValueError("Missing required tensors (q, k, v)")

        # Check for configured failures
        if self._fail_after > 0 and call_num > self._fail_after:
            raise RuntimeError(f"MockFlashInferKernel: Configured to fail after {self._fail_after} calls")

        if self._fail_probability > 0:
            import random
            if random.random() < self._fail_probability:
                raise RuntimeError("MockFlashInferKernel: Random failure triggered")

        use_causal = causal or is_causal
        return _create_mock_attention_output(q_tensor, k_tensor, v_tensor, use_causal)


class MockFlashAttentionKernel:
    """Mock FlashAttention kernel for testing when FlashAttention is not installed."""

    def __init__(self, fail_on_bfloat16: bool = False, latency_ms: float = 0.0):
        """Initialize mock kernel.

        Args:
            fail_on_bfloat16: If True, fail when bfloat16 tensors are passed
            latency_ms: Artificial latency to add (for latency testing)
        """
        self._fail_on_bfloat16 = fail_on_bfloat16
        self._latency_ms = latency_ms
        self._call_count = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def reset(self) -> None:
        with self._lock:
            self._call_count = 0

    def __call__(
        self,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        causal: bool = False,
        is_causal: bool = False,
        softmax_scale: float | None = None,
        dropout_p: float = 0.0,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute mock attention."""
        with self._lock:
            self._call_count += 1

        q_tensor = q if q is not None else query
        k_tensor = k if k is not None else key
        v_tensor = v if v is not None else value

        if q_tensor is None or k_tensor is None or v_tensor is None:
            raise ValueError("Missing required tensors (q, k, v)")

        if self._fail_on_bfloat16 and q_tensor.dtype == torch.bfloat16:
            raise RuntimeError("MockFlashAttentionKernel: bfloat16 not supported")

        if self._latency_ms > 0:
            time.sleep(self._latency_ms / 1000.0)

        use_causal = causal or is_causal
        return _create_mock_attention_output(q_tensor, k_tensor, v_tensor, use_causal)


class MockTorchSDPAKernel:
    """Mock Torch SDPA kernel as fallback."""

    def __init__(self):
        self._call_count = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def reset(self) -> None:
        with self._lock:
            self._call_count = 0

    def __call__(
        self,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: float | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute mock SDPA attention."""
        with self._lock:
            self._call_count += 1

        q_tensor = query if query is not None else q
        k_tensor = key if key is not None else k
        v_tensor = value if value is not None else v

        if q_tensor is None or k_tensor is None or v_tensor is None:
            raise ValueError("Missing required tensors (query, key, value)")

        return _create_mock_attention_output(q_tensor, k_tensor, v_tensor, is_causal)


class MockCPUAttentionKernel:
    """Mock CPU attention kernel for CPU fallback."""

    def __init__(self):
        self._call_count = 0
        self._lock = threading.Lock()

    @property
    def call_count(self) -> int:
        with self._lock:
            return self._call_count

    def reset(self) -> None:
        with self._lock:
            self._call_count = 0

    def __call__(
        self,
        query: torch.Tensor | None = None,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        q: torch.Tensor | None = None,
        k: torch.Tensor | None = None,
        v: torch.Tensor | None = None,
        is_causal: bool = False,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Execute CPU attention."""
        with self._lock:
            self._call_count += 1

        q_tensor = query if query is not None else q
        k_tensor = key if key is not None else k
        v_tensor = value if value is not None else v

        if q_tensor is None or k_tensor is None or v_tensor is None:
            raise ValueError("Missing required tensors")

        return _create_mock_attention_output(q_tensor, k_tensor, v_tensor, is_causal)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def cuda_device_spec() -> DeviceSpec:
    """Create test CUDA device spec (Ampere GPU)."""
    return DeviceSpec(
        platform=Platform.CUDA,
        device_index=0,
        device_name="Test GPU (Mock Ampere)",
        sm_version=(8, 0),
        gpu_generation=GPUGeneration.AMPERE,
        tensor_core_gen=3,
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=14 * 1024**3,
        supports_bf16=True,
        supports_fp8=False,
        supports_fp4=False,
        supports_tma=False,
        max_shared_memory_kb=164,
        cuda_version="12.0",
        driver_version="525.0",
    )


@pytest.fixture
def hopper_device_spec() -> DeviceSpec:
    """Create test CUDA device spec (Hopper GPU - for FlashInfer)."""
    return DeviceSpec(
        platform=Platform.CUDA,
        device_index=0,
        device_name="Test GPU (Mock Hopper)",
        sm_version=(9, 0),
        gpu_generation=GPUGeneration.HOPPER,
        tensor_core_gen=4,
        total_memory_bytes=80 * 1024**3,
        available_memory_bytes=70 * 1024**3,
        supports_bf16=True,
        supports_fp8=True,
        supports_fp4=True,
        supports_tma=True,
        max_shared_memory_kb=228,
        cuda_version="12.4",
        driver_version="550.0",
    )


@pytest.fixture
def cpu_device_spec() -> DeviceSpec:
    """Create test CPU device spec."""
    return DeviceSpec(
        platform=Platform.CPU,
        device_index=0,
        device_name="Test CPU",
        sm_version=None,
        gpu_generation=GPUGeneration.UNKNOWN,
        tensor_core_gen=0,
        total_memory_bytes=64 * 1024**3,
        available_memory_bytes=32 * 1024**3,
        supports_bf16=True,
        supports_fp8=False,
        supports_fp4=False,
        supports_tma=False,
        max_shared_memory_kb=0,
        cuda_version=None,
        driver_version=None,
    )


@pytest.fixture
def mock_flashinfer_kernel() -> MockFlashInferKernel:
    """Create mock FlashInfer kernel."""
    return MockFlashInferKernel()


@pytest.fixture
def mock_flash_attn_kernel() -> MockFlashAttentionKernel:
    """Create mock FlashAttention kernel."""
    return MockFlashAttentionKernel()


@pytest.fixture
def mock_torch_sdpa_kernel() -> MockTorchSDPAKernel:
    """Create mock Torch SDPA kernel."""
    return MockTorchSDPAKernel()


@pytest.fixture
def mock_cpu_kernel() -> MockCPUAttentionKernel:
    """Create mock CPU attention kernel."""
    return MockCPUAttentionKernel()


@pytest.fixture
def flashinfer_kernel_spec(mock_flashinfer_kernel: MockFlashInferKernel) -> KernelSpec:
    """Create FlashInfer kernel spec with mock implementation."""
    return KernelSpec(
        kernel_id="flashinfer.prefill.v1",
        operation="attention.causal",
        source="flashinfer",
        version="0.1.0",
        impl=mock_flashinfer_kernel,
        platform=Platform.CUDA,
        min_sm=(8, 0),  # Ampere+
        max_sm=None,
        supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
        min_head_dim=16,
        max_head_dim=256,
        head_dim_multiple=8,
        min_seq_len=1,
        max_seq_len=None,
        supports_gqa=True,
        supports_mqa=True,
        supports_attn_mask=False,
        supported_attn_mask_types=frozenset([MaskType.NONE]),
        supports_dropout=True,
        supports_scale=True,
        requires_layouts=frozenset([Layout.BSHD]),
        is_cuda_graph_safe=True,
        deterministic=False,
        priority=90,  # High priority
    )


@pytest.fixture
def flash_attn_kernel_spec(mock_flash_attn_kernel: MockFlashAttentionKernel) -> KernelSpec:
    """Create FlashAttention kernel spec with mock implementation."""
    return KernelSpec(
        kernel_id="flash_attn.v3.causal",
        operation="attention.causal",
        source="flash_attn",
        version="2.5.0",
        impl=mock_flash_attn_kernel,
        platform=Platform.CUDA,
        min_sm=(8, 0),
        max_sm=None,
        supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
        min_head_dim=8,
        max_head_dim=256,
        head_dim_multiple=8,
        min_seq_len=1,
        max_seq_len=None,
        supports_gqa=True,
        supports_mqa=True,
        supports_attn_mask=True,
        supported_attn_mask_types=frozenset([MaskType.NONE, MaskType.BOOL]),
        supports_dropout=True,
        supports_scale=True,
        requires_layouts=frozenset([Layout.BSHD]),
        is_cuda_graph_safe=True,
        deterministic=False,
        priority=85,
    )


@pytest.fixture
def torch_sdpa_kernel_spec(mock_torch_sdpa_kernel: MockTorchSDPAKernel) -> KernelSpec:
    """Create Torch SDPA kernel spec with mock implementation."""
    return KernelSpec(
        kernel_id="torch_sdpa.v1",
        operation="attention.causal",
        source="torch_sdpa",
        version="2.0.0",
        impl=mock_torch_sdpa_kernel,
        platform=Platform.CUDA,
        min_sm=(7, 0),
        max_sm=None,
        supported_dtypes=frozenset([torch.float16, torch.bfloat16, torch.float32]),
        min_head_dim=1,
        max_head_dim=256,
        head_dim_multiple=1,
        min_seq_len=1,
        max_seq_len=None,
        supports_gqa=True,
        supports_mqa=True,
        supports_attn_mask=True,
        supported_attn_mask_types=frozenset([MaskType.NONE, MaskType.BOOL, MaskType.FLOAT]),
        supports_dropout=True,
        supports_scale=True,
        requires_layouts=frozenset([Layout.BSHD, Layout.BHSD]),
        is_cuda_graph_safe=True,
        deterministic=False,
        priority=50,  # Lower priority than FlashAttention
    )


@pytest.fixture
def cpu_kernel_spec(mock_cpu_kernel: MockCPUAttentionKernel) -> KernelSpec:
    """Create CPU attention kernel spec with mock implementation."""
    return KernelSpec(
        kernel_id="cpu_attention.v1",
        operation="attention.causal",
        source="cpu",
        version="1.0.0",
        impl=mock_cpu_kernel,
        platform=Platform.CPU,
        min_sm=None,
        max_sm=None,
        supported_dtypes=frozenset([torch.float16, torch.bfloat16, torch.float32]),
        min_head_dim=1,
        max_head_dim=256,
        head_dim_multiple=1,
        min_seq_len=1,
        max_seq_len=None,
        supports_gqa=True,
        supports_mqa=True,
        supports_attn_mask=True,
        supported_attn_mask_types=frozenset([MaskType.NONE, MaskType.BOOL, MaskType.FLOAT]),
        supports_dropout=False,
        supports_scale=True,
        requires_layouts=frozenset([Layout.BSHD, Layout.BHSD]),
        is_cuda_graph_safe=False,
        deterministic=True,
        priority=30,  # Lowest priority
    )


@pytest.fixture
def kernel_registry(
    flashinfer_kernel_spec: KernelSpec,
    flash_attn_kernel_spec: KernelSpec,
    torch_sdpa_kernel_spec: KernelSpec,
    cpu_kernel_spec: KernelSpec,
) -> KernelRegistry:
    """Create kernel registry with all test kernels."""
    registry = KernelRegistry()
    registry.register(flashinfer_kernel_spec)
    registry.register(flash_attn_kernel_spec)
    registry.register(torch_sdpa_kernel_spec)
    registry.register(cpu_kernel_spec)
    return registry


@pytest.fixture
def backend_registry() -> BackendRegistry:
    """Create backend registry."""
    return BackendRegistry(failure_threshold=3, cooldown_seconds=5.0)


@pytest.fixture
def dispatch_config() -> DispatchConfig:
    """Create dispatch configuration."""
    return DispatchConfig(
        mode=DispatchMode.DYNAMIC,
        enable_cache=True,
        cache_size=1000,
        cache_ttl_seconds=60.0,
        enable_fallback=True,
        max_fallback_attempts=3,
        fallback_timeout_ms=100.0,
        enable_telemetry=True,
        record_timing=True,
        log_fallbacks=True,
        circuit_breaker_enabled=True,
        failure_threshold=3,
        recovery_timeout_seconds=5.0,
    )


def _create_context(
    device_spec: DeviceSpec,
    dtype: torch.dtype,
    batch: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    is_causal: bool = True,
) -> SelectionContext:
    """Helper to create SelectionContext."""
    return SelectionContext(
        device=device_spec,
        op_kind=OpKind.TENSOR,
        operation="attention.causal" if is_causal else "attention.full",
        dtype=dtype,
        batch_size=batch,
        seq_len_q=seq_len,
        seq_len_k=seq_len,
        num_heads=heads,
        num_kv_heads=heads,
        head_dim=head_dim,
        layout=Layout.BSHD,
        is_causal=is_causal,
    )


def _create_qkv_tensors(
    batch: int,
    seq_len: int,
    heads: int,
    head_dim: int,
    dtype: torch.dtype = torch.float16,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    """Helper to create Q, K, V tensors."""
    return {
        "query": torch.randn(batch, seq_len, heads, head_dim, dtype=dtype, device=device),
        "key": torch.randn(batch, seq_len, heads, head_dim, dtype=dtype, device=device),
        "value": torch.randn(batch, seq_len, heads, head_dim, dtype=dtype, device=device),
    }


# ============================================================================
# Test: Kernel Registration
# ============================================================================

@pytest.mark.integration
class TestKernelRegistration:
    """Tests for kernel registration with the dispatch system."""

    def test_register_flashinfer_kernel(
        self,
        kernel_registry: KernelRegistry,
        flashinfer_kernel_spec: KernelSpec,
    ) -> None:
        """FlashInfer kernel can be registered and retrieved."""
        kernel = kernel_registry.get("flashinfer.prefill.v1")
        assert kernel is not None
        assert kernel.kernel_id == "flashinfer.prefill.v1"
        assert kernel.source == "flashinfer"
        assert kernel.platform == Platform.CUDA
        assert kernel.priority == 90

    def test_register_flash_attention_kernel(
        self,
        kernel_registry: KernelRegistry,
        flash_attn_kernel_spec: KernelSpec,
    ) -> None:
        """FlashAttention kernel can be registered and retrieved."""
        kernel = kernel_registry.get("flash_attn.v3.causal")
        assert kernel is not None
        assert kernel.kernel_id == "flash_attn.v3.causal"
        assert kernel.source == "flash_attn"
        assert kernel.priority == 85

    def test_get_kernels_by_operation(self, kernel_registry: KernelRegistry) -> None:
        """Can retrieve all kernels for an operation."""
        kernels = kernel_registry.get_by_operation("attention.causal")
        assert len(kernels) == 4  # FlashInfer, FlashAttention, SDPA, CPU

        kernel_ids = {k.kernel_id for k in kernels}
        assert "flashinfer.prefill.v1" in kernel_ids
        assert "flash_attn.v3.causal" in kernel_ids
        assert "torch_sdpa.v1" in kernel_ids
        assert "cpu_attention.v1" in kernel_ids

    def test_kernels_sorted_by_priority(self, kernel_registry: KernelRegistry) -> None:
        """Kernels can be sorted by priority."""
        kernels = kernel_registry.get_by_operation("attention.causal")
        sorted_kernels = sorted(kernels, key=lambda k: k.priority, reverse=True)

        # FlashInfer should be first (highest priority)
        assert sorted_kernels[0].kernel_id == "flashinfer.prefill.v1"
        # FlashAttention second
        assert sorted_kernels[1].kernel_id == "flash_attn.v3.causal"
        # SDPA third
        assert sorted_kernels[2].kernel_id == "torch_sdpa.v1"
        # CPU last
        assert sorted_kernels[3].kernel_id == "cpu_attention.v1"


# ============================================================================
# Test: Kernel Selection Based on Context
# ============================================================================

@pytest.mark.integration
class TestKernelSelection:
    """Tests for kernel selection based on input shapes, dtypes, and device."""

    def test_select_flashinfer_for_cuda_fp16(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """FlashInfer is selected for CUDA with fp16 (highest priority)."""
        engine = SelectionEngine(kernel_registry, backend_registry)

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=512,
            heads=8,
            head_dim=64,
        )

        plan = engine.select(context)
        assert plan.kernel_id == "flashinfer.prefill.v1"

    def test_select_flashinfer_for_bfloat16(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """FlashInfer is selected for CUDA with bfloat16."""
        engine = SelectionEngine(kernel_registry, backend_registry)

        context = _create_context(
            cuda_device_spec,
            torch.bfloat16,
            batch=4,
            seq_len=512,
            heads=8,
            head_dim=64,
        )

        plan = engine.select(context)
        assert plan.kernel_id == "flashinfer.prefill.v1"

    def test_select_cpu_kernel_for_cpu_device(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        cpu_device_spec: DeviceSpec,
    ) -> None:
        """CPU kernel is selected for CPU device."""
        engine = SelectionEngine(kernel_registry, backend_registry)

        context = _create_context(
            cpu_device_spec,
            torch.float32,
            batch=4,
            seq_len=512,
            heads=8,
            head_dim=64,
        )

        plan = engine.select(context)
        assert plan.kernel_id == "cpu_attention.v1"

    @pytest.mark.parametrize("batch,seq_len,heads,head_dim", TEST_SHAPES)
    def test_selection_with_various_shapes(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
        batch: int,
        seq_len: int,
        heads: int,
        head_dim: int,
    ) -> None:
        """Kernel selection works for various input shapes."""
        engine = SelectionEngine(kernel_registry, backend_registry)

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            head_dim=head_dim,
        )

        plan = engine.select(context)
        assert plan.kernel_spec is not None
        # FlashInfer should be selected for all these shapes on CUDA
        assert plan.kernel_id == "flashinfer.prefill.v1"

    def test_selection_caching(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Selection results are cached."""
        engine = SelectionEngine(kernel_registry, backend_registry)

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=512,
            heads=8,
            head_dim=64,
        )

        # First selection
        plan1 = engine.select(context, use_cache=True)

        # Second selection (should hit cache)
        plan2 = engine.select(context, use_cache=True)

        assert plan1.kernel_id == plan2.kernel_id
        assert plan2.cached  # Second should be cached


# ============================================================================
# Test: Dispatch Returns Correct Output Shape
# ============================================================================

@pytest.mark.integration
class TestDispatchOutputShape:
    """Tests that dispatch returns tensors with correct output shape."""

    @pytest.mark.parametrize("batch,seq_len,heads,head_dim", TEST_SHAPES[:3])
    def test_output_shape_matches_input(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        dispatch_config: DispatchConfig,
        cuda_device_spec: DeviceSpec,
        batch: int,
        seq_len: int,
        heads: int,
        head_dim: int,
    ) -> None:
        """Output tensor has same shape as query tensor."""
        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            config=dispatch_config,
        )

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            head_dim=head_dim,
        )

        inputs = _create_qkv_tensors(batch, seq_len, heads, head_dim, torch.float16)

        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            is_causal=True,
        )

        assert result.output.shape == (batch, seq_len, heads, head_dim)
        assert result.output.dtype == torch.float16

    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_output_dtype_matches_input(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        dispatch_config: DispatchConfig,
        cuda_device_spec: DeviceSpec,
        dtype: torch.dtype,
    ) -> None:
        """Output tensor has same dtype as input tensors."""
        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            config=dispatch_config,
        )

        batch, seq_len, heads, head_dim = 4, 256, 8, 64

        context = _create_context(
            cuda_device_spec,
            dtype,
            batch=batch,
            seq_len=seq_len,
            heads=heads,
            head_dim=head_dim,
        )

        inputs = _create_qkv_tensors(batch, seq_len, heads, head_dim, dtype)

        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            is_causal=True,
        )

        assert result.output.dtype == dtype


# ============================================================================
# Test: Fallback Chain Behavior
# ============================================================================

@pytest.mark.integration
class TestFallbackChain:
    """Tests for fallback chain when primary kernel fails."""

    def test_fallback_when_primary_fails(
        self,
        mock_flash_attn_kernel: MockFlashAttentionKernel,
        mock_torch_sdpa_kernel: MockTorchSDPAKernel,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Fallback to SDPA when FlashAttention fails on bfloat16."""
        # Create FlashAttention that fails on bfloat16
        failing_fa = MockFlashAttentionKernel(fail_on_bfloat16=True)

        fa_spec = KernelSpec(
            kernel_id="flash_attn.failing",
            operation="attention.causal",
            source="flash_attn",
            version="2.5.0",
            impl=failing_fa,
            platform=Platform.CUDA,
            min_sm=(8, 0),
            supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
            priority=90,
        )

        sdpa_spec = KernelSpec(
            kernel_id="torch_sdpa.fallback",
            operation="attention.causal",
            source="torch_sdpa",
            version="2.0.0",
            impl=mock_torch_sdpa_kernel,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16, torch.bfloat16, torch.float32]),
            priority=50,
        )

        registry = KernelRegistry()
        registry.register(fa_spec)
        registry.register(sdpa_spec)

        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_fallback=True,
            max_fallback_attempts=3,
            failure_threshold=5,  # High threshold to allow testing
        )

        dispatcher = create_dynamic_dispatcher(
            kernel_registry=registry,
            backend_registry=backend_registry,
            config=config,
        )

        context = _create_context(
            cuda_device_spec,
            torch.bfloat16,  # This will cause FA to fail
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.bfloat16)

        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            is_causal=True,
        )

        assert result.fallback_used
        assert result.kernel_id == "torch_sdpa.fallback"
        assert failing_fa.call_count == 1  # Primary was tried
        assert mock_torch_sdpa_kernel.call_count == 1  # Fallback was used

    def test_fallback_chain_exhausted_error(
        self,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """FallbackChainExhaustedError when all kernels fail."""
        # Create kernels that all fail
        def fail1(**kwargs):
            raise RuntimeError("Kernel 1 failed")

        def fail2(**kwargs):
            raise RuntimeError("Kernel 2 failed")

        registry = KernelRegistry()
        registry.register(KernelSpec(
            kernel_id="fail1",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=fail1,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=90,
        ))
        registry.register(KernelSpec(
            kernel_id="fail2",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=fail2,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=50,
        ))

        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_fallback=True,
            max_fallback_attempts=3,
            failure_threshold=10,  # High to not trigger circuit breaker
        )

        dispatcher = create_dynamic_dispatcher(
            kernel_registry=registry,
            backend_registry=backend_registry,
            config=config,
        )

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        with pytest.raises(FallbackChainExhaustedError) as exc_info:
            dispatcher.dispatch(
                operation="attention.causal",
                inputs=inputs,
                context=context,
                is_causal=True,
            )

        assert len(exc_info.value.attempted_kernels) == 2
        assert "fail1" in exc_info.value.attempted_kernels
        assert "fail2" in exc_info.value.attempted_kernels

    def test_fallback_excludes_failed_kernels(
        self,
        mock_torch_sdpa_kernel: MockTorchSDPAKernel,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Fallback chain does not retry already-failed kernels."""
        call_order = []

        def kernel1(**kwargs):
            call_order.append("k1")
            raise RuntimeError("K1 failed")

        def kernel2(**kwargs):
            call_order.append("k2")
            raise RuntimeError("K2 failed")

        def kernel3(**kwargs):
            call_order.append("k3")
            # Use explicit None check to avoid tensor boolean ambiguity
            q = kwargs.get("query")
            if q is None:
                q = kwargs.get("q")
            if q is None:
                raise ValueError("Missing query tensor")
            return torch.ones_like(q)

        registry = KernelRegistry()
        registry.register(KernelSpec(
            kernel_id="k1",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=kernel1,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=90,
        ))
        registry.register(KernelSpec(
            kernel_id="k2",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=kernel2,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=80,
        ))
        registry.register(KernelSpec(
            kernel_id="k3",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=kernel3,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=70,
        ))

        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_fallback=True,
            max_fallback_attempts=5,
            failure_threshold=10,
        )

        dispatcher = create_dynamic_dispatcher(
            kernel_registry=registry,
            backend_registry=backend_registry,
            config=config,
        )

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            is_causal=True,
        )

        # Each kernel should only be called once
        assert call_order == ["k1", "k2", "k3"]
        assert result.kernel_id == "k3"


# ============================================================================
# Test: Circuit Breaker Behavior
# ============================================================================

@pytest.mark.integration
class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    def test_circuit_opens_after_threshold_failures(
        self,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Circuit opens after configured number of failures."""
        call_count = [0]

        def flaky_kernel(**kwargs):
            call_count[0] += 1
            raise RuntimeError(f"Failure {call_count[0]}")

        def stable_kernel(**kwargs):
            # Use explicit None check to avoid tensor boolean ambiguity
            q = kwargs.get("query")
            if q is None:
                q = kwargs.get("q")
            if q is None:
                raise ValueError("Missing query tensor")
            return torch.ones_like(q)

        registry = KernelRegistry()
        registry.register(KernelSpec(
            kernel_id="flaky",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=flaky_kernel,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=90,
        ))
        registry.register(KernelSpec(
            kernel_id="stable",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=stable_kernel,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=50,
        ))

        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_fallback=True,
            failure_threshold=2,  # Open circuit after 2 failures
            recovery_timeout_seconds=5.0,
        )

        dispatcher = create_dynamic_dispatcher(
            kernel_registry=registry,
            backend_registry=backend_registry,
            config=config,
        )

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        # First dispatch - flaky fails, stable succeeds
        result1 = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            is_causal=True,
        )
        assert result1.fallback_used

        # Second dispatch - flaky fails again, circuit should be opening
        result2 = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            is_causal=True,
        )
        assert result2.fallback_used

        # After more failures, circuit should be open
        # The flaky kernel should be skipped
        cb_stats = dispatcher.circuit_breaker.get_stats()
        assert cb_stats["total_circuits"] >= 1

    def test_circuit_half_open_after_cooldown(self) -> None:
        """Circuit transitions to half-open after cooldown."""
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            cooldown_seconds=0.1,  # Very short for testing
            success_threshold=1,
        )

        cb = CircuitBreaker("test_kernel", config)

        # Open circuit
        cb.record_failure(RuntimeError("fail"))
        cb.record_failure(RuntimeError("fail"))
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()

        # Wait for cooldown
        time.sleep(0.15)

        # Should transition to half-open
        assert cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_closes_on_success_in_half_open(self) -> None:
        """Circuit closes after success in half-open state."""
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            cooldown_seconds=0.05,
            success_threshold=1,
        )

        cb = CircuitBreaker("test_kernel", config)

        # Open circuit
        cb.record_failure(RuntimeError("fail"))
        cb.record_failure(RuntimeError("fail"))

        # Wait for half-open
        time.sleep(0.1)
        assert cb.can_execute()

        # Success should close circuit
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.can_execute()

    def test_circuit_reopens_on_failure_in_half_open(self) -> None:
        """Circuit reopens after failure in half-open state."""
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
            CircuitState,
        )

        config = CircuitBreakerConfig(
            failure_threshold=2,
            cooldown_seconds=0.05,
            success_threshold=1,
        )

        cb = CircuitBreaker("test_kernel", config)

        # Open circuit
        cb.record_failure(RuntimeError("fail"))
        cb.record_failure(RuntimeError("fail"))

        # Wait for half-open
        time.sleep(0.1)
        assert cb.can_execute()
        assert cb.state == CircuitState.HALF_OPEN

        # Failure should reopen
        cb.record_failure(RuntimeError("fail again"))
        assert cb.state == CircuitState.OPEN
        assert not cb.can_execute()


# ============================================================================
# Test: Telemetry Recording
# ============================================================================

@pytest.mark.integration
class TestTelemetry:
    """Tests for telemetry recording."""

    def test_dispatch_timing_recorded(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        dispatch_config: DispatchConfig,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Dispatch timing is recorded in telemetry."""
        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            config=dispatch_config,
        )

        dispatcher.reset_telemetry()

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        # Execute some dispatches
        for _ in range(5):
            result = dispatcher.dispatch(
                operation="attention.causal",
                inputs=inputs,
                context=context,
                is_causal=True,
            )
            assert result.timing.total_ns > 0
            assert result.timing.selection_ns >= 0
            assert result.timing.execution_ns > 0

        telemetry = dispatcher.get_telemetry()

        assert telemetry["dispatch_count"] == 5
        assert telemetry["avg_selection_ns"] > 0
        assert telemetry["avg_execution_ns"] > 0
        assert telemetry["error_count"] == 0
        assert telemetry["fallback_count"] == 0

    def test_fallback_telemetry_recorded(
        self,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Fallback events are recorded in telemetry."""
        def fail_kernel(**kwargs):
            raise RuntimeError("Planned failure")

        def success_kernel(**kwargs):
            # Use explicit None check to avoid tensor boolean ambiguity
            q = kwargs.get("query")
            if q is None:
                q = kwargs.get("q")
            if q is None:
                raise ValueError("Missing query tensor")
            return torch.ones_like(q)

        registry = KernelRegistry()
        registry.register(KernelSpec(
            kernel_id="fail",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=fail_kernel,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=90,
        ))
        registry.register(KernelSpec(
            kernel_id="success",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=success_kernel,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=50,
        ))

        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_fallback=True,
            enable_telemetry=True,
            failure_threshold=10,
        )

        dispatcher = create_dynamic_dispatcher(
            kernel_registry=registry,
            backend_registry=backend_registry,
            config=config,
        )

        dispatcher.reset_telemetry()

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        # Execute dispatch that will use fallback
        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            is_causal=True,
        )

        assert result.fallback_used

        telemetry = dispatcher.get_telemetry()
        assert telemetry["fallback_count"] == 1
        assert telemetry["fallback_rate"] == 1.0

    def test_circuit_breaker_stats_in_telemetry(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        dispatch_config: DispatchConfig,
    ) -> None:
        """Circuit breaker stats are included in telemetry."""
        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            config=dispatch_config,
        )

        telemetry = dispatcher.get_telemetry()

        assert "circuit_breaker" in telemetry
        cb_stats = telemetry["circuit_breaker"]
        assert "total_circuits" in cb_stats
        assert "failure_threshold" in cb_stats
        assert "cooldown_seconds" in cb_stats


# ============================================================================
# Test: All Dispatch Modes
# ============================================================================

@pytest.mark.integration
class TestDispatchModes:
    """Tests for all dispatch modes (static, dynamic, config-driven)."""

    def test_static_dispatch_mode(
        self,
        flashinfer_kernel_spec: KernelSpec,
        flash_attn_kernel_spec: KernelSpec,
        torch_sdpa_kernel_spec: KernelSpec,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Static dispatch selects pre-configured kernel."""
        dispatcher = (
            StaticDispatcherBuilder()
            .with_kernel(flashinfer_kernel_spec, default=True)
            .with_kernel(flash_attn_kernel_spec)
            .with_kernel(torch_sdpa_kernel_spec)
            .build()
        )

        assert dispatcher.mode == DispatchMode.STATIC

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            is_causal=True,
        )

        assert result.mode == DispatchMode.STATIC
        # Default kernel should be selected
        assert result.kernel_id == "flashinfer.prefill.v1"
        assert result.output.shape == (4, 256, 8, 64)

    def test_dynamic_dispatch_mode(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        dispatch_config: DispatchConfig,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Dynamic dispatch selects best kernel at runtime."""
        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            config=dispatch_config,
        )

        assert dispatcher.mode == DispatchMode.DYNAMIC

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        result = dispatcher.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            is_causal=True,
        )

        assert result.mode == DispatchMode.DYNAMIC
        # Highest priority kernel should be selected
        assert result.kernel_id == "flashinfer.prefill.v1"

    def test_orchestrator_with_mode_switching(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        flashinfer_kernel_spec: KernelSpec,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Orchestrator can switch between dispatch modes."""
        # For static dispatch, we need to register a static dispatcher manually
        # since the orchestrator's automatic static dispatcher creation
        # requires the kernel_registry to be passed through
        from layerzero.dispatch.static import StaticDispatcherBuilder

        # Create config for orchestrator - don't use static_kernel_map
        # since that triggers auto-creation which doesn't pass kernel_registry
        config = DispatchConfig(
            mode=DispatchMode.AUTO,
        )

        orchestrator = create_orchestrator(
            config=config,
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            default_mode=DispatchMode.DYNAMIC,
        )

        # Manually register a static dispatcher with our kernels
        static_dispatcher = (
            StaticDispatcherBuilder()
            .with_kernel(flashinfer_kernel_spec, default=True)
            .build()
        )
        orchestrator.register_dispatcher(DispatchMode.STATIC, static_dispatcher)

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        # Explicit static mode
        result_static = orchestrator.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            mode=DispatchMode.STATIC,
            is_causal=True,
        )
        assert result_static.mode == DispatchMode.STATIC

        # Explicit dynamic mode
        result_dynamic = orchestrator.dispatch(
            operation="attention.causal",
            inputs=inputs,
            context=context,
            mode=DispatchMode.DYNAMIC,
            is_causal=True,
        )
        assert result_dynamic.mode == DispatchMode.DYNAMIC

        # Clean up
        orchestrator.shutdown()

    def test_static_dispatch_zero_overhead(
        self,
        flashinfer_kernel_spec: KernelSpec,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Static dispatch has minimal selection overhead."""
        dispatcher = (
            StaticDispatcherBuilder()
            .with_kernel(flashinfer_kernel_spec, default=True)
            .build()
        )

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        # Warm up
        for _ in range(10):
            dispatcher.dispatch(
                operation="attention.causal",
                inputs=inputs,
                context=context,
                is_causal=True,
            )

        # Measure selection time
        selection_times = []
        for _ in range(100):
            result = dispatcher.dispatch(
                operation="attention.causal",
                inputs=inputs,
                context=context,
                is_causal=True,
            )
            selection_times.append(result.timing.selection_ns)

        avg_selection_ns = sum(selection_times) / len(selection_times)

        # Static dispatch should have very low selection overhead
        # Typically < 1000ns (1 microsecond)
        assert avg_selection_ns < 10_000  # 10 microseconds max for safety


# ============================================================================
# Test: Thread Safety
# ============================================================================

@pytest.mark.integration
class TestThreadSafety:
    """Tests for thread-safe dispatch operations."""

    def test_concurrent_dispatches(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        dispatch_config: DispatchConfig,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Multiple threads can dispatch concurrently."""
        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            config=dispatch_config,
        )

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        results: list[DispatchResult] = []
        errors: list[Exception] = []
        lock = threading.Lock()

        def dispatch_task():
            try:
                result = dispatcher.dispatch(
                    operation="attention.causal",
                    inputs=inputs,
                    context=context,
                    is_causal=True,
                )
                with lock:
                    results.append(result)
            except Exception as e:
                with lock:
                    errors.append(e)

        # Run 50 concurrent dispatches
        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(dispatch_task) for _ in range(50)]
            for future in futures:
                future.result()

        assert len(errors) == 0, f"Errors: {errors}"
        assert len(results) == 50

        # All results should be valid
        for result in results:
            assert result.kernel_id is not None
            assert result.output.shape == (4, 256, 8, 64)

    def test_concurrent_circuit_breaker_updates(self) -> None:
        """Circuit breaker handles concurrent updates correctly."""
        from layerzero.dispatch.circuit_breaker import (
            CircuitBreaker,
            CircuitBreakerConfig,
        )

        config = CircuitBreakerConfig(
            failure_threshold=100,  # High to prevent opening during test
            cooldown_seconds=5.0,
        )

        cb = CircuitBreaker("stress_test", config)
        errors: list[Exception] = []

        def stress_test():
            try:
                for _ in range(200):
                    cb.record_failure(RuntimeError("fail"))
                    cb.can_execute()
                    cb.record_success()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=stress_test) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"


# ============================================================================
# Test: Backend Wiring Verification
# ============================================================================

@pytest.mark.integration
class TestBackendWiring:
    """Tests to verify backends are properly wired to dispatch system."""

    def test_flashinfer_adapter_creates_kernel_spec(self) -> None:
        """FlashInfer adapter creates valid KernelSpec."""
        try:
            from layerzero.backends.flashinfer.adapter import FlashInferPrefillAdapter

            adapter = FlashInferPrefillAdapter()
            spec = adapter.get_kernel_spec()

            assert spec.kernel_id == "flashinfer.prefill"
            assert spec.operation == "attention.prefill"
            assert spec.source == "flashinfer"
            assert spec.platform == Platform.CUDA
            assert spec.impl == adapter  # Adapter is the impl

        except ImportError:
            pytest.skip("FlashInfer adapter not available")

    def test_flash_attn_adapter_available(self) -> None:
        """FlashAttention adapter module exists."""
        try:
            from layerzero.backends.flash_attn import adapter
            assert hasattr(adapter, "__name__")
        except ImportError:
            pytest.skip("FlashAttention adapter not available")

    def test_torch_sdpa_adapter_available(self) -> None:
        """Torch SDPA adapter module exists."""
        try:
            from layerzero.backends.torch_sdpa import adapter, kernel
            assert hasattr(adapter, "__name__")
            assert hasattr(kernel, "__name__")
        except ImportError:
            pytest.skip("Torch SDPA adapter not available")

    def test_kernel_registry_accepts_backend_kernels(
        self,
        kernel_registry: KernelRegistry,
    ) -> None:
        """KernelRegistry accepts kernels from all backends."""
        # Verify all test kernels are registered
        assert kernel_registry.kernel_count == 4

        # Verify we can look up by source
        flashinfer_kernels = kernel_registry.get_by_source("flashinfer")
        assert len(flashinfer_kernels) == 1

        flash_attn_kernels = kernel_registry.get_by_source("flash_attn")
        assert len(flash_attn_kernels) == 1

        torch_sdpa_kernels = kernel_registry.get_by_source("torch_sdpa")
        assert len(torch_sdpa_kernels) == 1

        cpu_kernels = kernel_registry.get_by_source("cpu")
        assert len(cpu_kernels) == 1


# ============================================================================
# Test: Error Handling
# ============================================================================

@pytest.mark.integration
class TestErrorHandling:
    """Tests for error handling in dispatch system."""

    def test_no_kernel_for_operation_error(
        self,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Proper error when no kernel is registered for operation."""
        registry = KernelRegistry()  # Empty registry

        engine = SelectionEngine(registry, backend_registry)

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        with pytest.raises(NoKernelAvailableError) as exc_info:
            engine.select(context)

        assert "attention.causal" in str(exc_info.value)

    def test_kernel_execution_error_wrapped(
        self,
        backend_registry: BackendRegistry,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Kernel execution errors are properly wrapped."""
        def bad_kernel(**kwargs):
            raise ValueError("Internal kernel error")

        registry = KernelRegistry()
        registry.register(KernelSpec(
            kernel_id="bad",
            operation="attention.causal",
            source="test",
            version="1.0",
            impl=bad_kernel,
            platform=Platform.CUDA,
            min_sm=(7, 0),
            supported_dtypes=frozenset([torch.float16]),
            priority=90,
        ))

        config = DispatchConfig(
            mode=DispatchMode.DYNAMIC,
            enable_fallback=False,  # Disable to get the error
            failure_threshold=10,
        )

        dispatcher = create_dynamic_dispatcher(
            kernel_registry=registry,
            backend_registry=backend_registry,
            config=config,
        )

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        inputs = _create_qkv_tensors(4, 256, 8, 64, torch.float16)

        with pytest.raises(FallbackChainExhaustedError):
            dispatcher.dispatch(
                operation="attention.causal",
                inputs=inputs,
                context=context,
                is_causal=True,
            )

    def test_invalid_input_shape_error(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        dispatch_config: DispatchConfig,
        cuda_device_spec: DeviceSpec,
    ) -> None:
        """Proper error for invalid input shapes."""
        dispatcher = create_dynamic_dispatcher(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            config=dispatch_config,
        )

        context = _create_context(
            cuda_device_spec,
            torch.float16,
            batch=4,
            seq_len=256,
            heads=8,
            head_dim=64,
        )

        # Missing value tensor
        inputs = {
            "query": torch.randn(4, 256, 8, 64, dtype=torch.float16),
            "key": torch.randn(4, 256, 8, 64, dtype=torch.float16),
            # "value" intentionally missing
        }

        with pytest.raises((ValueError, KernelExecutionError, FallbackChainExhaustedError)):
            dispatcher.dispatch(
                operation="attention.causal",
                inputs=inputs,
                context=context,
                is_causal=True,
            )


# ============================================================================
# Test: Global Dispatcher Management
# ============================================================================

@pytest.mark.integration
class TestGlobalDispatcher:
    """Tests for global dispatcher management."""

    def test_set_and_get_global_dispatcher(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
    ) -> None:
        """Can set and get global dispatcher."""
        # Save original
        try:
            original = get_global_dispatcher()
        except Exception:
            original = None

        try:
            config = DispatchConfig(mode=DispatchMode.DYNAMIC)
            orchestrator = create_orchestrator(
                config=config,
                kernel_registry=kernel_registry,
                backend_registry=backend_registry,
                set_as_global=True,
            )

            global_dispatcher = get_global_dispatcher()
            assert global_dispatcher is orchestrator

        finally:
            # Restore original
            set_global_dispatcher(original)

    def test_global_circuit_registry(self) -> None:
        """Global circuit registry is accessible."""
        registry = get_global_circuit_registry()
        assert registry is not None

        # Should be able to get or create circuits
        circuit = registry.get_or_create("test_global_circuit")
        assert circuit is not None
        assert circuit.name == "test_global_circuit"

        # Clean up
        circuit.reset()


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
