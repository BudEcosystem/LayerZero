"""
Shared test fixtures for selection tests.
"""
from __future__ import annotations

import pytest
import torch

from layerzero.device import GPUGeneration
from layerzero.enums import Layout, MaskType, OpKind, Platform
from layerzero.models.device_spec import DeviceSpec
from layerzero.models.kernel_spec import KernelSpec
from layerzero.models.execution_plan import ExecutionPlan
from layerzero.models.selection_context import SelectionContext
from layerzero.policy.policy import Policy


def make_device_spec(
    platform: Platform = Platform.CUDA,
    device_index: int = 0,
    device_name: str = "Test GPU",
    sm_version: tuple[int, int] | None = (8, 0),
    gpu_generation: GPUGeneration = GPUGeneration.AMPERE,
) -> DeviceSpec:
    """Create a test DeviceSpec with sensible defaults."""
    return DeviceSpec(
        platform=platform,
        device_index=device_index,
        device_name=device_name,
        sm_version=sm_version,
        gpu_generation=gpu_generation,
        tensor_core_gen=3 if platform == Platform.CUDA else 0,
        total_memory_bytes=16 * 1024**3,
        available_memory_bytes=12 * 1024**3,
        supports_bf16=True,
        supports_fp8=platform == Platform.CUDA and sm_version and sm_version[0] >= 9,
        supports_fp4=False,
        supports_tma=platform == Platform.CUDA and sm_version and sm_version[0] >= 9,
        max_shared_memory_kb=164 if platform == Platform.CUDA else 0,
        cuda_version="12.4" if platform == Platform.CUDA else None,
        driver_version="550.54" if platform == Platform.CUDA else None,
    )


@pytest.fixture
def device_spec_cuda_sm80() -> DeviceSpec:
    """Create a CUDA SM 8.0 DeviceSpec."""
    return make_device_spec(
        platform=Platform.CUDA,
        device_name="NVIDIA A100",
        sm_version=(8, 0),
        gpu_generation=GPUGeneration.AMPERE,
    )


@pytest.fixture
def device_spec_cuda_sm70() -> DeviceSpec:
    """Create a CUDA SM 7.0 DeviceSpec (Volta generation, pre-Turing)."""
    return make_device_spec(
        platform=Platform.CUDA,
        device_name="NVIDIA V100",
        sm_version=(7, 0),
        gpu_generation=GPUGeneration.UNKNOWN,  # Volta predates Turing enum
    )


@pytest.fixture
def device_spec_cpu() -> DeviceSpec:
    """Create a CPU DeviceSpec."""
    return make_device_spec(
        platform=Platform.CPU,
        device_name="Intel Xeon",
        sm_version=None,
        gpu_generation=GPUGeneration.UNKNOWN,
    )


@pytest.fixture
def device_spec() -> DeviceSpec:
    """Create a default test DeviceSpec (CUDA SM 8.0)."""
    return make_device_spec()


@pytest.fixture
def empty_policy() -> Policy:
    """Create empty policy with no rules."""
    return Policy(
        version="1.0",
        locks=(),
        allows=(),
        denies=(),
        boosts=(),
    )


@pytest.fixture
def flash_kernel() -> KernelSpec:
    """Create a FlashAttention-like kernel spec."""
    return KernelSpec(
        kernel_id="flash_attn.v3.causal",
        operation="attention.causal",
        source="flash_attn",
        version="3.0",
        platform=Platform.CUDA,
        min_sm=(8, 0),
        max_sm=(9, 9),
        supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
        min_head_dim=16,
        max_head_dim=256,
        head_dim_multiple=8,
        max_seq_len=128 * 1024,
        supports_gqa=True,
        requires_layouts=frozenset([Layout.BSHD]),
        is_cuda_graph_safe=True,
        deterministic=False,
        priority=90,
    )


@pytest.fixture
def sdpa_kernel() -> KernelSpec:
    """Create an SDPA-like kernel spec with broader compatibility."""
    return KernelSpec(
        kernel_id="sdpa.default",
        operation="attention.causal",
        source="torch",
        version="2.0",
        platform=Platform.CUDA,
        min_sm=(7, 0),
        max_sm=(9, 9),
        supported_dtypes=frozenset([torch.float16, torch.float32, torch.bfloat16]),
        min_head_dim=1,
        max_head_dim=512,
        head_dim_multiple=1,
        min_seq_len=1,
        max_seq_len=None,
        supports_gqa=True,
        requires_layouts=frozenset([Layout.BSHD, Layout.BHSD]),
        is_cuda_graph_safe=True,
        deterministic=True,
        priority=50,
    )


@pytest.fixture
def cpu_kernel() -> KernelSpec:
    """Create a CPU-only kernel spec."""
    return KernelSpec(
        kernel_id="cpu.attention",
        operation="attention.causal",
        source="torch",
        version="1.0",
        platform=Platform.CPU,
        supported_dtypes=frozenset([torch.float32]),
        priority=30,
    )


@pytest.fixture
def gqa_unsupported_kernel() -> KernelSpec:
    """Create a kernel that doesn't support GQA."""
    return KernelSpec(
        kernel_id="no_gqa.attention",
        operation="attention.causal",
        source="legacy",
        version="1.0",
        platform=Platform.CUDA,
        min_sm=(7, 0),
        supported_dtypes=frozenset([torch.float16]),
        supports_gqa=False,
        priority=40,
    )


@pytest.fixture
def norm_kernel() -> KernelSpec:
    """Create a normalization kernel spec."""
    return KernelSpec(
        kernel_id="triton.rms_norm",
        operation="norm.rms",
        source="triton",
        version="1.0",
        platform=Platform.CUDA,
        min_sm=(8, 0),
        supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
        priority=80,
    )


@pytest.fixture
def high_priority_kernel() -> KernelSpec:
    """Create a high priority kernel."""
    return KernelSpec(
        kernel_id="flash_attn.v3",
        operation="attention.causal",
        source="flash_attn",
        version="3.0",
        platform=Platform.CUDA,
        priority=90,
        transform_cost_hint=0,
    )


@pytest.fixture
def medium_priority_kernel() -> KernelSpec:
    """Create a medium priority kernel."""
    return KernelSpec(
        kernel_id="sdpa.default",
        operation="attention.causal",
        source="torch",
        version="2.0",
        platform=Platform.CUDA,
        priority=50,
        transform_cost_hint=5,
    )


@pytest.fixture
def low_priority_kernel() -> KernelSpec:
    """Create a low priority kernel."""
    return KernelSpec(
        kernel_id="legacy.attention",
        operation="attention.causal",
        source="legacy",
        version="1.0",
        platform=Platform.CUDA,
        priority=20,
        transform_cost_hint=10,
    )


@pytest.fixture
def kernel_spec() -> KernelSpec:
    """Create a generic test KernelSpec."""
    return KernelSpec(
        kernel_id="test.kernel.v1",
        operation="attention.causal",
        source="test",
        version="1.0",
        platform=Platform.CUDA,
        min_sm=(7, 0),
        max_sm=(9, 0),
        supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
        priority=50,
    )


@pytest.fixture
def execution_plan(kernel_spec: KernelSpec) -> ExecutionPlan:
    """Create a test ExecutionPlan."""
    return ExecutionPlan(
        kernel_id=kernel_spec.kernel_id,
        kernel_spec=kernel_spec,
        pre_transforms=(),
        post_transforms=(),
        cached=False,
        cache_key=None,
    )


def make_selection_context(
    device: DeviceSpec,
    operation: str = "attention.causal",
    dtype: torch.dtype = torch.float16,
    head_dim: int = 64,
    seq_len: int = 512,
    layout: Layout = Layout.BSHD,
    enable_gqa: bool = False,
    is_cuda_graph_capturing: bool = False,
    requires_deterministic: bool = False,
    is_contiguous: bool = True,
    stride_last_dim: int = 1,
) -> SelectionContext:
    """Helper to create SelectionContext."""
    return SelectionContext(
        device=device,
        op_kind=OpKind.TENSOR,
        operation=operation,
        dtype=dtype,
        batch_size=2,
        seq_len_q=seq_len,
        seq_len_k=seq_len,
        num_heads=8,
        num_kv_heads=8 if not enable_gqa else 2,
        head_dim=head_dim,
        layout=layout,
        enable_gqa=enable_gqa,
        is_cuda_graph_capturing=is_cuda_graph_capturing,
        requires_deterministic=requires_deterministic,
        is_contiguous=is_contiguous,
        stride_last_dim=stride_last_dim,
    )


@pytest.fixture
def selection_context(device_spec: DeviceSpec) -> SelectionContext:
    """Create a test SelectionContext."""
    return make_selection_context(device_spec)
