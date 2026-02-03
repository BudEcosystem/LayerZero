"""
Tests for FilterPhase.

TDD tests for kernel filtering based on compatibility with SelectionContext.
"""
from __future__ import annotations

import pytest
import torch

from layerzero.enums import Layout, Platform
from layerzero.models.device_spec import DeviceSpec
from layerzero.models.kernel_spec import KernelSpec
from layerzero.selection.filter import FilterPhase

from .conftest import make_selection_context


class TestFilterPhaseInit:
    """Test FilterPhase initialization."""

    def test_default_init(self) -> None:
        """Test default initialization."""
        filter_phase = FilterPhase()
        assert filter_phase is not None


class TestFilterPhasePlatform:
    """Test platform filtering."""

    def test_cuda_kernel_passes_cuda_device(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test CUDA kernel passes with CUDA device."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 1
        assert flash_kernel.kernel_id in [k.kernel_id for k in valid]
        assert len(filtered_out) == 0

    def test_cuda_kernel_filtered_on_cpu(
        self,
        flash_kernel: KernelSpec,
        device_spec_cpu: DeviceSpec,
    ) -> None:
        """Test CUDA kernel is filtered out on CPU device."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cpu)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out
        reasons = filtered_out[flash_kernel.kernel_id]
        assert any("PLATFORM" in r.code for r in reasons)

    def test_cpu_kernel_filtered_on_cuda(
        self,
        cpu_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test CPU kernel is filtered out on CUDA device."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80)

        valid, filtered_out = filter_phase.filter([cpu_kernel], ctx)

        assert len(valid) == 0
        assert cpu_kernel.kernel_id in filtered_out


class TestFilterPhaseSMVersion:
    """Test SM version filtering."""

    def test_kernel_passes_within_sm_range(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel passes when SM version is in range."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 1
        assert len(filtered_out) == 0

    def test_kernel_filtered_sm_too_old(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm70: DeviceSpec,
    ) -> None:
        """Test kernel filtered when SM version is too old."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm70)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out
        reasons = filtered_out[flash_kernel.kernel_id]
        assert any("SM_TOO_OLD" in r.code for r in reasons)

    def test_kernel_filtered_sm_too_new(
        self,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered when SM version exceeds max."""
        kernel = KernelSpec(
            kernel_id="old.kernel",
            operation="attention.causal",
            source="legacy",
            version="1.0",
            platform=Platform.CUDA,
            min_sm=(6, 0),
            max_sm=(7, 5),
            supported_dtypes=frozenset([torch.float16]),
            priority=20,
        )

        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80)

        valid, filtered_out = filter_phase.filter([kernel], ctx)

        assert len(valid) == 0
        assert kernel.kernel_id in filtered_out
        reasons = filtered_out[kernel.kernel_id]
        assert any("SM_TOO_NEW" in r.code for r in reasons)


class TestFilterPhaseDtype:
    """Test dtype filtering."""

    def test_kernel_passes_supported_dtype(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel passes with supported dtype."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, dtype=torch.float16)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 1

    def test_kernel_filtered_unsupported_dtype(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered with unsupported dtype."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, dtype=torch.float32)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out
        reasons = filtered_out[flash_kernel.kernel_id]
        assert any("DTYPE" in r.code for r in reasons)


class TestFilterPhaseHeadDim:
    """Test head dimension filtering."""

    def test_kernel_passes_valid_head_dim(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel passes with valid head dimension."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, head_dim=64)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 1

    def test_kernel_filtered_head_dim_too_small(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered when head_dim is too small."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, head_dim=8)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out
        reasons = filtered_out[flash_kernel.kernel_id]
        assert any("HEAD_DIM_TOO_SMALL" in r.code for r in reasons)

    def test_kernel_filtered_head_dim_too_large(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered when head_dim is too large."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, head_dim=512)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out
        reasons = filtered_out[flash_kernel.kernel_id]
        assert any("HEAD_DIM_TOO_LARGE" in r.code for r in reasons)

    def test_kernel_filtered_head_dim_alignment(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered when head_dim not multiple of required."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, head_dim=65)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out
        reasons = filtered_out[flash_kernel.kernel_id]
        assert any("HEAD_DIM_ALIGNMENT" in r.code for r in reasons)


class TestFilterPhaseSeqLen:
    """Test sequence length filtering."""

    def test_kernel_passes_valid_seq_len(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel passes with valid sequence length."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, seq_len=4096)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 1

    def test_kernel_filtered_seq_too_long(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered when seq_len exceeds max."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, seq_len=200 * 1024)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out
        reasons = filtered_out[flash_kernel.kernel_id]
        assert any("SEQ_TOO_LONG" in r.code for r in reasons)


class TestFilterPhaseGQA:
    """Test GQA support filtering."""

    def test_kernel_passes_gqa_supported(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel passes when GQA is enabled and supported."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, enable_gqa=True)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 1

    def test_kernel_filtered_gqa_unsupported(
        self,
        gqa_unsupported_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered when GQA required but not supported."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, enable_gqa=True)

        valid, filtered_out = filter_phase.filter([gqa_unsupported_kernel], ctx)

        assert len(valid) == 0
        assert gqa_unsupported_kernel.kernel_id in filtered_out
        reasons = filtered_out[gqa_unsupported_kernel.kernel_id]
        assert any("GQA" in r.code for r in reasons)


class TestFilterPhaseLayout:
    """Test layout filtering."""

    def test_kernel_passes_supported_layout(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel passes with supported layout."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, layout=Layout.BSHD)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 1

    def test_kernel_filtered_unsupported_layout(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered with unsupported layout."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, layout=Layout.BHSD)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out
        reasons = filtered_out[flash_kernel.kernel_id]
        assert any("LAYOUT" in r.code for r in reasons)


class TestFilterPhaseCudaGraph:
    """Test CUDA graph safety filtering."""

    def test_kernel_passes_cuda_graph_safe(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel passes during CUDA graph capture when safe."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, is_cuda_graph_capturing=True)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 1

    def test_kernel_filtered_cuda_graph_unsafe(
        self,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered during CUDA graph capture when unsafe."""
        unsafe_kernel = KernelSpec(
            kernel_id="unsafe.kernel",
            operation="attention.causal",
            source="custom",
            version="1.0",
            platform=Platform.CUDA,
            min_sm=(8, 0),
            supported_dtypes=frozenset([torch.float16]),
            is_cuda_graph_safe=False,
            priority=60,
        )

        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, is_cuda_graph_capturing=True)

        valid, filtered_out = filter_phase.filter([unsafe_kernel], ctx)

        assert len(valid) == 0
        assert unsafe_kernel.kernel_id in filtered_out
        reasons = filtered_out[unsafe_kernel.kernel_id]
        assert any("CUDA_GRAPH" in r.code for r in reasons)


class TestFilterPhaseDeterminism:
    """Test determinism filtering."""

    def test_kernel_passes_determinism_not_required(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel passes when determinism not required."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, requires_deterministic=False)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 1

    def test_kernel_filtered_non_deterministic(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test non-deterministic kernel filtered when determinism required."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, requires_deterministic=True)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out
        reasons = filtered_out[flash_kernel.kernel_id]
        assert any("DETERMINISTIC" in r.code or "NON_DETERMINISTIC" in r.code for r in reasons)

    def test_deterministic_kernel_passes(
        self,
        sdpa_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test deterministic kernel passes when determinism required."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, requires_deterministic=True)

        valid, filtered_out = filter_phase.filter([sdpa_kernel], ctx)

        assert len(valid) == 1


class TestFilterPhaseContiguity:
    """Test contiguity and stride filtering."""

    def test_kernel_passes_contiguous(
        self,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel passes with contiguous tensor."""
        kernel = KernelSpec(
            kernel_id="contiguous.kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            platform=Platform.CUDA,
            min_sm=(8, 0),
            supported_dtypes=frozenset([torch.float16]),
            requires_contiguous=True,
            priority=50,
        )

        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, is_contiguous=True)

        valid, filtered_out = filter_phase.filter([kernel], ctx)

        assert len(valid) == 1

    def test_kernel_filtered_non_contiguous(
        self,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered with non-contiguous tensor."""
        kernel = KernelSpec(
            kernel_id="contiguous.kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            platform=Platform.CUDA,
            min_sm=(8, 0),
            supported_dtypes=frozenset([torch.float16]),
            requires_contiguous=True,
            priority=50,
        )

        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, is_contiguous=False)

        valid, filtered_out = filter_phase.filter([kernel], ctx)

        assert len(valid) == 0
        assert kernel.kernel_id in filtered_out

    def test_kernel_filtered_bad_stride(
        self,
        flash_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test kernel filtered with bad last dimension stride."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, stride_last_dim=2)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert flash_kernel.kernel_id in filtered_out


class TestFilterPhaseMultipleCandidates:
    """Test filtering with multiple candidates."""

    def test_filter_multiple_candidates(
        self,
        flash_kernel: KernelSpec,
        sdpa_kernel: KernelSpec,
        cpu_kernel: KernelSpec,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test filtering multiple candidates."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80)

        candidates = [flash_kernel, sdpa_kernel, cpu_kernel]
        valid, filtered_out = filter_phase.filter(candidates, ctx)

        assert len(valid) == 2
        valid_ids = [k.kernel_id for k in valid]
        assert flash_kernel.kernel_id in valid_ids
        assert sdpa_kernel.kernel_id in valid_ids
        assert cpu_kernel.kernel_id in filtered_out

    def test_filter_empty_candidates(
        self,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test filtering with empty candidate list."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80)

        valid, filtered_out = filter_phase.filter([], ctx)

        assert len(valid) == 0
        assert len(filtered_out) == 0

    def test_filter_all_filtered_out(
        self,
        flash_kernel: KernelSpec,
        device_spec_cpu: DeviceSpec,
    ) -> None:
        """Test all candidates filtered out."""
        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cpu)

        valid, filtered_out = filter_phase.filter([flash_kernel], ctx)

        assert len(valid) == 0
        assert len(filtered_out) == 1


class TestFilterPhaseMultipleReasons:
    """Test multiple filter reasons."""

    def test_multiple_reasons_reported(
        self,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test that multiple failure reasons are reported."""
        kernel = KernelSpec(
            kernel_id="bad.kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            platform=Platform.CUDA,
            min_sm=(9, 0),
            supported_dtypes=frozenset([torch.float64]),
            max_head_dim=32,
            priority=10,
        )

        filter_phase = FilterPhase()
        ctx = make_selection_context(device_spec_cuda_sm80, head_dim=64, dtype=torch.float16)

        valid, filtered_out = filter_phase.filter([kernel], ctx)

        assert len(valid) == 0
        reasons = filtered_out[kernel.kernel_id]
        assert len(reasons) >= 2
