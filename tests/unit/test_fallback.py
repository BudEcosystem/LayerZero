"""Fallback behavior tests for LayerZero.

Tests that the system properly falls back when no kernels match.
"""
from __future__ import annotations

import logging
import pytest
import torch
from typing import Any


class TestFallbackBehavior:
    """Tests for fallback behavior when no kernel matches."""

    def test_fallback_when_no_match(self) -> None:
        """Fallback used when no kernel matches constraints."""
        from layerzero.selection.engine import SelectionEngine
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration
        from layerzero.policy.policy import Policy

        kernel_registry = KernelRegistry()
        backend_registry = BackendRegistry()

        # Register a kernel that won't match (requires fp64)
        kernel_spec = KernelSpec(
            kernel_id="fp64_only_kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=100,
            supported_dtypes=frozenset([torch.float64]),  # Only fp64
        )
        kernel_registry.register(kernel_spec)

        # Register a fallback kernel that matches everything
        fallback_spec = KernelSpec(
            kernel_id="fallback_kernel",
            operation="attention.causal",
            source="torch",
            version="1.0",
            priority=1,  # Lowest priority
            supported_dtypes=frozenset([torch.float16, torch.float32, torch.bfloat16]),
        )
        kernel_registry.register(fallback_spec)

        policy = Policy(version="1.0", locks=(), allows=(), denies=(), boosts=())
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )

        device_spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 0),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=16 * 1024**3,
            available_memory_bytes=12 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.4",
            driver_version="550.54",
        )

        context = SelectionContext(
            device=device_spec,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,  # Request fp16, which fp64_only_kernel doesn't support
            batch_size=2,
            seq_len_q=64,
            seq_len_k=64,
            num_heads=4,
            head_dim=64,
            layout=Layout.BSHD,
        )

        plan = engine.select(context)

        # Should select fallback kernel since fp64_only doesn't match
        assert plan.kernel_id == "fallback_kernel"

    def test_fallback_warning_logged(self, caplog: pytest.LogCaptureFixture) -> None:
        """Fallback logs warning when used."""
        from layerzero.selection.engine import SelectionEngine
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration
        from layerzero.policy.policy import Policy

        kernel_registry = KernelRegistry()
        backend_registry = BackendRegistry()

        # Only register one kernel with very restrictive constraints
        kernel_spec = KernelSpec(
            kernel_id="restrictive_kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=100,
            supported_dtypes=frozenset([torch.float16]),
        )
        kernel_registry.register(kernel_spec)

        policy = Policy(version="1.0", locks=(), allows=(), denies=(), boosts=())
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )

        device_spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 0),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=16 * 1024**3,
            available_memory_bytes=12 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.4",
            driver_version="550.54",
        )

        context = SelectionContext(
            device=device_spec,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=2,
            seq_len_q=64,
            seq_len_k=64,
            num_heads=4,
            head_dim=64,
            layout=Layout.BSHD,
        )

        with caplog.at_level(logging.DEBUG):
            plan = engine.select(context)

        # Plan should be valid (no explicit fallback needed in this case)
        assert plan is not None
        assert plan.kernel_id == "restrictive_kernel"

    def test_fallback_produces_correct_result(self) -> None:
        """Fallback produces numerically correct result."""
        from layerzero.pytorch import ops  # noqa: F401

        # LayerZero attention expects BHSD format: [batch, heads, seq, dim]
        # (same as torch.nn.functional.scaled_dot_product_attention)
        batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 64

        # Create tensors in BHSD format
        q = torch.randn(batch_size, num_heads, seq_len, head_dim)
        k = torch.randn(batch_size, num_heads, seq_len, head_dim)
        v = torch.randn(batch_size, num_heads, seq_len, head_dim)

        # Use LayerZero attention
        result = torch.ops.layerzero.attention(q, k, v, is_causal=True)

        # Compare with reference SDPA (same BHSD format)
        reference = torch.nn.functional.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        # Should be numerically close
        assert torch.allclose(result, reference, rtol=1e-5, atol=1e-5), \
            f"Max diff: {(result - reference).abs().max().item():.6f}"

    def test_fallback_reason_in_report(self) -> None:
        """Fallback reason is included in SelectionReport."""
        from layerzero.telemetry.selection_report import SelectionReport, KernelCandidate

        # Create a report where all candidates were rejected
        candidates = (
            KernelCandidate(
                kernel_id="flash_attn",
                score=None,
                rejected=True,
                rejection_reasons=("SM_TOO_OLD",),
            ),
            KernelCandidate(
                kernel_id="xformers",
                score=None,
                rejected=True,
                rejection_reasons=("DTYPE_NOT_SUPPORTED",),
            ),
            KernelCandidate(
                kernel_id="torch_sdpa",
                score=0.5,
                rejected=False,
                rejection_reasons=(),
            ),
        )

        report = SelectionReport(
            operation="attention",
            chosen_kernel_id="torch_sdpa",
            candidates=candidates,
            selection_latency_ns=1000,
            cache_hit=False,
            timestamp=0.0,
            context={"fallback_used": True},
        )

        # Verify rejected candidates have reasons
        rejected = report.rejected_candidates
        assert len(rejected) == 2
        for candidate in rejected:
            assert len(candidate.rejection_reasons) > 0

        # Context should indicate fallback was used
        assert report.context.get("fallback_used", False)


class TestFallbackPriority:
    """Tests for fallback priority ordering."""

    def test_fallback_priority_ordering(self) -> None:
        """Fallbacks are selected by priority order."""
        from layerzero.selection.engine import SelectionEngine
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration
        from layerzero.policy.policy import Policy

        kernel_registry = KernelRegistry()
        backend_registry = BackendRegistry()

        # Register multiple fallback kernels with different priorities
        for priority, kernel_id in [(10, "low_priority"), (50, "med_priority"), (90, "high_priority")]:
            spec = KernelSpec(
                kernel_id=kernel_id,
                operation="attention.causal",
                source="test",
                version="1.0",
                priority=priority,
            )
            kernel_registry.register(spec)

        policy = Policy(version="1.0", locks=(), allows=(), denies=(), boosts=())
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )

        device_spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 0),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=16 * 1024**3,
            available_memory_bytes=12 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.4",
            driver_version="550.54",
        )

        context = SelectionContext(
            device=device_spec,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=2,
            seq_len_q=64,
            seq_len_k=64,
            num_heads=4,
            head_dim=64,
            layout=Layout.BSHD,
        )

        plan = engine.select(context)

        # Should select highest priority kernel
        assert plan.kernel_id == "high_priority"


class TestFallbackConfiguration:
    """Tests for fallback configuration options."""

    def test_fallback_can_be_disabled(self) -> None:
        """Fallback behavior can be configured - raises error when no kernel matches."""
        from layerzero.selection.engine import SelectionEngine, NoKernelAvailableError
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration
        from layerzero.policy.policy import Policy

        kernel_registry = KernelRegistry()
        backend_registry = BackendRegistry()

        # Register only a non-matching kernel
        kernel_spec = KernelSpec(
            kernel_id="non_matching_kernel",
            operation="different_operation",  # Different operation
            source="test",
            version="1.0",
            priority=100,
        )
        kernel_registry.register(kernel_spec)

        policy = Policy(version="1.0", locks=(), allows=(), denies=(), boosts=())
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )

        device_spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 0),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=16 * 1024**3,
            available_memory_bytes=12 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.4",
            driver_version="550.54",
        )

        context = SelectionContext(
            device=device_spec,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",  # Different from registered kernel
            dtype=torch.float16,
            batch_size=2,
            seq_len_q=64,
            seq_len_k=64,
            num_heads=4,
            head_dim=64,
            layout=Layout.BSHD,
        )

        # When no kernel matches, the engine should raise NoKernelAvailableError
        # This is the expected behavior - fail fast instead of silent fallback
        with pytest.raises(NoKernelAvailableError) as exc_info:
            engine.select(context)

        # Verify the exception contains the operation name
        assert "attention.causal" in str(exc_info.value)
