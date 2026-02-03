"""
Tests for SelectionEngine.

TDD tests for the core kernel selection engine implementing
the filter → score → select → cache pipeline.
"""
from __future__ import annotations

import pytest
import torch

from layerzero.enums import Layout, OpKind, Platform
from layerzero.models.device_spec import DeviceSpec
from layerzero.models.kernel_spec import KernelSpec
from layerzero.models.execution_plan import ExecutionPlan, SelectionReport
from layerzero.models.selection_context import SelectionContext
from layerzero.policy.policy import Policy
from layerzero.policy.rule import Rule, RuleType, Condition, ConditionOp
from layerzero.registry.kernel_registry import KernelRegistry
from layerzero.registry.backend_registry import BackendRegistry
from layerzero.selection.engine import SelectionEngine, NoKernelAvailableError
from layerzero.selection.cache import SelectionCache

from .conftest import make_device_spec, make_selection_context


@pytest.fixture
def kernel_registry(
    flash_kernel: KernelSpec,
    sdpa_kernel: KernelSpec,
    cpu_kernel: KernelSpec,
    norm_kernel: KernelSpec,
) -> KernelRegistry:
    """Create kernel registry with test kernels."""
    registry = KernelRegistry()
    registry.register(flash_kernel)
    registry.register(sdpa_kernel)
    registry.register(cpu_kernel)
    registry.register(norm_kernel)
    return registry


@pytest.fixture
def backend_registry() -> BackendRegistry:
    """Create empty backend registry."""
    return BackendRegistry()


class TestSelectionEngineInit:
    """Test SelectionEngine initialization."""

    def test_init_with_registries(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
    ) -> None:
        """Test initialization with registries."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        assert engine is not None

    def test_init_with_policy(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        empty_policy: Policy,
    ) -> None:
        """Test initialization with custom policy."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=empty_policy,
        )
        assert engine.policy is empty_policy

    def test_init_with_cache(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
    ) -> None:
        """Test initialization with custom cache."""
        cache = SelectionCache(max_size=500)
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            cache=cache,
        )
        assert engine.cache is cache

    def test_default_policy_created(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
    ) -> None:
        """Test default empty policy is created if not provided."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        assert engine.policy is not None
        assert engine.policy.is_empty


class TestSelectionEngineSelect:
    """Test basic selection functionality."""

    def test_select_returns_execution_plan(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test select returns an ExecutionPlan."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx)

        assert isinstance(plan, ExecutionPlan)
        assert plan.kernel_id is not None
        assert plan.kernel_spec is not None

    def test_select_highest_priority_kernel(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test selects highest priority compatible kernel."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx)

        # flash_attn has highest priority (90)
        assert plan.kernel_id == "flash_attn.v3.causal"

    def test_select_with_incompatible_dtype(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test selects compatible kernel when dtype limits options."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        # flash_attn doesn't support float32, sdpa does
        ctx = make_selection_context(device_spec_cuda_sm80, dtype=torch.float32)

        plan = engine.select(ctx)

        assert plan.kernel_id == "sdpa.default"

    def test_select_different_operation(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test selects kernel for different operation."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80, operation="norm.rms")

        plan = engine.select(ctx)

        assert plan.kernel_id == "triton.rms_norm"


class TestSelectionEngineFiltering:
    """Test kernel filtering."""

    def test_platform_filtering(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cpu: DeviceSpec,
    ) -> None:
        """Test kernels filtered by platform."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cpu, dtype=torch.float32)

        plan = engine.select(ctx)

        # Only cpu.attention supports CPU platform
        assert plan.kernel_id == "cpu.attention"

    def test_determinism_filtering(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test non-deterministic kernels filtered when determinism required."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80, requires_deterministic=True)

        plan = engine.select(ctx)

        # flash_attn is non-deterministic, sdpa is deterministic
        assert plan.kernel_id == "sdpa.default"


class TestSelectionEnginePolicyLock:
    """Test policy lock functionality."""

    def test_policy_lock_overrides_selection(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test policy lock forces specific kernel."""
        policy = Policy(
            version="1.0",
            locks=(
                Rule(
                    rule_type=RuleType.LOCK,
                    target="attention.causal",  # operation
                    conditions=(),
                    value="sdpa.default",  # force this kernel
                ),
            ),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx)

        # Lock forces sdpa.default even though flash_attn has higher priority
        assert plan.kernel_id == "sdpa.default"

    def test_conditional_policy_lock(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test conditional policy lock."""
        policy = Policy(
            version="1.0",
            locks=(
                Rule(
                    rule_type=RuleType.LOCK,
                    target="attention.causal",
                    conditions=(
                        Condition(field="head_dim", op=ConditionOp.GE, value=128),
                    ),
                    value="sdpa.default",
                ),
            ),
            allows=(),
            denies=(),
            boosts=(),
        )
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )

        # Lock condition not met (head_dim=64)
        ctx1 = make_selection_context(device_spec_cuda_sm80, head_dim=64)
        plan1 = engine.select(ctx1)
        assert plan1.kernel_id == "flash_attn.v3.causal"

        # Lock condition met (head_dim=128)
        ctx2 = make_selection_context(device_spec_cuda_sm80, head_dim=128)
        plan2 = engine.select(ctx2)
        assert plan2.kernel_id == "sdpa.default"


class TestSelectionEnginePolicyDenyAllow:
    """Test policy deny/allow functionality."""

    def test_policy_deny_kernel(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test policy deny removes kernel from selection."""
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(
                Rule(
                    rule_type=RuleType.DENY,
                    target="flash_attn.*",
                    conditions=(),
                    value=None,
                ),
            ),
            boosts=(),
        )
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx)

        # flash_attn denied, falls back to sdpa
        assert plan.kernel_id == "sdpa.default"

    def test_policy_allow_restricts_selection(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test policy allow restricts to allowed kernels only."""
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(
                Rule(
                    rule_type=RuleType.ALLOW,
                    target="sdpa.*",
                    conditions=(),
                    value=None,
                ),
            ),
            denies=(),
            boosts=(),
        )
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx)

        # Only sdpa allowed
        assert plan.kernel_id == "sdpa.default"


class TestSelectionEnginePolicyBoost:
    """Test policy boost functionality."""

    def test_policy_boost_changes_selection(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test policy boost can change which kernel is selected."""
        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(
                Rule(
                    rule_type=RuleType.BOOST_ADD,
                    target="sdpa.*",
                    conditions=(),
                    value=50,  # 50 + 50 = 100 > 90 (flash priority)
                ),
            ),
        )
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx)

        # sdpa boosted above flash_attn
        assert plan.kernel_id == "sdpa.default"


class TestSelectionEngineCaching:
    """Test selection caching."""

    def test_selection_cached(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test selection result is cached."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        # First selection
        plan1 = engine.select(ctx, use_cache=True)
        assert not plan1.cached

        # Second selection - should be from cache
        plan2 = engine.select(ctx, use_cache=True)
        assert plan2.cached
        assert plan2.kernel_id == plan1.kernel_id

    def test_cache_disabled(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test caching can be disabled."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan1 = engine.select(ctx, use_cache=False)
        plan2 = engine.select(ctx, use_cache=False)

        # Neither should be from cache
        assert not plan1.cached
        assert not plan2.cached

    def test_different_context_different_cache_entry(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test different contexts have different cache entries."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx1 = make_selection_context(device_spec_cuda_sm80, head_dim=64)
        ctx2 = make_selection_context(device_spec_cuda_sm80, head_dim=128)

        plan1 = engine.select(ctx1, use_cache=True)
        plan2 = engine.select(ctx2, use_cache=True)

        # Different contexts, neither from cache initially
        assert not plan1.cached
        assert not plan2.cached

        # Now both should be cached
        plan1_cached = engine.select(ctx1, use_cache=True)
        plan2_cached = engine.select(ctx2, use_cache=True)
        assert plan1_cached.cached
        assert plan2_cached.cached


class TestSelectionEngineCacheInvalidation:
    """Test cache invalidation on policy change."""

    def test_policy_change_invalidates_cache(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        empty_policy: Policy,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test changing policy invalidates relevant cache entries."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=empty_policy,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        # Cache a selection
        plan1 = engine.select(ctx, use_cache=True)
        assert plan1.kernel_id == "flash_attn.v3.causal"

        # Update policy
        new_policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(
                Rule(
                    rule_type=RuleType.DENY,
                    target="flash_attn.*",
                    conditions=(),
                    value=None,
                ),
            ),
            boosts=(),
        )
        engine.update_policy(new_policy)

        # Selection should not use old cache
        plan2 = engine.select(ctx, use_cache=True)
        assert not plan2.cached  # Not from cache due to policy change
        assert plan2.kernel_id == "sdpa.default"


class TestSelectionEngineNoKernelAvailable:
    """Test behavior when no kernel is available."""

    def test_no_candidates_raises_error(
        self,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test raises error when no candidates for operation."""
        empty_registry = KernelRegistry()
        engine = SelectionEngine(
            kernel_registry=empty_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        with pytest.raises(NoKernelAvailableError) as exc_info:
            engine.select(ctx)

        assert "attention.causal" in str(exc_info.value)

    def test_all_filtered_raises_error(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test raises error when all candidates filtered out."""
        # Create context that no kernel supports
        ctx = SelectionContext(
            device=device_spec_cuda_sm80,
            op_kind=OpKind.TENSOR,
            operation="attention.causal",
            dtype=torch.float64,  # No kernel supports float64
            batch_size=2,
            seq_len_q=512,
            seq_len_k=512,
            num_heads=8,
            num_kv_heads=8,
            head_dim=64,
            layout=Layout.BSHD,
        )
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )

        with pytest.raises(NoKernelAvailableError):
            engine.select(ctx)


class TestSelectionEngineDebugMode:
    """Test debug mode with SelectionReport."""

    def test_debug_mode_returns_report(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test debug mode includes SelectionReport."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx, debug=True)

        assert plan.debug_info is not None
        assert isinstance(plan.debug_info, SelectionReport)

    def test_debug_report_contains_candidates(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test debug report contains all candidates."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx, debug=True)
        report = plan.debug_info

        assert report is not None
        assert len(report.candidates) > 0
        assert "flash_attn.v3.causal" in report.candidates
        assert "sdpa.default" in report.candidates

    def test_debug_report_contains_filtered_reasons(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test debug report contains filter reasons."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx, debug=True)
        report = plan.debug_info

        assert report is not None
        # cpu.attention should be filtered (wrong platform)
        assert "cpu.attention" in report.filtered_out
        reasons = report.filtered_out["cpu.attention"]
        assert any("PLATFORM" in r.code for r in reasons)

    def test_debug_report_contains_scores(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test debug report contains scores for valid candidates."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx, debug=True)
        report = plan.debug_info

        assert report is not None
        assert len(report.scores) > 0
        assert "flash_attn.v3.causal" in report.scores
        assert "sdpa.default" in report.scores
        # flash has higher score
        assert report.scores["flash_attn.v3.causal"] > report.scores["sdpa.default"]

    def test_debug_report_timing(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test debug report contains timing information."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        plan = engine.select(ctx, debug=True)
        report = plan.debug_info

        assert report is not None
        assert report.selection_time_us >= 0


class TestSelectionEngineThreadSafety:
    """Test thread safety of selection engine."""

    def test_concurrent_selections(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test concurrent selection operations."""
        import threading

        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        errors: list[Exception] = []
        results: list[ExecutionPlan] = []

        def worker(head_dim: int) -> None:
            try:
                ctx = make_selection_context(device_spec_cuda_sm80, head_dim=head_dim)
                for _ in range(20):
                    plan = engine.select(ctx, use_cache=True)
                    results.append(plan)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(dim,))
            for dim in [32, 64, 128, 256]
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 80  # 4 threads * 20 iterations


class TestSelectionEngineSelectBatch:
    """Test batch selection functionality."""

    def test_select_batch(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test selecting kernels for multiple contexts."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        contexts = [
            make_selection_context(device_spec_cuda_sm80, head_dim=64),
            make_selection_context(device_spec_cuda_sm80, head_dim=128),
            make_selection_context(device_spec_cuda_sm80, operation="norm.rms"),
        ]

        plans = engine.select_batch(contexts)

        assert len(plans) == 3
        assert all(isinstance(p, ExecutionPlan) for p in plans)


class TestSelectionEngineCacheStats:
    """Test cache statistics."""

    def test_cache_hit_rate(
        self,
        kernel_registry: KernelRegistry,
        backend_registry: BackendRegistry,
        device_spec_cuda_sm80: DeviceSpec,
    ) -> None:
        """Test cache hit rate tracking."""
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
        )
        ctx = make_selection_context(device_spec_cuda_sm80)

        # First call - miss
        engine.select(ctx, use_cache=True)

        # Second call - hit
        engine.select(ctx, use_cache=True)

        # Third call - hit
        engine.select(ctx, use_cache=True)

        # 2 hits / 3 total
        stats = engine.cache_stats()
        assert stats["hit_rate"] >= 0.5  # At least 50% hit rate
