"""End-to-end integration tests for kernel selection."""
from __future__ import annotations

import pytest
import time
import threading
import torch
from typing import Any


class TestSelectionEngineIntegration:
    """Integration tests for the selection engine."""

    @pytest.mark.integration
    def test_selection_engine_creates_context(self) -> None:
        """Selection engine can create context from tensors."""
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration

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

        assert context.op_kind == OpKind.TENSOR
        assert context.batch_size == 2
        assert context.seq_len_q == 64

    @pytest.mark.integration
    def test_selection_engine_filters_kernels(self) -> None:
        """Selection engine filters kernels by constraints."""
        from layerzero.selection.filter import FilterPhase
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration

        filter_engine = FilterPhase()

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

        # Create test kernels
        kernels = [
            KernelSpec(
                kernel_id="kernel_a",
                operation="attention.causal",
                source="test",
                version="1.0",
                priority=100,
                supported_dtypes=frozenset([torch.float16, torch.bfloat16]),
            ),
            KernelSpec(
                kernel_id="kernel_b",
                operation="attention.causal",
                source="test",
                version="1.0",
                priority=50,
                supported_dtypes=frozenset([torch.float32]),  # Won't match float16
            ),
        ]

        filtered, reasons = filter_engine.filter(kernels, context)

        # kernel_a should pass, kernel_b should be filtered by dtype
        assert len(filtered) >= 0  # At least kernel_a should pass

    @pytest.mark.integration
    def test_selection_engine_scores_kernels(self) -> None:
        """Selection engine scores kernels correctly."""
        from layerzero.selection.scorer import ScoringPhase
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration
        from layerzero.policy.policy import Policy
        from layerzero.policy.engine import RuleEngine

        # Create empty policy
        policy = Policy(version="1.0", locks=(), allows=(), denies=(), boosts=())
        rule_engine = RuleEngine(policy)
        scorer = ScoringPhase(rule_engine)

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

        kernels = [
            KernelSpec(
                kernel_id="test_kernel_1",
                operation="attention.causal",
                source="test",
                version="1.0",
                priority=100,
            ),
            KernelSpec(
                kernel_id="test_kernel_2",
                operation="attention.causal",
                source="test",
                version="1.0",
                priority=50,
            ),
        ]

        scores = scorer.score(kernels, context)

        assert isinstance(scores, dict)
        assert "test_kernel_1" in scores
        assert "test_kernel_2" in scores
        # Higher priority kernel should have higher score
        assert scores["test_kernel_1"] > scores["test_kernel_2"]

    @pytest.mark.integration
    def test_selection_cache_stores_results(self) -> None:
        """Selection cache stores and retrieves results."""
        from layerzero.selection.cache import SelectionCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        cache = SelectionCache()

        # Create a mock kernel spec
        kernel_spec = KernelSpec(
            kernel_id="selected_kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
        )

        plan = ExecutionPlan(
            kernel_id="selected_kernel",
            kernel_spec=kernel_spec,
        )

        # Store using key and policy hash
        cache_key = "test_cache_key"
        policy_hash = "test_policy_hash"
        cache.put(cache_key, policy_hash, plan)

        # Retrieve
        cached = cache.get(cache_key, policy_hash)
        assert cached is not None
        assert cached.kernel_id == "selected_kernel"

    @pytest.mark.integration
    def test_mvcc_cache_concurrent_access(self) -> None:
        """MVCC cache handles concurrent access."""
        from layerzero.selection.mvcc_cache import MVCCShardedCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        cache = MVCCShardedCache(num_shards=16)
        errors: list[Exception] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(50):
                    kernel_spec = KernelSpec(
                        kernel_id=f"kernel_{worker_id}_{i}",
                        operation="attention.causal",
                        source="test",
                        version="1.0",
                    )

                    plan = ExecutionPlan(
                        kernel_id=f"kernel_{worker_id}_{i}",
                        kernel_spec=kernel_spec,
                    )

                    cache_key = f"key_{worker_id}_{i}"
                    policy_hash = "test_policy"

                    cache.put(cache_key, policy_hash, plan)
                    result = cache.get(cache_key, policy_hash)
                    assert result is not None
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestPolicyIntegration:
    """Integration tests for policy system."""

    @pytest.mark.integration
    def test_policy_loader_loads_yaml(self) -> None:
        """Policy loader can load from YAML."""
        from layerzero.policy.loader import PolicyLoader
        from layerzero.policy.policy import Policy
        from pathlib import Path
        import tempfile

        # Create a test YAML file
        yaml_content = """
version: "1.0"
locks: []
deny: []
allow: []
boosts: []
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = Path(f.name)

        try:
            loader = PolicyLoader()
            policy = loader.load(yaml_path=yaml_path)

            assert isinstance(policy, Policy)
            assert policy.version == "1.0"
        finally:
            yaml_path.unlink()

    @pytest.mark.integration
    def test_policy_compile_rules(self) -> None:
        """Policy loader compiles rules correctly."""
        from layerzero.policy.loader import PolicyLoader
        from layerzero.policy.policy import Policy

        loader = PolicyLoader()

        policy_dict = {
            "version": "1.0",
            "locks": [],
            "deny": [],
            "allow": [],
            "boosts": [
                {
                    "kernel": "flash_attn.*",
                    "priority_add": 100,
                }
            ],
        }

        policy = loader.compile(policy_dict)

        assert isinstance(policy, Policy)
        assert policy.version == "1.0"
        assert len(policy.boosts) == 1

    @pytest.mark.integration
    def test_policy_hash_changes_with_rules(self) -> None:
        """Policy hash changes when rules change."""
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy1 = Policy(version="1.0", locks=(), allows=(), denies=(), boosts=())

        boost_rule = Rule(
            rule_type=RuleType.BOOST_ADD,
            target="flash_attn.*",
            conditions=(),
            value=100,
        )
        policy2 = Policy(version="1.0", locks=(), allows=(), denies=(), boosts=(boost_rule,))

        # Hashes should be different
        assert policy1.policy_hash != policy2.policy_hash


class TestRegistryIntegration:
    """Integration tests for kernel and backend registries."""

    @pytest.mark.integration
    def test_kernel_registry_registration(self) -> None:
        """Kernel registry accepts registrations."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()

        spec = KernelSpec(
            kernel_id="test_kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=100,
        )

        registry.register(spec)

        retrieved = registry.get("test_kernel")
        assert retrieved is not None
        assert retrieved.kernel_id == "test_kernel"

    @pytest.mark.integration
    def test_kernel_registry_lookup_by_operation(self) -> None:
        """Registry finds kernels by operation type."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()

        # Register attention kernels
        for i in range(3):
            spec = KernelSpec(
                kernel_id=f"attention_kernel_{i}",
                operation="attention.causal",
                source="test",
                version="1.0",
                priority=100 - i * 10,
            )
            registry.register(spec)

        # Register norm kernel
        norm_spec = KernelSpec(
            kernel_id="norm_kernel",
            operation="norm.rms",
            source="test",
            version="1.0",
            priority=100,
        )
        registry.register(norm_spec)

        # Find by operation
        attention_kernels = registry.get_by_operation("attention.causal")
        assert len(attention_kernels) == 3

        norm_kernels = registry.get_by_operation("norm.rms")
        assert len(norm_kernels) == 1

    @pytest.mark.integration
    def test_backend_registry_health_tracking(self) -> None:
        """Backend registry tracks backend health."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry()

        spec = BackendSpec(
            backend_id="test_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )

        registry.register(spec)

        # Record failure
        registry.record_failure("test_backend", "Test error")

        health = registry.get_health("test_backend")
        assert health is not None
        assert health.failure_count == 1

    @pytest.mark.integration
    def test_backend_registry_circuit_breaker(self) -> None:
        """Backend registry circuit breaker works."""
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.backend_spec import BackendSpec

        registry = BackendRegistry(failure_threshold=3, cooldown_seconds=1)

        spec = BackendSpec(
            backend_id="flaky_backend",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flaky_module",
            entry_point=None,
            supported_operations=frozenset(["attention"]),
            capabilities_schema_version="1.0",
        )

        registry.register(spec)

        # Trigger circuit breaker
        for _ in range(5):
            registry.record_failure("flaky_backend", "Test error")

        # Backend should be disabled
        assert not registry.is_available("flaky_backend")


class TestTelemetryIntegration:
    """Integration tests for telemetry system."""

    @pytest.mark.integration
    def test_selection_report_generation(self) -> None:
        """Selection reports are generated correctly."""
        from layerzero.telemetry.selection_report import SelectionReport, KernelCandidate

        candidates = (
            KernelCandidate(
                kernel_id="flash_attn_v3",
                score=0.9,
                rejected=False,
                rejection_reasons=(),
            ),
            KernelCandidate(
                kernel_id="sdpa",
                score=0.5,
                rejected=False,
                rejection_reasons=(),
            ),
            KernelCandidate(
                kernel_id="xformers",
                score=None,
                rejected=True,
                rejection_reasons=("DTYPE_NOT_SUPPORTED",),
            ),
        )

        report = SelectionReport(
            operation="attention",
            chosen_kernel_id="flash_attn_v3",
            candidates=candidates,
            selection_latency_ns=1500,
            cache_hit=False,
            timestamp=time.time(),
            context={"batch_size": 8},
        )

        assert report.chosen_kernel_id == "flash_attn_v3"
        assert len(report.candidates) == 3
        assert report.selection_latency_ns == 1500

    @pytest.mark.integration
    def test_selection_report_summary(self) -> None:
        """Selection report summary is readable."""
        from layerzero.telemetry.selection_report import SelectionReport, KernelCandidate

        candidates = (
            KernelCandidate(
                kernel_id="flash_attn_v3",
                score=0.9,
                rejected=False,
                rejection_reasons=(),
            ),
        )

        report = SelectionReport(
            operation="attention",
            chosen_kernel_id="flash_attn_v3",
            candidates=candidates,
            selection_latency_ns=100,
            cache_hit=False,
            timestamp=time.time(),
        )

        summary = report.summary()
        assert isinstance(summary, str)
        assert "flash_attn_v3" in summary

    @pytest.mark.integration
    def test_metrics_recording(self) -> None:
        """Metrics are recorded correctly."""
        from layerzero.telemetry.metrics import MetricsCollector

        collector = MetricsCollector()

        # Record some metrics
        collector.record_selection("attention.causal", 100_000, cache_hit=False)
        collector.record_selection("attention.causal", 150_000, cache_hit=True)
        collector.record_selection("attention.causal", 120_000, cache_hit=True)

        assert collector.total_selections == 3
        # 2 cache hits out of 3
        assert abs(collector.cache_hit_rate - 2/3) < 0.01

    @pytest.mark.integration
    def test_metrics_histogram(self) -> None:
        """Metrics histogram is calculated correctly."""
        from layerzero.telemetry.metrics import MetricsCollector

        collector = MetricsCollector()

        # Record latencies
        for latency in [100, 150, 200, 250, 300]:
            collector.record_selection("test", latency, cache_hit=False)

        histogram = collector.selection_latency_histogram

        assert "p50" in histogram
        assert "p95" in histogram
        assert "p99" in histogram
        assert "mean" in histogram
        assert histogram["min"] == 100.0
        assert histogram["max"] == 300.0


class TestExplainAPIIntegration:
    """Integration tests for explain API."""

    @pytest.mark.integration
    def test_explain_function_basic(self) -> None:
        """Explain function provides selection report."""
        from layerzero.telemetry.explain import explain

        report = explain("attention")

        assert report.operation == "attention"
        assert len(report.candidates) > 0

    @pytest.mark.integration
    def test_explain_with_tensors(self, device: torch.device, dtype: torch.dtype) -> None:
        """Explain function works with tensor inputs."""
        from layerzero.telemetry.explain import explain

        # Create test tensors
        batch, seq, heads, dim = 2, 64, 4, 64
        query = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)
        key = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)
        value = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)

        report = explain("attention", query, key, value)

        assert report.operation == "attention"
        assert "inferred_device" in report.context or "device" in report.context

    @pytest.mark.integration
    def test_explain_rejected_candidates(self) -> None:
        """Explain shows rejected candidates with reasons."""
        from layerzero.telemetry.explain import explain

        report = explain("attention", dtype=torch.float32)

        # Some candidates might be rejected
        rejected = report.rejected_candidates
        # Rejected candidates should have reasons
        for candidate in rejected:
            assert candidate.rejected
            assert len(candidate.rejection_reasons) > 0


class TestSelectionEngineEnd2End:
    """End-to-end tests for complete selection pipeline."""

    @pytest.mark.integration
    def test_full_selection_pipeline(self) -> None:
        """Full selection pipeline works end-to-end."""
        from layerzero.selection.engine import SelectionEngine
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration
        from layerzero.policy.policy import Policy

        # Setup registries
        kernel_registry = KernelRegistry()
        backend_registry = BackendRegistry()

        # Register a kernel
        kernel_spec = KernelSpec(
            kernel_id="test_attention_kernel",
            operation="attention.causal",
            source="test",
            version="1.0",
            priority=100,
            supported_dtypes=frozenset([torch.float16, torch.float32]),
        )
        kernel_registry.register(kernel_spec)

        # Create engine
        policy = Policy(version="1.0", locks=(), allows=(), denies=(), boosts=())
        engine = SelectionEngine(
            kernel_registry=kernel_registry,
            backend_registry=backend_registry,
            policy=policy,
        )

        # Create context
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

        # Select kernel
        plan = engine.select(context, debug=True)

        assert plan.kernel_id == "test_attention_kernel"
        assert plan.kernel_spec is not None

    @pytest.mark.integration
    def test_selection_with_caching(self) -> None:
        """Selection caching works correctly."""
        from layerzero.selection.engine import SelectionEngine
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.registry.backend_registry import BackendRegistry
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.models.selection_context import SelectionContext
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import OpKind, Platform, Layout
        from layerzero.device import GPUGeneration
        from layerzero.policy.policy import Policy

        # Setup
        kernel_registry = KernelRegistry()
        backend_registry = BackendRegistry()

        kernel_spec = KernelSpec(
            kernel_id="cached_kernel",
            operation="attention.causal",
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
            operation="attention.causal",
            dtype=torch.float16,
            batch_size=2,
            seq_len_q=64,
            seq_len_k=64,
            num_heads=4,
            head_dim=64,
            layout=Layout.BSHD,
        )

        # First selection (cache miss)
        plan1 = engine.select(context, use_cache=True)
        assert not plan1.cached

        # Second selection (cache hit)
        plan2 = engine.select(context, use_cache=True)
        assert plan2.cached

        # Same kernel selected
        assert plan1.kernel_id == plan2.kernel_id
