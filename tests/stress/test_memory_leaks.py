"""Memory leak tests for LayerZero.

Uses gc and tracemalloc for memory tracking since memray may not be available.
Tests verify bounded memory growth in long-running scenarios.
"""
from __future__ import annotations

import gc
import sys
import tracemalloc
import threading
import pytest
import torch
from typing import Any


def get_memory_mb() -> float:
    """Get current memory usage in MB."""
    gc.collect()
    current, _ = tracemalloc.get_traced_memory()
    return current / (1024 * 1024)


@pytest.fixture(autouse=True)
def memory_tracking():
    """Start memory tracking for each test."""
    tracemalloc.start()
    gc.collect()
    yield
    tracemalloc.stop()


class TestMemoryLeaks:
    """Tests for memory leak detection."""

    @pytest.mark.stress
    def test_no_leak_selection_loop(self) -> None:
        """No memory leak in selection loop."""
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

        # Register a kernel
        kernel_spec = KernelSpec(
            kernel_id="test_kernel",
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

        # Warmup
        for _ in range(10):
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
            _ = engine.select(context)

        gc.collect()
        initial_memory = get_memory_mb()

        # Run selection loop many times
        for i in range(1000):
            context = SelectionContext(
                device=device_spec,
                op_kind=OpKind.TENSOR,
                operation="attention.causal",
                dtype=torch.float16,
                batch_size=2 + (i % 8),
                seq_len_q=64 + (i % 128),
                seq_len_k=64 + (i % 128),
                num_heads=4,
                head_dim=64,
                layout=Layout.BSHD,
            )
            _ = engine.select(context)

        gc.collect()
        final_memory = get_memory_mb()

        # Allow some growth but bound it (e.g., max 10MB growth)
        growth = final_memory - initial_memory
        assert growth < 10.0, f"Memory grew by {growth:.2f}MB, possible leak"

    @pytest.mark.stress
    def test_no_leak_cache_operations(self) -> None:
        """No memory leak in cache operations."""
        from layerzero.selection.cache import SelectionCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        cache = SelectionCache()

        # Warmup
        for i in range(100):
            kernel_spec = KernelSpec(
                kernel_id=f"warmup_{i}",
                operation="attention.causal",
                source="test",
                version="1.0",
            )
            plan = ExecutionPlan(kernel_id=f"warmup_{i}", kernel_spec=kernel_spec)
            cache.put(f"warmup_key_{i}", "policy_v1", plan)

        gc.collect()
        initial_memory = get_memory_mb()

        # Run cache operations many times
        for i in range(5000):
            kernel_spec = KernelSpec(
                kernel_id=f"kernel_{i}",
                operation="attention.causal",
                source="test",
                version="1.0",
            )
            plan = ExecutionPlan(kernel_id=f"kernel_{i}", kernel_spec=kernel_spec)

            # Put and get
            cache.put(f"key_{i % 1000}", f"policy_v{i % 10}", plan)
            _ = cache.get(f"key_{i % 1000}", f"policy_v{i % 10}")

        gc.collect()
        final_memory = get_memory_mb()

        # Cache has bounded size, so memory should be bounded
        growth = final_memory - initial_memory
        assert growth < 50.0, f"Memory grew by {growth:.2f}MB, possible leak"

    @pytest.mark.stress
    def test_no_leak_attention_calls(self) -> None:
        """No memory leak in attention calls."""
        from layerzero.pytorch import ops  # noqa: F401

        batch_size, seq_len, num_heads, head_dim = 2, 64, 4, 64

        # Warmup
        for _ in range(5):
            q = torch.randn(batch_size, seq_len, num_heads, head_dim)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim)
            _ = torch.ops.layerzero.attention(q, k, v)
            del q, k, v

        gc.collect()
        initial_memory = get_memory_mb()

        # Run attention many times
        for _ in range(100):
            q = torch.randn(batch_size, seq_len, num_heads, head_dim)
            k = torch.randn(batch_size, seq_len, num_heads, head_dim)
            v = torch.randn(batch_size, seq_len, num_heads, head_dim)
            result = torch.ops.layerzero.attention(q, k, v)
            del q, k, v, result

        gc.collect()
        final_memory = get_memory_mb()

        growth = final_memory - initial_memory
        assert growth < 20.0, f"Memory grew by {growth:.2f}MB, possible leak"

    @pytest.mark.stress
    def test_bounded_memory_growth(self) -> None:
        """Memory growth is bounded over extended operations."""
        from layerzero.selection.mvcc_cache import MVCCShardedCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        cache = MVCCShardedCache(num_shards=64, max_entries_per_shard=100)

        memory_samples: list[float] = []

        for iteration in range(10):
            # Do many operations
            for i in range(500):
                kernel_spec = KernelSpec(
                    kernel_id=f"kernel_{iteration}_{i}",
                    operation="attention.causal",
                    source="test",
                    version="1.0",
                )
                plan = ExecutionPlan(
                    kernel_id=f"kernel_{iteration}_{i}",
                    kernel_spec=kernel_spec,
                )
                cache.put(f"key_{i % 200}", f"policy_{iteration}", plan)
                _ = cache.get(f"key_{i % 200}", f"policy_{iteration}")

            gc.collect()
            memory_samples.append(get_memory_mb())

        # Memory should stabilize (last samples should be similar)
        if len(memory_samples) >= 5:
            last_5 = memory_samples[-5:]
            variance = max(last_5) - min(last_5)
            assert variance < 5.0, f"Memory not stabilized: variance={variance:.2f}MB"


class TestObjectLifecycle:
    """Tests for proper object cleanup."""

    @pytest.mark.stress
    def test_context_cleanup(self) -> None:
        """SelectionContext objects are properly cleaned up."""
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

        gc.collect()
        initial_objects = len(gc.get_objects())

        # Create and destroy many contexts
        for i in range(1000):
            context = SelectionContext(
                device=device_spec,
                op_kind=OpKind.TENSOR,
                operation="attention.causal",
                dtype=torch.float16,
                batch_size=i % 16 + 1,
                seq_len_q=64,
                seq_len_k=64,
                num_heads=4,
                head_dim=64,
                layout=Layout.BSHD,
            )
            del context

        gc.collect()
        final_objects = len(gc.get_objects())

        # Object count should not grow significantly
        growth = final_objects - initial_objects
        assert growth < 1000, f"Object count grew by {growth}"

    @pytest.mark.stress
    def test_registry_cleanup(self) -> None:
        """Registry entries can be cleaned up."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()

        gc.collect()
        initial_memory = get_memory_mb()

        # Register many kernels
        for i in range(1000):
            spec = KernelSpec(
                kernel_id=f"kernel_{i}",
                operation="attention.causal",
                source="test",
                version="1.0",
                priority=i,
            )
            registry.register(spec)

        # Clear registry
        registry.clear()

        gc.collect()
        final_memory = get_memory_mb()

        # Memory should return close to initial
        growth = final_memory - initial_memory
        assert growth < 5.0, f"Memory not reclaimed after clear: {growth:.2f}MB"


class TestConcurrentMemory:
    """Tests for memory under concurrent access."""

    @pytest.mark.stress
    def test_concurrent_cache_memory(self) -> None:
        """Cache memory is bounded under concurrent access."""
        from layerzero.selection.mvcc_cache import MVCCShardedCache
        from layerzero.models.execution_plan import ExecutionPlan
        from layerzero.models.kernel_spec import KernelSpec

        cache = MVCCShardedCache(num_shards=32, max_entries_per_shard=50)
        errors: list[Exception] = []
        memory_samples: list[float] = []

        def worker(worker_id: int) -> None:
            try:
                for i in range(200):
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
                    cache.put(f"key_{worker_id}_{i % 50}", "policy_v1", plan)
                    _ = cache.get(f"key_{worker_id}_{i % 50}", "policy_v1")
            except Exception as e:
                errors.append(e)

        gc.collect()
        initial_memory = get_memory_mb()

        # Run concurrent workers
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        gc.collect()
        final_memory = get_memory_mb()

        assert len(errors) == 0, f"Errors occurred: {errors}"

        # Memory should be bounded due to cache limits
        growth = final_memory - initial_memory
        assert growth < 30.0, f"Memory grew by {growth:.2f}MB under concurrent load"
