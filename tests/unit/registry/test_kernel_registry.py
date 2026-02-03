"""
Test suite for KernelRegistry.

Tests kernel registration, lookup, and filtering.
Following TDD methodology - tests define expected behavior.
"""
import pytest
import threading
import time
from concurrent.futures import ThreadPoolExecutor


class TestKernelRegistryCreation:
    """Test KernelRegistry construction."""

    def test_kernel_registry_creation(self):
        """KernelRegistry can be created."""
        from layerzero.registry.kernel_registry import KernelRegistry

        registry = KernelRegistry()
        assert registry.kernel_count == 0

    def test_kernel_registry_empty_operations(self):
        """Empty registry has no operations."""
        from layerzero.registry.kernel_registry import KernelRegistry

        registry = KernelRegistry()
        assert registry.operations == frozenset()


class TestKernelRegistration:
    """Test kernel registration."""

    def test_register_kernel(self):
        """Register a kernel successfully."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        spec = KernelSpec(
            kernel_id="flash_attn.v3.causal",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
        )

        registry.register(spec)
        assert registry.kernel_count == 1

    def test_register_duplicate_raises(self):
        """Registering duplicate kernel_id raises ValueError."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        spec = KernelSpec(
            kernel_id="flash_attn.v3.causal",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
        )

        registry.register(spec)
        with pytest.raises(ValueError, match="already registered"):
            registry.register(spec)

    def test_register_many(self):
        """Register multiple kernels atomically."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id="flash_attn.v3.causal",
                operation="attention.causal",
                source="flash_attn",
                version="3.0.0",
            ),
            KernelSpec(
                kernel_id="flash_attn.v3.full",
                operation="attention.full",
                source="flash_attn",
                version="3.0.0",
            ),
            KernelSpec(
                kernel_id="xformers.causal",
                operation="attention.causal",
                source="xformers",
                version="0.0.25",
            ),
        ]

        registry.register_many(specs)
        assert registry.kernel_count == 3


class TestKernelLookup:
    """Test kernel lookup operations."""

    def test_get_by_id(self):
        """Lookup kernel by ID."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        spec = KernelSpec(
            kernel_id="flash_attn.v3.causal",
            operation="attention.causal",
            source="flash_attn",
            version="3.0.0",
        )
        registry.register(spec)

        result = registry.get("flash_attn.v3.causal")
        assert result is not None
        assert result.kernel_id == "flash_attn.v3.causal"

    def test_get_by_id_not_found(self):
        """Lookup non-existent kernel returns None."""
        from layerzero.registry.kernel_registry import KernelRegistry

        registry = KernelRegistry()
        result = registry.get("nonexistent")
        assert result is None

    def test_get_by_operation(self):
        """Lookup kernels by operation type."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id="flash_attn.v3.causal",
                operation="attention.causal",
                source="flash_attn",
                version="3.0.0",
            ),
            KernelSpec(
                kernel_id="xformers.causal",
                operation="attention.causal",
                source="xformers",
                version="0.0.25",
            ),
            KernelSpec(
                kernel_id="flash_attn.v3.full",
                operation="attention.full",
                source="flash_attn",
                version="3.0.0",
            ),
        ]
        registry.register_many(specs)

        results = registry.get_by_operation("attention.causal")
        assert len(results) == 2
        assert all(r.operation == "attention.causal" for r in results)

    def test_get_by_operation_empty(self):
        """Lookup operation with no kernels returns empty list."""
        from layerzero.registry.kernel_registry import KernelRegistry

        registry = KernelRegistry()
        results = registry.get_by_operation("nonexistent")
        assert results == []

    def test_get_by_source(self):
        """Lookup kernels by source library."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id="flash_attn.v3.causal",
                operation="attention.causal",
                source="flash_attn",
                version="3.0.0",
            ),
            KernelSpec(
                kernel_id="flash_attn.v3.full",
                operation="attention.full",
                source="flash_attn",
                version="3.0.0",
            ),
            KernelSpec(
                kernel_id="xformers.causal",
                operation="attention.causal",
                source="xformers",
                version="0.0.25",
            ),
        ]
        registry.register_many(specs)

        results = registry.get_by_source("flash_attn")
        assert len(results) == 2
        assert all(r.source == "flash_attn" for r in results)

    def test_get_all(self):
        """Get all registered kernels."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id="kernel1",
                operation="op1",
                source="src1",
                version="1.0.0",
            ),
            KernelSpec(
                kernel_id="kernel2",
                operation="op2",
                source="src2",
                version="2.0.0",
            ),
        ]
        registry.register_many(specs)

        results = registry.get_all()
        assert len(results) == 2


class TestKernelUnregistration:
    """Test kernel unregistration."""

    def test_unregister_existing(self):
        """Unregister existing kernel returns True."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        spec = KernelSpec(
            kernel_id="test.kernel",
            operation="test.op",
            source="test",
            version="1.0.0",
        )
        registry.register(spec)

        result = registry.unregister("test.kernel")
        assert result is True
        assert registry.kernel_count == 0
        assert registry.get("test.kernel") is None

    def test_unregister_nonexistent(self):
        """Unregister non-existent kernel returns False."""
        from layerzero.registry.kernel_registry import KernelRegistry

        registry = KernelRegistry()
        result = registry.unregister("nonexistent")
        assert result is False


class TestKernelFiltering:
    """Test kernel filtering."""

    def test_filter_by_operation(self):
        """Filter kernels by operation."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id="kernel1",
                operation="attention.causal",
                source="flash_attn",
                version="1.0.0",
            ),
            KernelSpec(
                kernel_id="kernel2",
                operation="norm.rms",
                source="liger",
                version="1.0.0",
            ),
        ]
        registry.register_many(specs)

        results = registry.filter(operation="attention.causal")
        assert len(results) == 1
        assert results[0].kernel_id == "kernel1"

    def test_filter_by_source(self):
        """Filter kernels by source."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id="kernel1",
                operation="op1",
                source="flash_attn",
                version="1.0.0",
            ),
            KernelSpec(
                kernel_id="kernel2",
                operation="op2",
                source="liger",
                version="1.0.0",
            ),
        ]
        registry.register_many(specs)

        results = registry.filter(source="flash_attn")
        assert len(results) == 1
        assert results[0].kernel_id == "kernel1"

    def test_filter_by_platform(self):
        """Filter kernels by platform."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.enums import Platform

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id="cuda_kernel",
                operation="op1",
                source="src1",
                version="1.0.0",
                platform=Platform.CUDA,
            ),
            KernelSpec(
                kernel_id="cpu_kernel",
                operation="op1",
                source="src1",
                version="1.0.0",
                platform=Platform.CPU,
            ),
        ]
        registry.register_many(specs)

        results = registry.filter(platform=Platform.CPU)
        assert len(results) == 1
        assert results[0].kernel_id == "cpu_kernel"

    def test_filter_combined(self):
        """Filter kernels with multiple criteria."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec
        from layerzero.enums import Platform

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id="kernel1",
                operation="attention.causal",
                source="flash_attn",
                version="1.0.0",
                platform=Platform.CUDA,
            ),
            KernelSpec(
                kernel_id="kernel2",
                operation="attention.causal",
                source="xformers",
                version="1.0.0",
                platform=Platform.CUDA,
            ),
            KernelSpec(
                kernel_id="kernel3",
                operation="norm.rms",
                source="flash_attn",
                version="1.0.0",
                platform=Platform.CUDA,
            ),
        ]
        registry.register_many(specs)

        results = registry.filter(
            operation="attention.causal",
            source="flash_attn",
        )
        assert len(results) == 1
        assert results[0].kernel_id == "kernel1"


class TestKernelRegistryOperations:
    """Test registry property access."""

    def test_operations_property(self):
        """operations property returns all registered operations."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id="kernel1",
                operation="attention.causal",
                source="src1",
                version="1.0.0",
            ),
            KernelSpec(
                kernel_id="kernel2",
                operation="attention.full",
                source="src1",
                version="1.0.0",
            ),
            KernelSpec(
                kernel_id="kernel3",
                operation="attention.causal",
                source="src2",
                version="1.0.0",
            ),
        ]
        registry.register_many(specs)

        ops = registry.operations
        assert ops == frozenset(["attention.causal", "attention.full"])

    def test_clear(self):
        """clear() removes all kernels."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        spec = KernelSpec(
            kernel_id="kernel1",
            operation="op1",
            source="src1",
            version="1.0.0",
        )
        registry.register(spec)
        assert registry.kernel_count == 1

        registry.clear()
        assert registry.kernel_count == 0
        assert registry.operations == frozenset()


class TestKernelRegistryThreadSafety:
    """Test thread safety of KernelRegistry."""

    def test_concurrent_registration(self):
        """Concurrent registration is thread-safe."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()

        def register_kernel(i: int):
            spec = KernelSpec(
                kernel_id=f"kernel_{i}",
                operation="op",
                source="src",
                version="1.0.0",
            )
            registry.register(spec)

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(register_kernel, range(100))

        assert registry.kernel_count == 100

    def test_concurrent_lookup(self):
        """Concurrent lookup is thread-safe."""
        from layerzero.registry.kernel_registry import KernelRegistry
        from layerzero.models.kernel_spec import KernelSpec

        registry = KernelRegistry()
        specs = [
            KernelSpec(
                kernel_id=f"kernel_{i}",
                operation="op",
                source="src",
                version="1.0.0",
            )
            for i in range(100)
        ]
        registry.register_many(specs)

        results = []

        def lookup_kernel(i: int):
            result = registry.get(f"kernel_{i}")
            results.append(result is not None)

        with ThreadPoolExecutor(max_workers=10) as executor:
            executor.map(lookup_kernel, range(100))

        assert all(results)
