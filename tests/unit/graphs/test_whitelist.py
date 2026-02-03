"""Tests for CUDA graph whitelist management."""
from __future__ import annotations

import threading
import pytest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

from layerzero.graphs.whitelist import (
    GraphWhitelist,
    DEFAULT_SAFE_KERNELS,
    DEFAULT_UNSAFE_KERNELS,
)


class TestDefaultKernelSets:
    """Tests for default kernel sets."""

    def test_default_safe_kernels_not_empty(self) -> None:
        """DEFAULT_SAFE_KERNELS contains expected operations."""
        assert len(DEFAULT_SAFE_KERNELS) > 0
        assert "attention" in DEFAULT_SAFE_KERNELS
        assert "attention.causal" in DEFAULT_SAFE_KERNELS
        assert "norm.rms" in DEFAULT_SAFE_KERNELS
        assert "matmul" in DEFAULT_SAFE_KERNELS

    def test_default_unsafe_kernels_not_empty(self) -> None:
        """DEFAULT_UNSAFE_KERNELS contains expected operations."""
        assert len(DEFAULT_UNSAFE_KERNELS) > 0
        assert "dynamic_shape_op" in DEFAULT_UNSAFE_KERNELS
        assert "host_to_device" in DEFAULT_UNSAFE_KERNELS
        assert "device_to_host" in DEFAULT_UNSAFE_KERNELS

    def test_no_overlap_between_safe_and_unsafe(self) -> None:
        """Safe and unsafe sets don't overlap."""
        overlap = DEFAULT_SAFE_KERNELS & DEFAULT_UNSAFE_KERNELS
        assert len(overlap) == 0, f"Overlapping kernels: {overlap}"

    def test_sets_are_frozenset(self) -> None:
        """Default sets are immutable frozensets."""
        assert isinstance(DEFAULT_SAFE_KERNELS, frozenset)
        assert isinstance(DEFAULT_UNSAFE_KERNELS, frozenset)


class TestGraphWhitelistInit:
    """Tests for GraphWhitelist initialization."""

    def test_default_initialization(self) -> None:
        """Default initialization uses default kernel sets."""
        whitelist = GraphWhitelist()

        assert whitelist.default_safe is False
        assert len(whitelist) == len(DEFAULT_SAFE_KERNELS)

    def test_custom_safe_kernels(self) -> None:
        """Custom safe kernels override defaults."""
        custom_safe = frozenset(["my_kernel_1", "my_kernel_2"])
        whitelist = GraphWhitelist(safe_kernels=custom_safe)

        assert len(whitelist) == 2
        assert whitelist.is_graph_safe("my_kernel_1")
        assert whitelist.is_graph_safe("my_kernel_2")
        assert not whitelist.is_graph_safe("attention", strict=True)

    def test_custom_unsafe_kernels(self) -> None:
        """Custom unsafe kernels override defaults."""
        custom_unsafe = frozenset(["bad_kernel"])
        whitelist = GraphWhitelist(unsafe_kernels=custom_unsafe)

        assert not whitelist.is_graph_safe("bad_kernel")

    def test_default_safe_true(self) -> None:
        """default_safe=True treats unknown as safe."""
        whitelist = GraphWhitelist(default_safe=True)

        assert whitelist.default_safe is True
        assert whitelist.is_graph_safe("unknown_kernel")

    def test_default_safe_false(self) -> None:
        """default_safe=False treats unknown as unsafe in strict mode."""
        whitelist = GraphWhitelist(default_safe=False)

        assert whitelist.default_safe is False
        assert not whitelist.is_graph_safe("unknown_kernel", strict=True)


class TestGraphWhitelistIsGraphSafe:
    """Tests for is_graph_safe method."""

    def test_safe_kernel_returns_true(self) -> None:
        """Known safe kernel returns True."""
        whitelist = GraphWhitelist()

        assert whitelist.is_graph_safe("attention") is True
        assert whitelist.is_graph_safe("attention.causal") is True
        assert whitelist.is_graph_safe("norm.rms") is True

    def test_unsafe_kernel_returns_false(self) -> None:
        """Known unsafe kernel returns False."""
        whitelist = GraphWhitelist()

        assert whitelist.is_graph_safe("dynamic_shape_op") is False
        assert whitelist.is_graph_safe("host_to_device") is False

    def test_prefix_matching_safe(self) -> None:
        """Prefix matching works for safe kernels."""
        whitelist = GraphWhitelist()

        # Sub-operation should match parent
        assert whitelist.is_graph_safe("attention.causal.prefill") is True
        assert whitelist.is_graph_safe("norm.rms.fast") is True

    def test_prefix_matching_unsafe(self) -> None:
        """Prefix matching works for unsafe kernels."""
        whitelist = GraphWhitelist()

        assert whitelist.is_graph_safe("dynamic_shape_op.resize") is False
        assert whitelist.is_graph_safe("nccl.all_reduce.ring") is False

    def test_unknown_strict_mode_true(self) -> None:
        """Unknown kernel rejected in strict mode."""
        whitelist = GraphWhitelist(default_safe=False)

        assert whitelist.is_graph_safe("totally_unknown", strict=True) is False

    def test_unknown_strict_mode_false(self) -> None:
        """Unknown kernel allowed in non-strict mode."""
        whitelist = GraphWhitelist(default_safe=False)

        assert whitelist.is_graph_safe("totally_unknown", strict=False) is True

    def test_unknown_uses_default_safe(self) -> None:
        """Unknown kernel uses default_safe when strict is None."""
        # default_safe=False
        whitelist_strict = GraphWhitelist(default_safe=False)
        assert whitelist_strict.is_graph_safe("unknown_kernel") is False

        # default_safe=True
        whitelist_lenient = GraphWhitelist(default_safe=True)
        assert whitelist_lenient.is_graph_safe("unknown_kernel") is True


class TestGraphWhitelistKernelSpec:
    """Tests for is_graph_safe_kernel method with KernelSpec."""

    def test_kernel_with_explicit_safe_flag(self) -> None:
        """Kernel with is_cuda_graph_safe=True uses that value."""
        whitelist = GraphWhitelist()

        kernel = MagicMock()
        kernel.is_cuda_graph_safe = True
        kernel.kernel_id = "some_kernel"

        assert whitelist.is_graph_safe_kernel(kernel) is True

    def test_kernel_with_explicit_unsafe_flag(self) -> None:
        """Kernel with is_cuda_graph_safe=False uses that value."""
        whitelist = GraphWhitelist()

        kernel = MagicMock()
        kernel.is_cuda_graph_safe = False
        kernel.kernel_id = "attention"  # Would be safe by whitelist

        assert whitelist.is_graph_safe_kernel(kernel) is False

    def test_kernel_without_flag_uses_whitelist(self) -> None:
        """Kernel with is_cuda_graph_safe=None checks whitelist."""
        whitelist = GraphWhitelist()

        kernel = MagicMock()
        kernel.is_cuda_graph_safe = None
        kernel.kernel_id = "attention"

        assert whitelist.is_graph_safe_kernel(kernel) is True

        kernel.kernel_id = "dynamic_shape_op"
        assert whitelist.is_graph_safe_kernel(kernel) is False


class TestGraphWhitelistModification:
    """Tests for whitelist modification methods."""

    def test_add_safe_kernel(self) -> None:
        """add_safe_kernel adds kernel to safe list."""
        whitelist = GraphWhitelist(safe_kernels=frozenset())

        whitelist.add_safe_kernel("new_kernel")

        assert whitelist.is_graph_safe("new_kernel") is True
        assert "new_kernel" in whitelist.get_safe_kernels()

    def test_add_safe_kernel_removes_from_unsafe(self) -> None:
        """add_safe_kernel removes kernel from unsafe list."""
        whitelist = GraphWhitelist(
            safe_kernels=frozenset(),
            unsafe_kernels=frozenset(["problematic"]),
        )

        assert whitelist.is_graph_safe("problematic") is False

        whitelist.add_safe_kernel("problematic")

        assert whitelist.is_graph_safe("problematic") is True
        assert "problematic" not in whitelist.get_unsafe_kernels()

    def test_remove_safe_kernel(self) -> None:
        """remove_safe_kernel removes kernel from safe list."""
        whitelist = GraphWhitelist()

        whitelist.remove_safe_kernel("attention")

        assert "attention" not in whitelist.get_safe_kernels()

    def test_add_unsafe_kernel(self) -> None:
        """add_unsafe_kernel adds kernel to unsafe list."""
        whitelist = GraphWhitelist(unsafe_kernels=frozenset())

        whitelist.add_unsafe_kernel("bad_kernel")

        assert whitelist.is_graph_safe("bad_kernel") is False
        assert "bad_kernel" in whitelist.get_unsafe_kernels()

    def test_add_unsafe_kernel_removes_from_safe(self) -> None:
        """add_unsafe_kernel removes kernel from safe list."""
        whitelist = GraphWhitelist(
            safe_kernels=frozenset(["attention"]),
            unsafe_kernels=frozenset(),
        )

        whitelist.add_unsafe_kernel("attention")

        assert whitelist.is_graph_safe("attention") is False
        assert "attention" not in whitelist.get_safe_kernels()

    def test_remove_unsafe_kernel(self) -> None:
        """remove_unsafe_kernel removes kernel from unsafe list."""
        whitelist = GraphWhitelist()

        whitelist.remove_unsafe_kernel("dynamic_shape_op")

        assert "dynamic_shape_op" not in whitelist.get_unsafe_kernels()


class TestGraphWhitelistAccessors:
    """Tests for whitelist accessor methods."""

    def test_get_safe_kernels_returns_copy(self) -> None:
        """get_safe_kernels returns frozen copy."""
        whitelist = GraphWhitelist()

        safe = whitelist.get_safe_kernels()

        assert isinstance(safe, frozenset)
        assert len(safe) == len(DEFAULT_SAFE_KERNELS)

    def test_get_unsafe_kernels_returns_copy(self) -> None:
        """get_unsafe_kernels returns frozen copy."""
        whitelist = GraphWhitelist()

        unsafe = whitelist.get_unsafe_kernels()

        assert isinstance(unsafe, frozenset)
        assert len(unsafe) == len(DEFAULT_UNSAFE_KERNELS)

    def test_len_returns_safe_count(self) -> None:
        """__len__ returns number of safe kernels."""
        whitelist = GraphWhitelist(
            safe_kernels=frozenset(["a", "b", "c"]),
        )

        assert len(whitelist) == 3

    def test_contains_checks_safety(self) -> None:
        """__contains__ checks if kernel is graph-safe."""
        whitelist = GraphWhitelist()

        assert "attention" in whitelist
        assert "dynamic_shape_op" not in whitelist


class TestGraphWhitelistThreadSafety:
    """Tests for thread safety of GraphWhitelist."""

    def test_concurrent_reads(self) -> None:
        """Concurrent reads are safe."""
        whitelist = GraphWhitelist()
        results = []

        def read_whitelist():
            for _ in range(100):
                _ = whitelist.is_graph_safe("attention")
                _ = whitelist.get_safe_kernels()
            results.append(True)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(read_whitelist) for _ in range(4)]
            for f in futures:
                f.result()

        assert len(results) == 4

    def test_concurrent_modifications(self) -> None:
        """Concurrent modifications don't corrupt state."""
        whitelist = GraphWhitelist(
            safe_kernels=frozenset(),
            unsafe_kernels=frozenset(),
        )

        initial_count = len(whitelist.get_safe_kernels())

        def add_kernels(thread_id: int):
            for i in range(100):
                whitelist.add_safe_kernel(f"kernel_{thread_id}_{i}")

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(add_kernels, i) for i in range(4)]
            for f in futures:
                f.result()

        # All kernels should be added (plus any initial ones)
        safe = whitelist.get_safe_kernels()
        expected_new_kernels = 400  # 4 threads * 100 kernels
        assert len(safe) == initial_count + expected_new_kernels

    def test_concurrent_read_write(self) -> None:
        """Concurrent reads and writes don't cause issues."""
        whitelist = GraphWhitelist()
        errors = []

        def reader():
            for _ in range(100):
                try:
                    _ = whitelist.is_graph_safe("attention")
                    _ = len(whitelist)
                except Exception as e:
                    errors.append(e)

        def writer():
            for i in range(100):
                try:
                    whitelist.add_safe_kernel(f"new_kernel_{i}")
                except Exception as e:
                    errors.append(e)

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = []
            for _ in range(2):
                futures.append(executor.submit(reader))
                futures.append(executor.submit(writer))
            for f in futures:
                f.result()

        assert len(errors) == 0, f"Errors during concurrent access: {errors}"
