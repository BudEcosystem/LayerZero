"""Tests for CUDA graph memory tracking."""
from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch

import torch

from layerzero.graphs.memory_tracker import (
    MemoryTracker,
    MemorySnapshot,
    MemoryDelta,
)


class TestMemorySnapshot:
    """Tests for MemorySnapshot dataclass."""

    def test_creation(self) -> None:
        """MemorySnapshot stores all values correctly."""
        snap = MemorySnapshot(
            allocated_bytes=1024 * 1024,
            reserved_bytes=2 * 1024 * 1024,
            max_allocated_bytes=1024 * 1024,
            timestamp_ns=123456789,
            label="test",
        )

        assert snap.allocated_bytes == 1024 * 1024
        assert snap.reserved_bytes == 2 * 1024 * 1024
        assert snap.max_allocated_bytes == 1024 * 1024
        assert snap.timestamp_ns == 123456789
        assert snap.label == "test"

    def test_allocated_mb_property(self) -> None:
        """allocated_mb converts bytes to megabytes."""
        snap = MemorySnapshot(
            allocated_bytes=10 * 1024 * 1024,  # 10MB
            reserved_bytes=0,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )

        assert snap.allocated_mb == 10.0

    def test_reserved_mb_property(self) -> None:
        """reserved_mb converts bytes to megabytes."""
        snap = MemorySnapshot(
            allocated_bytes=0,
            reserved_bytes=5 * 1024 * 1024,  # 5MB
            max_allocated_bytes=0,
            timestamp_ns=0,
        )

        assert snap.reserved_mb == 5.0

    def test_default_label(self) -> None:
        """Default label is empty string."""
        snap = MemorySnapshot(
            allocated_bytes=0,
            reserved_bytes=0,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )

        assert snap.label == ""


class TestMemoryDelta:
    """Tests for MemoryDelta dataclass."""

    def test_creation(self) -> None:
        """MemoryDelta stores before/after snapshots."""
        before = MemorySnapshot(
            allocated_bytes=1024 * 1024,
            reserved_bytes=2 * 1024 * 1024,
            max_allocated_bytes=1024 * 1024,
            timestamp_ns=100,
        )
        after = MemorySnapshot(
            allocated_bytes=2 * 1024 * 1024,
            reserved_bytes=3 * 1024 * 1024,
            max_allocated_bytes=2 * 1024 * 1024,
            timestamp_ns=200,
        )

        delta = MemoryDelta(before=before, after=after, operation="test")

        assert delta.before is before
        assert delta.after is after
        assert delta.operation == "test"

    def test_allocated_delta_bytes(self) -> None:
        """allocated_delta_bytes calculates difference."""
        before = MemorySnapshot(
            allocated_bytes=1024 * 1024,
            reserved_bytes=0,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )
        after = MemorySnapshot(
            allocated_bytes=3 * 1024 * 1024,
            reserved_bytes=0,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )

        delta = MemoryDelta(before=before, after=after)

        assert delta.allocated_delta_bytes == 2 * 1024 * 1024

    def test_allocated_delta_mb(self) -> None:
        """allocated_delta_mb returns megabytes."""
        before = MemorySnapshot(
            allocated_bytes=1024 * 1024,
            reserved_bytes=0,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )
        after = MemorySnapshot(
            allocated_bytes=6 * 1024 * 1024,
            reserved_bytes=0,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )

        delta = MemoryDelta(before=before, after=after)

        assert delta.allocated_delta_mb == 5.0

    def test_reserved_delta_bytes(self) -> None:
        """reserved_delta_bytes calculates difference."""
        before = MemorySnapshot(
            allocated_bytes=0,
            reserved_bytes=2 * 1024 * 1024,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )
        after = MemorySnapshot(
            allocated_bytes=0,
            reserved_bytes=5 * 1024 * 1024,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )

        delta = MemoryDelta(before=before, after=after)

        assert delta.reserved_delta_bytes == 3 * 1024 * 1024

    def test_reserved_delta_mb(self) -> None:
        """reserved_delta_mb returns megabytes."""
        before = MemorySnapshot(
            allocated_bytes=0,
            reserved_bytes=1024 * 1024,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )
        after = MemorySnapshot(
            allocated_bytes=0,
            reserved_bytes=3 * 1024 * 1024,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )

        delta = MemoryDelta(before=before, after=after)

        assert delta.reserved_delta_mb == 2.0

    def test_negative_delta(self) -> None:
        """Negative deltas when memory decreases."""
        before = MemorySnapshot(
            allocated_bytes=5 * 1024 * 1024,
            reserved_bytes=0,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )
        after = MemorySnapshot(
            allocated_bytes=2 * 1024 * 1024,
            reserved_bytes=0,
            max_allocated_bytes=0,
            timestamp_ns=0,
        )

        delta = MemoryDelta(before=before, after=after)

        assert delta.allocated_delta_mb == -3.0


class TestMemoryTrackerInit:
    """Tests for MemoryTracker initialization."""

    def test_default_device(self) -> None:
        """Default device is 0."""
        tracker = MemoryTracker()
        assert tracker.device == 0

    def test_custom_device(self) -> None:
        """Custom device accepted."""
        tracker = MemoryTracker(device=1)
        assert tracker.device == 1

    def test_empty_snapshots(self) -> None:
        """Starts with empty snapshots."""
        tracker = MemoryTracker()
        assert tracker.snapshots == {}

    def test_empty_history(self) -> None:
        """Starts with empty history."""
        tracker = MemoryTracker()
        assert tracker.history == []


class TestMemoryTrackerSnapshot:
    """Tests for snapshot functionality."""

    def test_snapshot_returns_none_without_cuda(self) -> None:
        """snapshot() returns None when CUDA unavailable."""
        with patch("torch.cuda.is_available", return_value=False):
            tracker = MemoryTracker()
            result = tracker.snapshot("test")
            assert result is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024 * 1024)
    @patch("torch.cuda.memory_reserved", return_value=2 * 1024 * 1024)
    @patch("torch.cuda.max_memory_allocated", return_value=1024 * 1024)
    def test_snapshot_captures_memory(
        self,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ) -> None:
        """snapshot() captures current memory state."""
        tracker = MemoryTracker(device=0)
        snap = tracker.snapshot("test_snap")

        assert snap is not None
        assert snap.allocated_bytes == 1024 * 1024
        assert snap.reserved_bytes == 2 * 1024 * 1024
        assert snap.max_allocated_bytes == 1024 * 1024
        assert snap.label == "test_snap"
        assert snap.timestamp_ns > 0

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_snapshot_stored_by_label(
        self,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ) -> None:
        """Labeled snapshots stored for later retrieval."""
        tracker = MemoryTracker()
        tracker.snapshot("before")
        tracker.snapshot("after")

        assert "before" in tracker.snapshots
        assert "after" in tracker.snapshots
        assert len(tracker.snapshots) == 2

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_snapshot_added_to_history(
        self,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ) -> None:
        """All snapshots added to history."""
        tracker = MemoryTracker()
        tracker.snapshot("one")
        tracker.snapshot("")  # Unlabeled
        tracker.snapshot("three")

        assert len(tracker.history) == 3

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_unlabeled_snapshot_not_in_snapshots_dict(
        self,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ) -> None:
        """Unlabeled snapshots not in snapshots dict."""
        tracker = MemoryTracker()
        tracker.snapshot("")  # Unlabeled

        assert "" not in tracker.snapshots
        assert len(tracker.history) == 1


class TestMemoryTrackerGetSnapshot:
    """Tests for get_snapshot method."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_get_existing_snapshot(
        self,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ) -> None:
        """get_snapshot returns existing snapshot."""
        tracker = MemoryTracker()
        tracker.snapshot("test")

        snap = tracker.get_snapshot("test")

        assert snap is not None
        assert snap.label == "test"

    def test_get_nonexistent_snapshot(self) -> None:
        """get_snapshot returns None for missing snapshot."""
        tracker = MemoryTracker()
        snap = tracker.get_snapshot("nonexistent")
        assert snap is None


class TestMemoryTrackerGetDelta:
    """Tests for get_delta method."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=2048)
    def test_get_delta_between_snapshots(self, *mocks) -> None:
        """get_delta calculates delta between two snapshots."""
        tracker = MemoryTracker()

        with patch("torch.cuda.memory_allocated", return_value=1024):
            tracker.snapshot("before")

        with patch("torch.cuda.memory_allocated", return_value=3072):
            tracker.snapshot("after")

        delta = tracker.get_delta("before", "after", operation="test_op")

        assert delta is not None
        assert delta.allocated_delta_bytes == 2048
        assert delta.operation == "test_op"

    def test_get_delta_missing_before(self) -> None:
        """get_delta returns None if before missing."""
        tracker = MemoryTracker()
        delta = tracker.get_delta("missing", "after")
        assert delta is None

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_get_delta_missing_after(
        self,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ) -> None:
        """get_delta returns None if after missing."""
        tracker = MemoryTracker()
        tracker.snapshot("before")

        delta = tracker.get_delta("before", "missing")
        assert delta is None


class TestMemoryTrackerTrackOperation:
    """Tests for track_operation method."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=2048)
    def test_track_operation_returns_result(self, *mocks) -> None:
        """track_operation returns function result."""
        tracker = MemoryTracker()

        with patch("torch.cuda.memory_allocated", return_value=1024):
            result, delta = tracker.track_operation(
                lambda x: x * 2,
                5,
                operation="multiply",
            )

        assert result == 10

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.synchronize")
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=2048)
    def test_track_operation_returns_delta(self, *mocks) -> None:
        """track_operation returns memory delta."""
        tracker = MemoryTracker()

        call_count = [0]

        def tracked_func():
            call_count[0] += 1
            return "done"

        # Simulate memory increase during operation
        memory_values = [1024, 3072]  # Before, after

        def mock_allocated(*args, **kwargs):
            return memory_values[min(call_count[0], 1)]

        with patch("torch.cuda.memory_allocated", side_effect=mock_allocated):
            result, delta = tracker.track_operation(
                tracked_func,
                operation="test_op",
            )

        assert result == "done"
        assert delta is not None
        assert delta.operation == "test_op"

    def test_track_operation_without_cuda(self) -> None:
        """track_operation works without CUDA (delta=None)."""
        with patch("torch.cuda.is_available", return_value=False):
            tracker = MemoryTracker()

            result, delta = tracker.track_operation(
                lambda: 42,
                operation="test",
            )

            assert result == 42
            assert delta is None


class TestMemoryTrackerCheckDelta:
    """Tests for check_memory_delta method."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=2048)
    def test_within_threshold(self, *mocks) -> None:
        """check_memory_delta returns True within threshold."""
        tracker = MemoryTracker()

        with patch("torch.cuda.memory_allocated", return_value=1024 * 1024):
            tracker.snapshot("before")

        with patch("torch.cuda.memory_allocated", return_value=1024 * 1024 + 512 * 1024):  # +0.5MB
            tracker.snapshot("after")

        within, message = tracker.check_memory_delta(
            "before", "after", threshold_mb=1.0
        )

        assert within is True
        assert "within threshold" in message

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=2048)
    def test_exceeds_threshold(self, *mocks) -> None:
        """check_memory_delta returns False when exceeds threshold."""
        tracker = MemoryTracker()

        with patch("torch.cuda.memory_allocated", return_value=1024 * 1024):
            tracker.snapshot("before")

        with patch("torch.cuda.memory_allocated", return_value=10 * 1024 * 1024):  # +9MB
            tracker.snapshot("after")

        within, message = tracker.check_memory_delta(
            "before", "after", threshold_mb=1.0
        )

        assert within is False
        assert "exceeds" in message

    def test_missing_snapshots_returns_true(self) -> None:
        """check_memory_delta returns True if snapshots missing."""
        tracker = MemoryTracker()

        within, message = tracker.check_memory_delta("a", "b", threshold_mb=1.0)

        assert within is True
        assert "No snapshots" in message


class TestMemoryTrackerReset:
    """Tests for reset and clear methods."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_reset_clears_all(
        self,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ) -> None:
        """reset() clears snapshots and history."""
        tracker = MemoryTracker()
        tracker.snapshot("test")

        assert len(tracker.snapshots) > 0
        assert len(tracker.history) > 0

        tracker.reset()

        assert tracker.snapshots == {}
        assert tracker.history == []

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.empty_cache")
    def test_clear_cache_calls_cuda(
        self,
        mock_empty_cache,
        mock_cuda_available,
    ) -> None:
        """clear_cache() calls torch.cuda.empty_cache()."""
        tracker = MemoryTracker()
        tracker.clear_cache()

        mock_empty_cache.assert_called_once()

    def test_clear_cache_noop_without_cuda(self) -> None:
        """clear_cache() is noop without CUDA."""
        with patch("torch.cuda.is_available", return_value=False):
            tracker = MemoryTracker()
            tracker.clear_cache()  # Should not raise

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.reset_max_memory_allocated")
    def test_reset_max_memory_calls_cuda(
        self,
        mock_reset,
        mock_cuda_available,
    ) -> None:
        """reset_max_memory() calls torch.cuda.reset_max_memory_allocated()."""
        tracker = MemoryTracker(device=0)
        tracker.reset_max_memory()

        mock_reset.assert_called_once_with(0)


class TestMemoryTrackerAccessors:
    """Tests for property accessors."""

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_history_returns_copy(
        self,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ) -> None:
        """history property returns copy."""
        tracker = MemoryTracker()
        tracker.snapshot("test")

        history1 = tracker.history
        history2 = tracker.history

        assert history1 is not history2
        assert history1 == history2

    @patch("torch.cuda.is_available", return_value=True)
    @patch("torch.cuda.memory_allocated", return_value=1024)
    @patch("torch.cuda.memory_reserved", return_value=2048)
    @patch("torch.cuda.max_memory_allocated", return_value=1024)
    def test_snapshots_returns_copy(
        self,
        mock_max_alloc,
        mock_reserved,
        mock_allocated,
        mock_cuda_available,
    ) -> None:
        """snapshots property returns copy."""
        tracker = MemoryTracker()
        tracker.snapshot("test")

        snaps1 = tracker.snapshots
        snaps2 = tracker.snapshots

        assert snaps1 is not snaps2
        assert snaps1 == snaps2
