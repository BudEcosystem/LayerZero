"""
Memory tracking for CUDA graph capture validation.

This module provides:
- MemoryTracker: Track memory changes during graph operations
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

import torch

logger = logging.getLogger(__name__)


@dataclass
class MemorySnapshot:
    """Snapshot of GPU memory state.

    Attributes:
        allocated_bytes: Currently allocated memory in bytes.
        reserved_bytes: Reserved memory in bytes.
        max_allocated_bytes: Maximum allocated memory in bytes.
        timestamp_ns: Timestamp in nanoseconds.
        label: Optional label for this snapshot.
    """

    allocated_bytes: int
    reserved_bytes: int
    max_allocated_bytes: int
    timestamp_ns: int
    label: str = ""

    @property
    def allocated_mb(self) -> float:
        """Get allocated memory in MB."""
        return self.allocated_bytes / (1024 * 1024)

    @property
    def reserved_mb(self) -> float:
        """Get reserved memory in MB."""
        return self.reserved_bytes / (1024 * 1024)


@dataclass
class MemoryDelta:
    """Change in memory between two snapshots.

    Attributes:
        before: Snapshot before operation.
        after: Snapshot after operation.
        operation: Description of operation.
    """

    before: MemorySnapshot
    after: MemorySnapshot
    operation: str = ""

    @property
    def allocated_delta_bytes(self) -> int:
        """Get change in allocated memory in bytes."""
        return self.after.allocated_bytes - self.before.allocated_bytes

    @property
    def allocated_delta_mb(self) -> float:
        """Get change in allocated memory in MB."""
        return self.allocated_delta_bytes / (1024 * 1024)

    @property
    def reserved_delta_bytes(self) -> int:
        """Get change in reserved memory in bytes."""
        return self.after.reserved_bytes - self.before.reserved_bytes

    @property
    def reserved_delta_mb(self) -> float:
        """Get change in reserved memory in MB."""
        return self.reserved_delta_bytes / (1024 * 1024)


class MemoryTracker:
    """Track GPU memory changes during operations.

    Used to detect unexpected memory allocations during CUDA graph
    capture, which would indicate graph-unsafe operations.

    Example:
        tracker = MemoryTracker()

        # Take snapshot before
        tracker.snapshot("before_capture")

        # Do capture
        with torch.cuda.graph(g):
            output = func(inputs)

        # Take snapshot after
        tracker.snapshot("after_capture")

        # Check memory delta
        delta = tracker.get_delta("before_capture", "after_capture")
        if delta.allocated_delta_mb > 1.0:
            print("Warning: Memory allocated during capture")
    """

    def __init__(self, device: int = 0) -> None:
        """Initialize memory tracker.

        Args:
            device: CUDA device index.
        """
        self._device = device
        self._snapshots: dict[str, MemorySnapshot] = {}
        self._history: list[MemorySnapshot] = []

    @property
    def device(self) -> int:
        """Get tracked device index."""
        return self._device

    def snapshot(self, label: str = "") -> MemorySnapshot | None:
        """Take memory snapshot.

        Args:
            label: Label for this snapshot.

        Returns:
            MemorySnapshot or None if CUDA not available.
        """
        if not torch.cuda.is_available():
            return None

        import time

        snap = MemorySnapshot(
            allocated_bytes=torch.cuda.memory_allocated(self._device),
            reserved_bytes=torch.cuda.memory_reserved(self._device),
            max_allocated_bytes=torch.cuda.max_memory_allocated(self._device),
            timestamp_ns=time.perf_counter_ns(),
            label=label,
        )

        if label:
            self._snapshots[label] = snap
        self._history.append(snap)

        return snap

    def get_snapshot(self, label: str) -> MemorySnapshot | None:
        """Get snapshot by label.

        Args:
            label: Snapshot label.

        Returns:
            MemorySnapshot or None if not found.
        """
        return self._snapshots.get(label)

    def get_delta(
        self,
        before_label: str,
        after_label: str,
        operation: str = "",
    ) -> MemoryDelta | None:
        """Get memory delta between two snapshots.

        Args:
            before_label: Label of before snapshot.
            after_label: Label of after snapshot.
            operation: Description of operation.

        Returns:
            MemoryDelta or None if snapshots not found.
        """
        before = self._snapshots.get(before_label)
        after = self._snapshots.get(after_label)

        if before is None or after is None:
            return None

        return MemoryDelta(
            before=before,
            after=after,
            operation=operation,
        )

    def track_operation(
        self,
        func: Callable[..., Any],
        *args: Any,
        operation: str = "",
        **kwargs: Any,
    ) -> tuple[Any, MemoryDelta | None]:
        """Track memory changes during operation.

        Args:
            func: Function to track.
            *args: Arguments to pass.
            operation: Description of operation.
            **kwargs: Keyword arguments to pass.

        Returns:
            Tuple of (result, MemoryDelta).
        """
        before = self.snapshot(f"_{operation}_before")
        result = func(*args, **kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        after = self.snapshot(f"_{operation}_after")

        delta = None
        if before is not None and after is not None:
            delta = MemoryDelta(
                before=before,
                after=after,
                operation=operation,
            )

        return result, delta

    def check_memory_delta(
        self,
        before_label: str,
        after_label: str,
        threshold_mb: float = 1.0,
    ) -> tuple[bool, str]:
        """Check if memory delta exceeds threshold.

        Args:
            before_label: Label of before snapshot.
            after_label: Label of after snapshot.
            threshold_mb: Threshold in megabytes.

        Returns:
            Tuple of (within_threshold, message).
        """
        delta = self.get_delta(before_label, after_label)

        if delta is None:
            return True, "No snapshots available"

        allocated_delta_mb = delta.allocated_delta_mb

        if abs(allocated_delta_mb) > threshold_mb:
            return False, (
                f"Memory delta {allocated_delta_mb:.2f}MB exceeds "
                f"threshold {threshold_mb:.2f}MB"
            )

        return True, f"Memory delta {allocated_delta_mb:.2f}MB within threshold"

    def reset(self) -> None:
        """Reset tracker state."""
        self._snapshots.clear()
        self._history.clear()

    def clear_cache(self) -> None:
        """Clear CUDA cache."""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def reset_max_memory(self) -> None:
        """Reset max memory stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_max_memory_allocated(self._device)

    @property
    def history(self) -> list[MemorySnapshot]:
        """Get snapshot history."""
        return self._history.copy()

    @property
    def snapshots(self) -> dict[str, MemorySnapshot]:
        """Get labeled snapshots."""
        return self._snapshots.copy()
