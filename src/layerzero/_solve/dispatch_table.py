"""
Dispatch table for kernel selection.

This module provides:
- BucketRange: Range for a dimension in a bucket
- ShapeBucket: Shape bucket for bucketed dispatch
- DispatchEntry: Entry in dispatch table
- DispatchTable: Dispatch table for kernel selection
"""
from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BucketRange:
    """Range for a dimension in a bucket.

    Represents a closed range [min_val, max_val] for a shape dimension.

    Attributes:
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).
    """

    min_val: int
    max_val: int

    def contains(self, value: int) -> bool:
        """Check if value is within range.

        Args:
            value: Value to check.

        Returns:
            True if value is in [min_val, max_val].
        """
        return self.min_val <= value <= self.max_val

    def to_dict(self) -> dict[str, int]:
        """Serialize to dictionary.

        Returns:
            Dict with 'min' and 'max' keys.
        """
        return {"min": self.min_val, "max": self.max_val}

    @classmethod
    def from_dict(cls, d: dict[str, int]) -> BucketRange:
        """Deserialize from dictionary.

        Args:
            d: Dict with 'min' and 'max' keys.

        Returns:
            New BucketRange instance.
        """
        return cls(min_val=d["min"], max_val=d["max"])


@dataclass
class ShapeBucket:
    """Shape bucket for bucketed dispatch.

    A bucket defines ranges for each shape dimension. A shape matches
    a bucket if all its dimensions fall within the bucket's ranges.

    Attributes:
        ranges: Mapping from dimension name to BucketRange.
    """

    ranges: dict[str, BucketRange] = field(default_factory=dict)

    def matches(self, shape: dict[str, int]) -> bool:
        """Check if shape matches this bucket.

        Args:
            shape: Shape dictionary mapping dimension names to values.

        Returns:
            True if all bucket dimensions match the shape.
        """
        for dim_name, range_ in self.ranges.items():
            if dim_name not in shape:
                return False
            if not range_.contains(shape[dim_name]):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict representation.
        """
        return {
            "ranges": {
                name: range_.to_dict()
                for name, range_ in self.ranges.items()
            }
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ShapeBucket:
        """Deserialize from dictionary.

        Args:
            d: Dict representation.

        Returns:
            New ShapeBucket instance.
        """
        ranges = {
            name: BucketRange.from_dict(range_dict)
            for name, range_dict in d.get("ranges", {}).items()
        }
        return cls(ranges=ranges)


@dataclass
class DispatchEntry:
    """Entry in dispatch table.

    Associates a kernel with a shape bucket and priority.

    Attributes:
        kernel_id: Kernel identifier.
        bucket: Shape bucket for matching.
        priority: Priority for selection (higher = preferred).
    """

    kernel_id: str
    bucket: ShapeBucket
    priority: int = 100

    def matches(self, shape: dict[str, int]) -> bool:
        """Check if shape matches this entry's bucket.

        Args:
            shape: Shape dictionary.

        Returns:
            True if shape matches bucket.
        """
        return self.bucket.matches(shape)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict representation.
        """
        return {
            "kernel_id": self.kernel_id,
            "bucket": self.bucket.to_dict(),
            "priority": self.priority,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DispatchEntry:
        """Deserialize from dictionary.

        Args:
            d: Dict representation.

        Returns:
            New DispatchEntry instance.
        """
        return cls(
            kernel_id=d["kernel_id"],
            bucket=ShapeBucket.from_dict(d["bucket"]),
            priority=d.get("priority", 100),
        )


class DispatchTable:
    """Dispatch table for kernel selection.

    Maintains a table of dispatch entries for efficient kernel lookup
    based on input shapes. Entries are organized by priority.

    Thread-safe for concurrent access.

    Example:
        table = DispatchTable()

        # Add entries
        bucket = ShapeBucket(ranges={"batch_size": BucketRange(1, 8)})
        entry = DispatchEntry("flash_attn", bucket, priority=100)
        table.add_entry(entry)

        # Lookup
        result = table.lookup({"batch_size": 4})
        print(result.kernel_id)  # "flash_attn"
    """

    def __init__(self) -> None:
        """Initialize empty dispatch table."""
        self._entries: list[DispatchEntry] = []
        self._lock = threading.RLock()

    def add_entry(self, entry: DispatchEntry) -> None:
        """Add dispatch entry to table.

        Entries are kept sorted by priority (highest first).

        Args:
            entry: Dispatch entry to add.
        """
        with self._lock:
            self._entries.append(entry)
            # Sort by priority (descending)
            self._entries.sort(key=lambda e: e.priority, reverse=True)

            logger.debug(
                "Added dispatch entry: kernel=%s, priority=%d",
                entry.kernel_id,
                entry.priority,
            )

    def lookup(self, shape: dict[str, int]) -> DispatchEntry | None:
        """Lookup kernel for shape.

        Returns the highest priority entry that matches the shape.

        Args:
            shape: Shape dictionary.

        Returns:
            Matching DispatchEntry or None if no match.
        """
        with self._lock:
            for entry in self._entries:
                if entry.matches(shape):
                    logger.debug(
                        "Dispatch lookup matched: shape=%s, kernel=%s",
                        shape,
                        entry.kernel_id,
                    )
                    return entry

            logger.debug("Dispatch lookup miss: shape=%s", shape)
            return None

    def clear(self) -> None:
        """Remove all entries from table."""
        with self._lock:
            self._entries.clear()
            logger.debug("Dispatch table cleared")

    @property
    def bucket_count(self) -> int:
        """Get number of unique buckets.

        Returns:
            Number of entries (each has unique bucket).
        """
        with self._lock:
            return len(self._entries)

    def save(self, path: Path) -> None:
        """Save dispatch table to file.

        Args:
            path: Path to save to (JSON format).
        """
        with self._lock:
            data = self.to_dict()

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info("Saved dispatch table to %s", path)

    @classmethod
    def load(cls, path: Path) -> DispatchTable:
        """Load dispatch table from file.

        Args:
            path: Path to load from.

        Returns:
            Loaded DispatchTable.
        """
        with open(path) as f:
            data = json.load(f)

        table = cls.from_dict(data)
        logger.info("Loaded dispatch table from %s", path)
        return table

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict representation.
        """
        with self._lock:
            return {
                "entries": [entry.to_dict() for entry in self._entries]
            }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DispatchTable:
        """Deserialize from dictionary.

        Args:
            d: Dict representation.

        Returns:
            New DispatchTable instance.
        """
        table = cls()
        for entry_dict in d.get("entries", []):
            entry = DispatchEntry.from_dict(entry_dict)
            table.add_entry(entry)
        return table

    def __len__(self) -> int:
        """Get number of entries in table."""
        with self._lock:
            return len(self._entries)

    def __iter__(self) -> Iterator[DispatchEntry]:
        """Iterate over entries."""
        with self._lock:
            # Return copy to avoid issues with concurrent modification
            return iter(list(self._entries))

    def __contains__(self, kernel_id: str) -> bool:
        """Check if kernel is in table."""
        with self._lock:
            return any(e.kernel_id == kernel_id for e in self._entries)
