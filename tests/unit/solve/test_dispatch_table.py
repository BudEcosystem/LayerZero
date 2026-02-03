"""Tests for dispatch table."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from unittest.mock import MagicMock, patch

from layerzero._solve.dispatch_table import (
    BucketRange,
    DispatchEntry,
    DispatchTable,
    ShapeBucket,
)


class TestBucketRange:
    """Tests for BucketRange dataclass."""

    def test_creation(self) -> None:
        """BucketRange stores min and max."""
        range_ = BucketRange(min_val=1, max_val=100)

        assert range_.min_val == 1
        assert range_.max_val == 100

    def test_contains_value_in_range(self) -> None:
        """contains() returns True for value in range."""
        range_ = BucketRange(min_val=10, max_val=50)

        assert range_.contains(10) is True
        assert range_.contains(30) is True
        assert range_.contains(50) is True

    def test_contains_value_outside_range(self) -> None:
        """contains() returns False for value outside range."""
        range_ = BucketRange(min_val=10, max_val=50)

        assert range_.contains(9) is False
        assert range_.contains(51) is False

    def test_single_value_range(self) -> None:
        """Range with min == max."""
        range_ = BucketRange(min_val=42, max_val=42)

        assert range_.contains(42) is True
        assert range_.contains(41) is False
        assert range_.contains(43) is False

    def test_to_dict(self) -> None:
        """BucketRange serializes to dict."""
        range_ = BucketRange(min_val=1, max_val=100)

        d = range_.to_dict()

        assert d == {"min": 1, "max": 100}

    def test_from_dict(self) -> None:
        """BucketRange deserializes from dict."""
        d = {"min": 1, "max": 100}

        range_ = BucketRange.from_dict(d)

        assert range_.min_val == 1
        assert range_.max_val == 100


class TestShapeBucket:
    """Tests for ShapeBucket."""

    def test_creation(self) -> None:
        """ShapeBucket stores dimension ranges."""
        bucket = ShapeBucket(
            ranges={
                "batch_size": BucketRange(1, 8),
                "seq_len": BucketRange(128, 2048),
            }
        )

        assert "batch_size" in bucket.ranges
        assert "seq_len" in bucket.ranges

    def test_matches_shape_in_bucket(self) -> None:
        """matches() returns True for shape in bucket."""
        bucket = ShapeBucket(
            ranges={
                "batch_size": BucketRange(1, 8),
                "seq_len": BucketRange(128, 2048),
            }
        )

        shape = {"batch_size": 4, "seq_len": 512}

        assert bucket.matches(shape) is True

    def test_matches_shape_outside_bucket(self) -> None:
        """matches() returns False for shape outside bucket."""
        bucket = ShapeBucket(
            ranges={
                "batch_size": BucketRange(1, 8),
                "seq_len": BucketRange(128, 2048),
            }
        )

        # batch_size out of range
        shape1 = {"batch_size": 16, "seq_len": 512}
        assert bucket.matches(shape1) is False

        # seq_len out of range
        shape2 = {"batch_size": 4, "seq_len": 4096}
        assert bucket.matches(shape2) is False

    def test_matches_missing_dimension(self) -> None:
        """matches() returns False if dimension missing."""
        bucket = ShapeBucket(
            ranges={
                "batch_size": BucketRange(1, 8),
                "seq_len": BucketRange(128, 2048),
            }
        )

        shape = {"batch_size": 4}  # Missing seq_len

        assert bucket.matches(shape) is False

    def test_matches_extra_dimensions_ignored(self) -> None:
        """Extra dimensions in shape are ignored."""
        bucket = ShapeBucket(
            ranges={
                "batch_size": BucketRange(1, 8),
            }
        )

        shape = {"batch_size": 4, "extra_dim": 100}

        assert bucket.matches(shape) is True

    def test_to_dict(self) -> None:
        """ShapeBucket serializes to dict."""
        bucket = ShapeBucket(
            ranges={
                "batch_size": BucketRange(1, 8),
            }
        )

        d = bucket.to_dict()

        assert "ranges" in d
        assert "batch_size" in d["ranges"]

    def test_from_dict(self) -> None:
        """ShapeBucket deserializes from dict."""
        d = {
            "ranges": {
                "batch_size": {"min": 1, "max": 8},
            }
        }

        bucket = ShapeBucket.from_dict(d)

        assert bucket.ranges["batch_size"].min_val == 1
        assert bucket.ranges["batch_size"].max_val == 8


class TestDispatchEntry:
    """Tests for DispatchEntry."""

    def test_creation(self) -> None:
        """DispatchEntry stores kernel and bucket."""
        bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        entry = DispatchEntry(
            kernel_id="flash_attn_v2",
            bucket=bucket,
            priority=100,
        )

        assert entry.kernel_id == "flash_attn_v2"
        assert entry.bucket is bucket
        assert entry.priority == 100

    def test_matches_delegates_to_bucket(self) -> None:
        """matches() delegates to bucket."""
        bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        entry = DispatchEntry(
            kernel_id="test_kernel",
            bucket=bucket,
            priority=100,
        )

        assert entry.matches({"batch_size": 4}) is True
        assert entry.matches({"batch_size": 16}) is False

    def test_to_dict(self) -> None:
        """DispatchEntry serializes to dict."""
        bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        entry = DispatchEntry(
            kernel_id="test_kernel",
            bucket=bucket,
            priority=100,
        )

        d = entry.to_dict()

        assert d["kernel_id"] == "test_kernel"
        assert d["priority"] == 100
        assert "bucket" in d

    def test_from_dict(self) -> None:
        """DispatchEntry deserializes from dict."""
        d = {
            "kernel_id": "test_kernel",
            "priority": 100,
            "bucket": {
                "ranges": {"batch_size": {"min": 1, "max": 8}},
            },
        }

        entry = DispatchEntry.from_dict(d)

        assert entry.kernel_id == "test_kernel"
        assert entry.priority == 100


class TestDispatchTable:
    """Tests for DispatchTable."""

    def test_creation(self) -> None:
        """DispatchTable creates empty table."""
        table = DispatchTable()

        assert len(table) == 0
        assert table.bucket_count == 0

    def test_add_entry(self) -> None:
        """add_entry adds dispatch entry."""
        table = DispatchTable()
        bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        entry = DispatchEntry(
            kernel_id="test_kernel",
            bucket=bucket,
            priority=100,
        )

        table.add_entry(entry)

        assert len(table) == 1

    def test_dispatch_table_lookup(self) -> None:
        """Dispatch table lookup works."""
        table = DispatchTable()

        # Add entry
        bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        entry = DispatchEntry(
            kernel_id="test_kernel",
            bucket=bucket,
            priority=100,
        )
        table.add_entry(entry)

        # Lookup should find the kernel
        result = table.lookup({"batch_size": 4})

        assert result is not None
        assert result.kernel_id == "test_kernel"

    def test_dispatch_table_bucket_match(self) -> None:
        """Bucket matching works correctly."""
        table = DispatchTable()

        # Add two entries with different buckets
        bucket1 = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        entry1 = DispatchEntry(
            kernel_id="small_batch_kernel",
            bucket=bucket1,
            priority=100,
        )

        bucket2 = ShapeBucket(
            ranges={"batch_size": BucketRange(9, 32)}
        )
        entry2 = DispatchEntry(
            kernel_id="large_batch_kernel",
            bucket=bucket2,
            priority=100,
        )

        table.add_entry(entry1)
        table.add_entry(entry2)

        # Lookup should match correct bucket
        small_result = table.lookup({"batch_size": 4})
        large_result = table.lookup({"batch_size": 16})

        assert small_result.kernel_id == "small_batch_kernel"
        assert large_result.kernel_id == "large_batch_kernel"

    def test_dispatch_table_priority_ordering(self) -> None:
        """Higher priority entries are selected first."""
        table = DispatchTable()

        # Add two entries with overlapping buckets but different priorities
        bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        low_priority = DispatchEntry(
            kernel_id="low_priority_kernel",
            bucket=bucket,
            priority=50,
        )
        high_priority = DispatchEntry(
            kernel_id="high_priority_kernel",
            bucket=bucket,
            priority=100,
        )

        table.add_entry(low_priority)
        table.add_entry(high_priority)

        result = table.lookup({"batch_size": 4})

        assert result.kernel_id == "high_priority_kernel"

    def test_dispatch_table_fallback(self) -> None:
        """Fallback when bucket miss."""
        table = DispatchTable()

        # Add fallback entry (empty bucket matches everything)
        fallback_bucket = ShapeBucket(ranges={})
        fallback = DispatchEntry(
            kernel_id="fallback_kernel",
            bucket=fallback_bucket,
            priority=10,
        )
        table.add_entry(fallback)

        # Add specific entry
        specific_bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        specific = DispatchEntry(
            kernel_id="specific_kernel",
            bucket=specific_bucket,
            priority=100,
        )
        table.add_entry(specific)

        # Specific match
        result1 = table.lookup({"batch_size": 4})
        assert result1.kernel_id == "specific_kernel"

        # Fallback for out-of-range
        result2 = table.lookup({"batch_size": 100})
        assert result2.kernel_id == "fallback_kernel"

    def test_dispatch_table_no_match(self) -> None:
        """Returns None when no match."""
        table = DispatchTable()

        bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        entry = DispatchEntry(
            kernel_id="test_kernel",
            bucket=bucket,
            priority=100,
        )
        table.add_entry(entry)

        result = table.lookup({"batch_size": 100})

        assert result is None

    def test_dispatch_table_persistence_save(self) -> None:
        """Dispatch table can be saved."""
        table = DispatchTable()

        bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        entry = DispatchEntry(
            kernel_id="test_kernel",
            bucket=bucket,
            priority=100,
        )
        table.add_entry(entry)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = Path(f.name)

        try:
            table.save(path)

            assert path.exists()

            # Verify contents
            with open(path) as f:
                data = json.load(f)

            assert "entries" in data
            assert len(data["entries"]) == 1
        finally:
            path.unlink(missing_ok=True)

    def test_dispatch_table_persistence_load(self) -> None:
        """Dispatch table can be loaded."""
        # Create table and save
        original = DispatchTable()
        bucket = ShapeBucket(
            ranges={"batch_size": BucketRange(1, 8)}
        )
        entry = DispatchEntry(
            kernel_id="test_kernel",
            bucket=bucket,
            priority=100,
        )
        original.add_entry(entry)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            path = Path(f.name)

        try:
            original.save(path)

            # Load and verify
            loaded = DispatchTable.load(path)

            assert len(loaded) == 1

            result = loaded.lookup({"batch_size": 4})
            assert result.kernel_id == "test_kernel"
        finally:
            path.unlink(missing_ok=True)

    def test_bucket_count(self) -> None:
        """bucket_count returns number of unique buckets."""
        table = DispatchTable()

        bucket1 = ShapeBucket(ranges={"batch_size": BucketRange(1, 8)})
        bucket2 = ShapeBucket(ranges={"batch_size": BucketRange(9, 16)})

        table.add_entry(DispatchEntry("k1", bucket1, 100))
        table.add_entry(DispatchEntry("k2", bucket2, 100))

        assert table.bucket_count == 2

    def test_to_dict(self) -> None:
        """DispatchTable serializes to dict."""
        table = DispatchTable()

        bucket = ShapeBucket(ranges={"batch_size": BucketRange(1, 8)})
        entry = DispatchEntry("test_kernel", bucket, 100)
        table.add_entry(entry)

        d = table.to_dict()

        assert "entries" in d
        assert len(d["entries"]) == 1

    def test_from_dict(self) -> None:
        """DispatchTable deserializes from dict."""
        d = {
            "entries": [
                {
                    "kernel_id": "test_kernel",
                    "priority": 100,
                    "bucket": {
                        "ranges": {"batch_size": {"min": 1, "max": 8}},
                    },
                }
            ]
        }

        table = DispatchTable.from_dict(d)

        assert len(table) == 1

    def test_clear(self) -> None:
        """clear() removes all entries."""
        table = DispatchTable()

        bucket = ShapeBucket(ranges={"batch_size": BucketRange(1, 8)})
        table.add_entry(DispatchEntry("k1", bucket, 100))
        table.add_entry(DispatchEntry("k2", bucket, 50))

        assert len(table) == 2

        table.clear()

        assert len(table) == 0

    def test_iter(self) -> None:
        """DispatchTable is iterable."""
        table = DispatchTable()

        bucket = ShapeBucket(ranges={})
        table.add_entry(DispatchEntry("k1", bucket, 100))
        table.add_entry(DispatchEntry("k2", bucket, 50))

        entries = list(table)

        assert len(entries) == 2
