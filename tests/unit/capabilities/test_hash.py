"""Tests for capabilities hash computation."""
from __future__ import annotations

import pytest
from typing import Any

from layerzero.capabilities.hash import (
    CapabilitiesHasher,
    compute_capabilities_hash,
    hash_in_cache_key,
)


class TestCapabilitiesHasher:
    """Tests for CapabilitiesHasher."""

    def test_hash_computed(self, valid_capabilities_v1) -> None:
        """Descriptor hash is computed."""
        hasher = CapabilitiesHasher()

        hash_value = hasher.compute(valid_capabilities_v1)

        assert hash_value is not None
        assert len(hash_value) > 0

    def test_hash_deterministic(self, valid_capabilities_v1) -> None:
        """Same descriptor produces same hash."""
        hasher = CapabilitiesHasher()

        hash1 = hasher.compute(valid_capabilities_v1)
        hash2 = hasher.compute(valid_capabilities_v1)

        assert hash1 == hash2

    def test_hash_changes_on_update(self, valid_capabilities_v1) -> None:
        """Hash changes when descriptor changes."""
        hasher = CapabilitiesHasher()

        hash1 = hasher.compute(valid_capabilities_v1)

        # Modify descriptor
        modified = valid_capabilities_v1.copy()
        modified["min_sm_version"] = 90

        hash2 = hasher.compute(modified)

        assert hash1 != hash2

    def test_hash_sha256_length(self, valid_capabilities_v1) -> None:
        """Hash is SHA256 (64 hex chars)."""
        hasher = CapabilitiesHasher(algorithm="sha256")

        hash_value = hasher.compute(valid_capabilities_v1)

        assert len(hash_value) == 64

    def test_hash_ignores_order(self) -> None:
        """Hash is consistent regardless of dict key order."""
        hasher = CapabilitiesHasher()

        desc1 = {
            "schema_version": "1.0",
            "kernel_id": "test",
            "operation": "attention",
        }
        desc2 = {
            "operation": "attention",
            "kernel_id": "test",
            "schema_version": "1.0",
        }

        hash1 = hasher.compute(desc1)
        hash2 = hasher.compute(desc2)

        assert hash1 == hash2


class TestCapabilitiesHashConvenience:
    """Tests for convenience functions."""

    def test_compute_capabilities_hash(self, valid_capabilities_v1) -> None:
        """compute_capabilities_hash convenience function works."""
        hash_value = compute_capabilities_hash(valid_capabilities_v1)

        assert hash_value is not None
        assert len(hash_value) == 64


class TestHashInCacheKey:
    """Tests for hash_in_cache_key."""

    def test_hash_in_cache_key(self, valid_capabilities_v1) -> None:
        """Hash included in cache key."""
        cache_key = hash_in_cache_key(
            prefix="kernel_cache",
            descriptor=valid_capabilities_v1,
        )

        assert cache_key is not None
        assert "kernel_cache" in cache_key

    def test_cache_key_deterministic(self, valid_capabilities_v1) -> None:
        """Cache key is deterministic."""
        key1 = hash_in_cache_key("prefix", valid_capabilities_v1)
        key2 = hash_in_cache_key("prefix", valid_capabilities_v1)

        assert key1 == key2

    def test_cache_key_changes_with_descriptor(self, valid_capabilities_v1) -> None:
        """Cache key changes when descriptor changes."""
        key1 = hash_in_cache_key("prefix", valid_capabilities_v1)

        modified = valid_capabilities_v1.copy()
        modified["kernel_id"] = "different_kernel"

        key2 = hash_in_cache_key("prefix", modified)

        assert key1 != key2

    def test_cache_key_format(self, valid_capabilities_v1) -> None:
        """Cache key has expected format."""
        cache_key = hash_in_cache_key("prefix", valid_capabilities_v1)

        # Expected format: "prefix_<hash>"
        assert cache_key.startswith("prefix_")
        parts = cache_key.split("_")
        assert len(parts) >= 2
