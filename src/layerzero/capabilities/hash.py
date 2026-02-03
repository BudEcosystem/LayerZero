"""
Capabilities hash computation for cache invalidation.

This module provides:
- CapabilitiesHasher: Computes hashes for descriptors
- compute_capabilities_hash: Convenience function
- hash_in_cache_key: Generate cache key with hash
"""
from __future__ import annotations

import hashlib
import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class CapabilitiesHasher:
    """Computes hashes for capabilities descriptors.

    Used for cache invalidation - when a descriptor changes,
    its hash changes, invalidating cached selections.

    Example:
        hasher = CapabilitiesHasher()

        hash1 = hasher.compute(descriptor)
        # Modify descriptor
        hash2 = hasher.compute(modified_descriptor)
        assert hash1 != hash2  # Hash changed
    """

    def __init__(self, algorithm: str = "sha256") -> None:
        """Initialize hasher.

        Args:
            algorithm: Hash algorithm (sha256, md5, etc.).
        """
        self._algorithm = algorithm

    def compute(self, descriptor: dict[str, Any]) -> str:
        """Compute hash for descriptor.

        Args:
            descriptor: Capabilities descriptor.

        Returns:
            Hexadecimal hash string.
        """
        # Serialize to deterministic JSON (sorted keys)
        json_str = json.dumps(descriptor, sort_keys=True, separators=(',', ':'))

        # Compute hash
        if self._algorithm == "sha256":
            hash_obj = hashlib.sha256(json_str.encode('utf-8'))
        elif self._algorithm == "md5":
            hash_obj = hashlib.md5(json_str.encode('utf-8'))
        elif self._algorithm == "sha1":
            hash_obj = hashlib.sha1(json_str.encode('utf-8'))
        else:
            # Default to sha256
            hash_obj = hashlib.sha256(json_str.encode('utf-8'))

        hash_value = hash_obj.hexdigest()

        logger.debug(
            "Computed %s hash for descriptor: %s",
            self._algorithm,
            hash_value[:16],
        )

        return hash_value


def compute_capabilities_hash(
    descriptor: dict[str, Any],
    algorithm: str = "sha256",
) -> str:
    """Convenience function to compute capabilities hash.

    Args:
        descriptor: Capabilities descriptor.
        algorithm: Hash algorithm.

    Returns:
        Hexadecimal hash string.
    """
    hasher = CapabilitiesHasher(algorithm=algorithm)
    return hasher.compute(descriptor)


def hash_in_cache_key(
    prefix: str,
    descriptor: dict[str, Any],
    algorithm: str = "sha256",
) -> str:
    """Generate cache key with descriptor hash.

    Args:
        prefix: Cache key prefix.
        descriptor: Capabilities descriptor.
        algorithm: Hash algorithm.

    Returns:
        Cache key in format "{prefix}_{hash}".
    """
    hash_value = compute_capabilities_hash(descriptor, algorithm=algorithm)
    return f"{prefix}_{hash_value}"
