"""
Capabilities descriptor validation module.

This module provides:
- CapabilitiesValidator: Validates capabilities descriptors
- ValidationResult: Result of validation
- ValidationError: Error raised on invalid descriptor
- SchemaV1: Schema definition for v1 descriptors
- CapabilitiesHasher: Computes hashes for cache invalidation
- Constraint: Individual constraint definition
- ConstraintSet: Set of constraints
- ConstraintLoader: Loads constraints from descriptors
"""
from __future__ import annotations

from layerzero.capabilities.validator import (
    CapabilitiesValidator,
    ValidationError,
    ValidationResult,
)
from layerzero.capabilities.schema import (
    SUPPORTED_SCHEMA_VERSIONS,
    SchemaV1,
    get_schema_for_version,
)
from layerzero.capabilities.hash import (
    CapabilitiesHasher,
    compute_capabilities_hash,
    hash_in_cache_key,
)
from layerzero.capabilities.constraints import (
    Constraint,
    ConstraintLoader,
    ConstraintSet,
    load_constraints_from_descriptor,
)

__all__ = [
    # Validator
    "CapabilitiesValidator",
    "ValidationError",
    "ValidationResult",
    # Schema
    "SUPPORTED_SCHEMA_VERSIONS",
    "SchemaV1",
    "get_schema_for_version",
    # Hash
    "CapabilitiesHasher",
    "compute_capabilities_hash",
    "hash_in_cache_key",
    # Constraints
    "Constraint",
    "ConstraintLoader",
    "ConstraintSet",
    "load_constraints_from_descriptor",
]
