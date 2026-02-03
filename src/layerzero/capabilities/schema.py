"""
Capabilities descriptor schema definitions.

This module provides:
- SUPPORTED_SCHEMA_VERSIONS: Set of supported schema versions
- SchemaV1: Schema definition for v1 descriptors
- get_schema_for_version: Get schema for a version
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


# Supported schema versions
SUPPORTED_SCHEMA_VERSIONS: frozenset[str] = frozenset(["1.0"])


class Schema(ABC):
    """Abstract base class for schema definitions."""

    @property
    @abstractmethod
    def version(self) -> str:
        """Get schema version."""
        pass

    @property
    @abstractmethod
    def required_fields(self) -> frozenset[str]:
        """Get required fields for this schema."""
        pass

    @abstractmethod
    def validate(self, descriptor: dict[str, Any]) -> list[str]:
        """Validate descriptor against schema.

        Args:
            descriptor: Capabilities descriptor.

        Returns:
            List of error messages (empty if valid).
        """
        pass


@dataclass
class SchemaV1(Schema):
    """Schema definition for v1 capabilities descriptors.

    Required fields:
    - schema_version: Must be "1.0"
    - kernel_id: Unique kernel identifier
    - operation: Operation type (attention, matmul, etc.)

    Optional fields:
    - backend: Backend name
    - constraints: Dict of constraint definitions
    - dtypes: List of supported dtypes
    - platforms: List of supported platforms
    - min_sm_version: Minimum CUDA SM version
    """

    _version: str = "1.0"
    _required_fields: frozenset[str] = field(
        default_factory=lambda: frozenset(["schema_version", "kernel_id", "operation"])
    )

    @property
    def version(self) -> str:
        """Get schema version."""
        return self._version

    @property
    def required_fields(self) -> frozenset[str]:
        """Get required fields."""
        return self._required_fields

    def validate(self, descriptor: dict[str, Any]) -> list[str]:
        """Validate descriptor against v1 schema.

        Args:
            descriptor: Capabilities descriptor.

        Returns:
            List of error messages.
        """
        errors: list[str] = []

        # Check required fields
        for field_name in self._required_fields:
            if field_name not in descriptor:
                errors.append(f"Missing required field: {field_name}")

        # Validate schema_version
        if "schema_version" in descriptor:
            if descriptor["schema_version"] != self._version:
                errors.append(
                    f"Schema version mismatch: expected {self._version}, "
                    f"got {descriptor['schema_version']}"
                )

        # Validate constraints if present
        if "constraints" in descriptor:
            constraint_errors = self._validate_constraints(descriptor["constraints"])
            errors.extend(constraint_errors)

        return errors

    def _validate_constraints(self, constraints: dict[str, Any]) -> list[str]:
        """Validate constraints section.

        Args:
            constraints: Constraints dictionary.

        Returns:
            List of error messages.
        """
        errors: list[str] = []

        for dim_name, constraint in constraints.items():
            if not isinstance(constraint, dict):
                errors.append(f"Constraint '{dim_name}' must be a dictionary")
                continue

            # Check min/max ordering
            if "min" in constraint and "max" in constraint:
                min_val = constraint["min"]
                max_val = constraint["max"]

                if min_val > max_val:
                    errors.append(
                        f"Invalid constraint '{dim_name}': min ({min_val}) > max ({max_val})"
                    )

            # Validate valid_values if present
            if "valid" in constraint:
                valid_vals = constraint["valid"]
                if not isinstance(valid_vals, list):
                    errors.append(
                        f"Constraint '{dim_name}' valid values must be a list"
                    )

        return errors


def get_schema_for_version(version: str) -> Schema | None:
    """Get schema for a version.

    Args:
        version: Schema version string.

    Returns:
        Schema instance or None if version not supported.
    """
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        logger.warning("Unknown schema version: %s", version)
        return None

    if version == "1.0":
        return SchemaV1()

    return None
