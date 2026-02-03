"""
Capabilities descriptor validation.

This module provides:
- ValidationError: Error raised on invalid descriptor
- ValidationResult: Result of validation
- CapabilitiesValidator: Validates capabilities descriptors
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from layerzero.capabilities.schema import (
    SUPPORTED_SCHEMA_VERSIONS,
    get_schema_for_version,
)

logger = logging.getLogger(__name__)


class ValidationError(Exception):
    """Error raised when descriptor validation fails.

    Attributes:
        errors: List of validation error messages.
    """

    def __init__(self, errors: list[str]) -> None:
        """Initialize validation error.

        Args:
            errors: List of error messages.
        """
        self.errors = errors
        message = f"Validation failed: {', '.join(errors)}"
        super().__init__(message)


@dataclass
class ValidationResult:
    """Result of descriptor validation.

    Attributes:
        is_valid: Whether descriptor is valid.
        errors: List of error messages (empty if valid).
    """

    is_valid: bool
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict with is_valid and errors.
        """
        return {
            "is_valid": self.is_valid,
            "errors": self.errors,
        }

    def __str__(self) -> str:
        """String representation."""
        if self.is_valid:
            return "ValidationResult: Valid"
        return f"ValidationResult: Invalid - {'; '.join(self.errors)}"


class CapabilitiesValidator:
    """Validates capabilities descriptors.

    Ensures descriptors conform to their schema version
    and contain all required fields with valid values.

    Example:
        validator = CapabilitiesValidator()

        result = validator.validate(descriptor)
        if not result.is_valid:
            print(f"Errors: {result.errors}")
    """

    def __init__(self, strict: bool = False) -> None:
        """Initialize validator.

        Args:
            strict: If True, validate() raises ValidationError on failure.
        """
        self._strict = strict

    def validate(self, descriptor: dict[str, Any]) -> ValidationResult:
        """Validate capabilities descriptor.

        Args:
            descriptor: Capabilities descriptor to validate.

        Returns:
            ValidationResult with validation status.

        Raises:
            ValidationError: If strict mode and validation fails.
        """
        errors: list[str] = []

        # Check schema_version present
        if "schema_version" not in descriptor:
            errors.append("Missing required field: schema_version")
            return self._create_result(False, errors)

        schema_version = descriptor["schema_version"]

        # Check schema version supported
        if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
            errors.append(
                f"Unsupported schema version: {schema_version}. "
                f"Supported versions: {', '.join(sorted(SUPPORTED_SCHEMA_VERSIONS))}"
            )
            return self._create_result(False, errors)

        # Get schema for version
        schema = get_schema_for_version(schema_version)

        if schema is None:
            errors.append(f"Failed to get schema for version: {schema_version}")
            return self._create_result(False, errors)

        # Validate against schema
        schema_errors = schema.validate(descriptor)
        errors.extend(schema_errors)

        is_valid = len(errors) == 0

        return self._create_result(is_valid, errors)

    def _create_result(
        self,
        is_valid: bool,
        errors: list[str],
    ) -> ValidationResult:
        """Create validation result.

        Args:
            is_valid: Whether validation passed.
            errors: List of error messages.

        Returns:
            ValidationResult.

        Raises:
            ValidationError: If strict mode and not valid.
        """
        if not is_valid and self._strict:
            raise ValidationError(errors)

        return ValidationResult(is_valid=is_valid, errors=errors)
