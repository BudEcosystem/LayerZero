"""Tests for capabilities descriptor validation."""
from __future__ import annotations

import pytest
from typing import Any

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


class TestCapabilitiesValidator:
    """Tests for CapabilitiesValidator."""

    def test_valid_schema_accepted(self, valid_capabilities_v1) -> None:
        """Valid schema v1 accepted."""
        validator = CapabilitiesValidator()

        result = validator.validate(valid_capabilities_v1)

        assert result.is_valid is True
        assert result.errors == []

    def test_unknown_schema_rejected(self, valid_capabilities_v2) -> None:
        """Unknown schema version rejected."""
        validator = CapabilitiesValidator()

        result = validator.validate(valid_capabilities_v2)

        assert result.is_valid is False
        assert any("schema_version" in str(e).lower() or "version" in str(e).lower()
                   for e in result.errors)

    def test_missing_required_field_rejected(self, invalid_missing_required) -> None:
        """Missing required field rejected."""
        validator = CapabilitiesValidator()

        result = validator.validate(invalid_missing_required)

        assert result.is_valid is False
        assert any("kernel_id" in str(e).lower() or "required" in str(e).lower()
                   for e in result.errors)

    def test_invalid_constraint_rejected(self, invalid_constraint) -> None:
        """Invalid constraint rejected."""
        validator = CapabilitiesValidator()

        result = validator.validate(invalid_constraint)

        assert result.is_valid is False
        assert any("constraint" in str(e).lower() or "min" in str(e).lower() or "max" in str(e).lower()
                   for e in result.errors)

    def test_minimal_descriptor_valid(self, minimal_valid_descriptor) -> None:
        """Minimal descriptor with only required fields is valid."""
        validator = CapabilitiesValidator()

        result = validator.validate(minimal_valid_descriptor)

        assert result.is_valid is True

    def test_missing_schema_version_rejected(self) -> None:
        """Descriptor without schema_version is rejected."""
        validator = CapabilitiesValidator()

        descriptor = {
            "kernel_id": "test",
            "operation": "attention",
        }

        result = validator.validate(descriptor)

        assert result.is_valid is False

    def test_validate_raises_on_strict(self, invalid_missing_required) -> None:
        """validate() can raise ValidationError in strict mode."""
        validator = CapabilitiesValidator(strict=True)

        with pytest.raises(ValidationError):
            validator.validate(invalid_missing_required)

    def test_validate_returns_result_on_non_strict(self, invalid_missing_required) -> None:
        """validate() returns ValidationResult in non-strict mode."""
        validator = CapabilitiesValidator(strict=False)

        result = validator.validate(invalid_missing_required)

        assert isinstance(result, ValidationResult)
        assert result.is_valid is False


class TestValidationResult:
    """Tests for ValidationResult."""

    def test_valid_result(self) -> None:
        """Valid result has no errors."""
        result = ValidationResult(is_valid=True, errors=[])

        assert result.is_valid is True
        assert result.errors == []
        assert len(result.errors) == 0

    def test_invalid_result_with_errors(self) -> None:
        """Invalid result contains error messages."""
        errors = ["Missing required field: kernel_id", "Invalid constraint"]
        result = ValidationResult(is_valid=False, errors=errors)

        assert result.is_valid is False
        assert len(result.errors) == 2

    def test_result_to_dict(self) -> None:
        """ValidationResult serializes to dict."""
        result = ValidationResult(is_valid=True, errors=[])

        d = result.to_dict()

        assert "is_valid" in d
        assert "errors" in d

    def test_result_str_representation(self) -> None:
        """ValidationResult has string representation."""
        result = ValidationResult(is_valid=False, errors=["Error 1"])

        s = str(result)

        assert "Error" in s or "error" in s


class TestSchemaVersions:
    """Tests for schema version handling."""

    def test_supported_versions_non_empty(self) -> None:
        """SUPPORTED_SCHEMA_VERSIONS is not empty."""
        assert len(SUPPORTED_SCHEMA_VERSIONS) > 0

    def test_version_1_supported(self) -> None:
        """Version 1.0 is supported."""
        assert "1.0" in SUPPORTED_SCHEMA_VERSIONS

    def test_get_schema_for_valid_version(self) -> None:
        """get_schema_for_version returns schema for valid version."""
        schema = get_schema_for_version("1.0")

        assert schema is not None
        assert isinstance(schema, SchemaV1)

    def test_get_schema_for_invalid_version(self) -> None:
        """get_schema_for_version returns None for invalid version."""
        schema = get_schema_for_version("99.0")

        assert schema is None


class TestSchemaV1:
    """Tests for SchemaV1."""

    def test_required_fields(self) -> None:
        """SchemaV1 has required fields defined."""
        schema = SchemaV1()

        assert "kernel_id" in schema.required_fields
        assert "operation" in schema.required_fields
        assert "schema_version" in schema.required_fields

    def test_validate_complete_descriptor(self, valid_capabilities_v1) -> None:
        """SchemaV1 validates complete descriptor."""
        schema = SchemaV1()

        errors = schema.validate(valid_capabilities_v1)

        assert errors == []

    def test_validate_missing_field(self) -> None:
        """SchemaV1 reports missing required field."""
        schema = SchemaV1()

        descriptor = {
            "schema_version": "1.0",
            "operation": "attention",
            # Missing kernel_id
        }

        errors = schema.validate(descriptor)

        assert len(errors) > 0
        assert any("kernel_id" in e for e in errors)

    def test_validate_constraint_ranges(self, valid_capabilities_v1) -> None:
        """SchemaV1 validates constraint ranges."""
        schema = SchemaV1()

        # Valid constraints
        errors = schema.validate(valid_capabilities_v1)
        assert errors == []

    def test_validate_invalid_constraint_range(self) -> None:
        """SchemaV1 rejects invalid constraint range."""
        schema = SchemaV1()

        descriptor = {
            "schema_version": "1.0",
            "kernel_id": "test",
            "operation": "attention",
            "constraints": {
                "head_dim": {"min": 100, "max": 50},  # Invalid: min > max
            },
        }

        errors = schema.validate(descriptor)

        assert len(errors) > 0
