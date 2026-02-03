"""Tests for HuggingFace Kernel Hub validation."""
from __future__ import annotations

from pathlib import Path
import pytest

from layerzero.backends.hf_kernels.validator import (
    ABIValidator,
    validate_abi3_compatibility,
    validate_manylinux_compatibility,
    validate_torch_ops_namespace,
)


class TestABIValidator:
    """Test ABI validation."""

    def test_validator_instantiation(self) -> None:
        """ABIValidator can be instantiated."""
        validator = ABIValidator()
        assert validator is not None

    def test_validator_has_validate_abi3(self) -> None:
        """Validator has validate_abi3 method."""
        validator = ABIValidator()
        assert hasattr(validator, "validate_abi3")
        assert callable(validator.validate_abi3)

    def test_validator_has_validate_manylinux(self) -> None:
        """Validator has validate_manylinux method."""
        validator = ABIValidator()
        assert hasattr(validator, "validate_manylinux")
        assert callable(validator.validate_manylinux)


class TestABI3Validation:
    """Test ABI3 compatibility validation."""

    def test_validate_abi3_returns_bool(self) -> None:
        """validate_abi3_compatibility returns boolean."""
        # Non-existent path should return False
        result = validate_abi3_compatibility("/nonexistent/kernel.so")
        assert isinstance(result, bool)
        assert result is False

    def test_validate_abi3_nonexistent_file(self) -> None:
        """Non-existent file returns False."""
        result = validate_abi3_compatibility("/path/that/does/not/exist.so")
        assert result is False


class TestManylinuxValidation:
    """Test manylinux compatibility validation."""

    def test_validate_manylinux_returns_bool(self) -> None:
        """validate_manylinux_compatibility returns boolean."""
        result = validate_manylinux_compatibility("/nonexistent/kernel.so")
        assert isinstance(result, bool)

    def test_validate_manylinux_nonexistent_file(self) -> None:
        """Non-existent file returns False."""
        result = validate_manylinux_compatibility("/path/that/does/not/exist.so")
        assert result is False


class TestNamespaceValidation:
    """Test torch.ops namespace validation."""

    def test_validate_namespace_returns_list(self) -> None:
        """validate_torch_ops_namespace returns list."""
        result = validate_torch_ops_namespace(
            namespace="my_kernel",
            existing_ops=set(),
        )
        assert isinstance(result, list)

    def test_validate_namespace_empty_existing(self) -> None:
        """Empty existing ops allows any namespace."""
        result = validate_torch_ops_namespace(
            namespace="unique_namespace",
            existing_ops=set(),
        )
        # No collisions with empty set
        assert len(result) == 0

    def test_validate_namespace_detects_collision(self) -> None:
        """Detect namespace collision."""
        existing = {"flash_attn", "triton_ops"}
        result = validate_torch_ops_namespace(
            namespace="flash_attn",
            existing_ops=existing,
        )
        # Should detect collision
        assert len(result) > 0
        assert "flash_attn" in result[0]

    def test_validate_namespace_no_collision(self) -> None:
        """No collision with different namespace."""
        existing = {"flash_attn", "triton_ops"}
        result = validate_torch_ops_namespace(
            namespace="my_custom_kernel",
            existing_ops=existing,
        )
        # No collision
        assert len(result) == 0


class TestValidatorWithMockKernel:
    """Test validator with mock kernel files."""

    def test_validate_mock_kernel_dir(
        self,
        mock_kernel_dir: Path,
    ) -> None:
        """Validate mock kernel files."""
        validator = ABIValidator()

        # Mock files don't have real SO structure
        kernel_path = mock_kernel_dir / "flash_attn.so"
        assert kernel_path.exists()

        # Should return False for invalid SO files
        result = validator.validate_abi3(str(kernel_path))
        assert result is False
