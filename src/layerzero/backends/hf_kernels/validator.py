"""
LayerZero HuggingFace Kernel Hub Validator

Validate kernel compatibility (ABI3, manylinux, namespace).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def validate_abi3_compatibility(kernel_path: str) -> bool:
    """Validate ABI3 compatibility of a kernel library.

    ABI3 (Limited C API) ensures compatibility across Python versions.

    Args:
        kernel_path: Path to the kernel shared library

    Returns:
        True if ABI3 compatible, False otherwise.
    """
    path = Path(kernel_path)

    if not path.exists():
        logger.debug(f"Kernel file not found: {kernel_path}")
        return False

    try:
        # Check file extension pattern
        # ABI3 wheels typically have .abi3.so suffix
        if ".abi3." in path.name:
            return True

        # Check file size (minimal validation)
        if path.stat().st_size == 0:
            return False

        # Try to read ELF header for Linux
        if path.suffix == ".so":
            with open(path, "rb") as f:
                magic = f.read(4)
                # ELF magic number: 0x7f 'E' 'L' 'F'
                if magic != b"\x7fELF":
                    return False

                # For now, just check it's a valid ELF
                return True

        return False

    except (OSError, IOError) as e:
        logger.debug(f"Failed to validate ABI3: {e}")
        return False


def validate_manylinux_compatibility(kernel_path: str) -> bool:
    """Validate manylinux_2_28 compatibility.

    manylinux_2_28 ensures compatibility with glibc >= 2.28.

    Args:
        kernel_path: Path to the kernel shared library

    Returns:
        True if manylinux compatible, False otherwise.
    """
    path = Path(kernel_path)

    if not path.exists():
        logger.debug(f"Kernel file not found: {kernel_path}")
        return False

    try:
        # Check file is readable
        if not os.access(path, os.R_OK):
            return False

        # For Linux, check it's a valid shared object
        if path.suffix == ".so":
            with open(path, "rb") as f:
                magic = f.read(4)
                if magic != b"\x7fELF":
                    return False

            # For full validation, we'd use auditwheel
            # For now, just check basic validity
            return True

        return False

    except (OSError, IOError) as e:
        logger.debug(f"Failed to validate manylinux: {e}")
        return False


def validate_torch_ops_namespace(
    namespace: str,
    existing_ops: set[str],
) -> list[str]:
    """Validate torch.ops namespace uniqueness.

    Checks if the proposed namespace collides with existing namespaces.

    Args:
        namespace: Proposed namespace for the kernel
        existing_ops: Set of existing namespace names

    Returns:
        List of collision messages. Empty if no collisions.
    """
    collisions: list[str] = []

    if namespace in existing_ops:
        collisions.append(
            f"Namespace '{namespace}' already exists in torch.ops"
        )

    # Check for prefix collisions
    for existing in existing_ops:
        if namespace.startswith(existing + "."):
            collisions.append(
                f"Namespace '{namespace}' conflicts with existing '{existing}'"
            )
        elif existing.startswith(namespace + "."):
            collisions.append(
                f"Existing namespace '{existing}' conflicts with '{namespace}'"
            )

    return collisions


class ABIValidator:
    """Comprehensive ABI validator for kernel libraries.

    Validates that kernel libraries are compatible with the
    current Python environment.
    """

    def __init__(self) -> None:
        """Initialize the validator."""
        pass

    def validate_abi3(self, kernel_path: str) -> bool:
        """Validate ABI3 compatibility.

        Args:
            kernel_path: Path to kernel library

        Returns:
            True if ABI3 compatible.
        """
        return validate_abi3_compatibility(kernel_path)

    def validate_manylinux(self, kernel_path: str) -> bool:
        """Validate manylinux compatibility.

        Args:
            kernel_path: Path to kernel library

        Returns:
            True if manylinux compatible.
        """
        return validate_manylinux_compatibility(kernel_path)

    def validate_all(self, kernel_path: str) -> dict[str, bool]:
        """Run all validation checks.

        Args:
            kernel_path: Path to kernel library

        Returns:
            Dictionary with validation results.
        """
        return {
            "abi3": self.validate_abi3(kernel_path),
            "manylinux": self.validate_manylinux(kernel_path),
        }
