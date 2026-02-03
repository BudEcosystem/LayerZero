"""
Data-driven constraints from capabilities descriptors.

This module provides:
- Constraint: Individual constraint definition
- ConstraintSet: Set of constraints
- ConstraintLoader: Loads constraints from descriptors
- load_constraints_from_descriptor: Convenience function
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


@dataclass
class Constraint:
    """Individual constraint definition.

    Can define constraints as:
    - Range: min_val to max_val
    - Valid set: explicit list of valid values

    If both are specified, valid_values takes precedence.

    Attributes:
        name: Constraint name (dimension name).
        min_val: Minimum value (inclusive).
        max_val: Maximum value (inclusive).
        valid_values: List of explicitly valid values.
    """

    name: str
    min_val: int | None = None
    max_val: int | None = None
    valid_values: list[int] | None = None

    def check(self, value: int) -> bool:
        """Check if value satisfies constraint.

        Args:
            value: Value to check.

        Returns:
            True if value is valid.
        """
        # valid_values takes precedence
        if self.valid_values is not None:
            return value in self.valid_values

        # Check range
        if self.min_val is not None and value < self.min_val:
            return False
        if self.max_val is not None and value > self.max_val:
            return False

        return True


@dataclass
class ConstraintSet:
    """Set of constraints for a kernel.

    Attributes:
        constraints: List of constraints.
    """

    constraints: list[Constraint] = field(default_factory=list)

    def get(self, name: str) -> Constraint | None:
        """Get constraint by name.

        Args:
            name: Constraint name.

        Returns:
            Constraint or None if not found.
        """
        for c in self.constraints:
            if c.name == name:
                return c
        return None

    def check_all(
        self,
        values: dict[str, int],
    ) -> tuple[bool, list[str]]:
        """Check all constraints against values.

        Args:
            values: Dictionary mapping constraint names to values.

        Returns:
            Tuple of (passed, error_messages).
        """
        errors: list[str] = []

        for constraint in self.constraints:
            if constraint.name in values:
                value = values[constraint.name]
                if not constraint.check(value):
                    errors.append(
                        f"Constraint '{constraint.name}' violated: "
                        f"value {value} not in valid range/set"
                    )

        return len(errors) == 0, errors

    def __len__(self) -> int:
        """Get number of constraints."""
        return len(self.constraints)

    def __iter__(self) -> Iterator[Constraint]:
        """Iterate over constraints."""
        return iter(self.constraints)


class ConstraintLoader:
    """Loads constraints from capabilities descriptors.

    Supports loading from dict or JSON files.

    Example:
        loader = ConstraintLoader()

        # From descriptor dict
        cs = loader.load_from_descriptor(descriptor)

        # From file
        cs = loader.load_from_file(Path("capabilities.json"))
    """

    def load_from_descriptor(
        self,
        descriptor: dict[str, Any],
    ) -> ConstraintSet:
        """Load constraints from descriptor.

        Args:
            descriptor: Capabilities descriptor.

        Returns:
            ConstraintSet with loaded constraints.
        """
        constraints: list[Constraint] = []

        constraints_dict = descriptor.get("constraints", {})

        for dim_name, constraint_def in constraints_dict.items():
            constraint = self._parse_constraint(dim_name, constraint_def)
            if constraint:
                constraints.append(constraint)

        logger.debug(
            "Loaded %d constraints from descriptor",
            len(constraints),
        )

        return ConstraintSet(constraints=constraints)

    def load_from_file(self, path: Path) -> ConstraintSet:
        """Load constraints from JSON file.

        Args:
            path: Path to JSON file.

        Returns:
            ConstraintSet with loaded constraints.
        """
        with open(path) as f:
            descriptor = json.load(f)

        logger.info("Loaded constraints from file: %s", path)
        return self.load_from_descriptor(descriptor)

    def _parse_constraint(
        self,
        name: str,
        definition: dict[str, Any],
    ) -> Constraint | None:
        """Parse constraint from definition.

        Args:
            name: Constraint name.
            definition: Constraint definition dict.

        Returns:
            Constraint or None if invalid.
        """
        if not isinstance(definition, dict):
            logger.warning("Invalid constraint definition for %s", name)
            return None

        min_val = definition.get("min")
        max_val = definition.get("max")
        valid_values = definition.get("valid")

        return Constraint(
            name=name,
            min_val=min_val,
            max_val=max_val,
            valid_values=valid_values,
        )


def load_constraints_from_descriptor(
    descriptor: dict[str, Any],
) -> ConstraintSet:
    """Convenience function to load constraints.

    Args:
        descriptor: Capabilities descriptor.

    Returns:
        ConstraintSet with loaded constraints.
    """
    loader = ConstraintLoader()
    return loader.load_from_descriptor(descriptor)
