"""Tests for data-driven constraints."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from typing import Any

from layerzero.capabilities.constraints import (
    Constraint,
    ConstraintLoader,
    ConstraintSet,
    load_constraints_from_descriptor,
)


class TestConstraint:
    """Tests for Constraint dataclass."""

    def test_range_constraint(self) -> None:
        """Range constraint with min and max."""
        constraint = Constraint(
            name="head_dim",
            min_val=32,
            max_val=256,
        )

        assert constraint.name == "head_dim"
        assert constraint.min_val == 32
        assert constraint.max_val == 256
        assert constraint.valid_values is None

    def test_valid_values_constraint(self) -> None:
        """Constraint with explicit valid values."""
        constraint = Constraint(
            name="head_dim",
            valid_values=[32, 64, 128, 256],
        )

        assert constraint.valid_values == [32, 64, 128, 256]

    def test_check_value_in_range(self) -> None:
        """check() validates value in range."""
        constraint = Constraint(name="head_dim", min_val=32, max_val=256)

        assert constraint.check(64) is True
        assert constraint.check(32) is True
        assert constraint.check(256) is True
        assert constraint.check(16) is False
        assert constraint.check(512) is False

    def test_check_value_in_valid_set(self) -> None:
        """check() validates value in valid set."""
        constraint = Constraint(name="head_dim", valid_values=[32, 64, 128])

        assert constraint.check(64) is True
        assert constraint.check(48) is False

    def test_check_prefers_valid_values(self) -> None:
        """valid_values takes precedence over range."""
        constraint = Constraint(
            name="head_dim",
            min_val=0,
            max_val=1000,
            valid_values=[32, 64, 128],
        )

        # 100 is in range but not in valid_values
        assert constraint.check(100) is False
        assert constraint.check(64) is True


class TestConstraintSet:
    """Tests for ConstraintSet."""

    def test_creation(self) -> None:
        """ConstraintSet stores constraints."""
        constraints = [
            Constraint("head_dim", min_val=32, max_val=256),
            Constraint("batch_size", min_val=1, max_val=128),
        ]
        cs = ConstraintSet(constraints=constraints)

        assert len(cs) == 2

    def test_get_constraint(self) -> None:
        """get() returns constraint by name."""
        constraints = [
            Constraint("head_dim", min_val=32, max_val=256),
        ]
        cs = ConstraintSet(constraints=constraints)

        c = cs.get("head_dim")

        assert c is not None
        assert c.name == "head_dim"

    def test_get_missing_constraint(self) -> None:
        """get() returns None for missing constraint."""
        cs = ConstraintSet(constraints=[])

        c = cs.get("head_dim")

        assert c is None

    def test_check_all(self) -> None:
        """check_all() validates multiple values."""
        constraints = [
            Constraint("head_dim", min_val=32, max_val=256),
            Constraint("batch_size", min_val=1, max_val=128),
        ]
        cs = ConstraintSet(constraints=constraints)

        values = {"head_dim": 64, "batch_size": 8}

        passed, errors = cs.check_all(values)

        assert passed is True
        assert errors == []

    def test_check_all_with_failure(self) -> None:
        """check_all() reports failures."""
        constraints = [
            Constraint("head_dim", min_val=32, max_val=256),
        ]
        cs = ConstraintSet(constraints=constraints)

        values = {"head_dim": 512}  # Out of range

        passed, errors = cs.check_all(values)

        assert passed is False
        assert len(errors) > 0

    def test_iteration(self) -> None:
        """ConstraintSet is iterable."""
        constraints = [
            Constraint("head_dim", min_val=32, max_val=256),
            Constraint("batch_size", min_val=1, max_val=128),
        ]
        cs = ConstraintSet(constraints=constraints)

        names = [c.name for c in cs]

        assert "head_dim" in names
        assert "batch_size" in names


class TestConstraintLoader:
    """Tests for ConstraintLoader."""

    def test_load_from_descriptor(self, valid_capabilities_v1) -> None:
        """Load constraints from descriptor."""
        loader = ConstraintLoader()

        constraint_set = loader.load_from_descriptor(valid_capabilities_v1)

        assert constraint_set is not None
        assert len(constraint_set) > 0

    def test_constraints_from_descriptor(self, valid_capabilities_v1) -> None:
        """Constraints loaded from descriptor."""
        loader = ConstraintLoader()

        cs = loader.load_from_descriptor(valid_capabilities_v1)

        # Check head_dim constraint exists
        head_dim = cs.get("head_dim")
        assert head_dim is not None
        assert head_dim.min_val == 32
        assert head_dim.max_val == 256

    def test_load_from_file(self, valid_capabilities_v1) -> None:
        """Load constraints from JSON file."""
        loader = ConstraintLoader()

        # Write descriptor to temp file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(valid_capabilities_v1, f)
            path = Path(f.name)

        try:
            cs = loader.load_from_file(path)

            assert cs is not None
            assert len(cs) > 0
        finally:
            path.unlink(missing_ok=True)

    def test_constraints_update_without_code(self, valid_capabilities_v1) -> None:
        """Constraints can be updated without code changes."""
        loader = ConstraintLoader()

        # Original constraints
        cs1 = loader.load_from_descriptor(valid_capabilities_v1)
        head_dim_max_1 = cs1.get("head_dim").max_val

        # Modify descriptor (simulating file update)
        modified = valid_capabilities_v1.copy()
        modified["constraints"]["head_dim"]["max"] = 512

        # Reload
        cs2 = loader.load_from_descriptor(modified)
        head_dim_max_2 = cs2.get("head_dim").max_val

        # Constraint should have changed
        assert head_dim_max_2 != head_dim_max_1
        assert head_dim_max_2 == 512


class TestLoadConstraintsConvenience:
    """Tests for convenience function."""

    def test_load_constraints_from_descriptor(self, valid_capabilities_v1) -> None:
        """load_constraints_from_descriptor convenience function."""
        cs = load_constraints_from_descriptor(valid_capabilities_v1)

        assert cs is not None
        assert len(cs) > 0
