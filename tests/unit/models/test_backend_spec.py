"""
Test suite for BackendSpec dataclass.

Tests backend specification and probing.
Following TDD methodology - tests define expected behavior.
"""
import json
import pytest


class TestBackendSpecCreation:
    """Test BackendSpec construction."""

    def test_backend_spec_required_fields(self):
        """BackendSpec must have backend_id, version, installed."""
        from layerzero.models.backend_spec import BackendSpec

        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point="layerzero.backends.flash_attn",
            supported_operations=frozenset(["attention.causal", "attention.full"]),
            capabilities_schema_version="1.0",
        )

        assert spec.backend_id == "flash_attn"
        assert spec.version == "2.5.6"
        assert spec.installed is True
        assert spec.healthy is True

    def test_backend_spec_not_installed(self):
        """BackendSpec can represent not-installed backend."""
        from layerzero.models.backend_spec import BackendSpec

        spec = BackendSpec(
            backend_id="flash_attn",
            version="unknown",
            installed=False,
            healthy=False,
            import_error="ModuleNotFoundError: No module named 'flash_attn'",
            module_name="flash_attn",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )

        assert spec.installed is False
        assert spec.import_error is not None
        assert "ModuleNotFoundError" in spec.import_error

    def test_backend_spec_unhealthy(self):
        """BackendSpec can represent unhealthy backend."""
        from layerzero.models.backend_spec import BackendSpec

        spec = BackendSpec(
            backend_id="xformers",
            version="0.0.25",
            installed=True,
            healthy=False,  # Repeated failures
            import_error=None,
            module_name="xformers",
            entry_point=None,
            supported_operations=frozenset(["attention.causal"]),
            capabilities_schema_version="1.0",
        )

        assert spec.installed is True
        assert spec.healthy is False

    def test_backend_spec_is_frozen(self):
        """BackendSpec is immutable (frozen)."""
        from layerzero.models.backend_spec import BackendSpec

        spec = BackendSpec(
            backend_id="test",
            version="1.0.0",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="test",
            entry_point=None,
            supported_operations=frozenset(),
            capabilities_schema_version="1.0",
        )

        with pytest.raises(AttributeError):
            spec.healthy = False  # type: ignore


class TestBackendSpecProbe:
    """Test BackendSpec.probe() class method."""

    def test_probe_installed_module(self):
        """probe() detects installed module."""
        from layerzero.models.backend_spec import BackendSpec

        # torch should always be installed in test env
        spec = BackendSpec.probe("torch", "torch")

        assert spec.backend_id == "torch"
        assert spec.installed is True
        assert spec.import_error is None

    def test_probe_missing_module(self):
        """probe() handles missing module gracefully."""
        from layerzero.models.backend_spec import BackendSpec

        spec = BackendSpec.probe("nonexistent_module_12345", "nonexistent_module_12345")

        assert spec.installed is False
        assert spec.import_error is not None
        assert "No module named" in spec.import_error or "ModuleNotFoundError" in spec.import_error


class TestBackendSpecSerialization:
    """Test BackendSpec JSON serialization."""

    def test_backend_spec_to_dict(self):
        """BackendSpec can be serialized to dict."""
        from layerzero.models.backend_spec import BackendSpec

        spec = BackendSpec(
            backend_id="flash_attn",
            version="2.5.6",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="flash_attn",
            entry_point="layerzero.backends.flash_attn",
            supported_operations=frozenset(["attention.causal", "attention.full"]),
            capabilities_schema_version="1.0",
        )

        d = spec.to_dict()
        assert d["backend_id"] == "flash_attn"
        assert d["version"] == "2.5.6"
        assert d["installed"] is True
        assert "attention.causal" in d["supported_operations"]

    def test_backend_spec_json_roundtrip(self):
        """BackendSpec serialize/deserialize preserves fields."""
        from layerzero.models.backend_spec import BackendSpec

        original = BackendSpec(
            backend_id="xformers",
            version="0.0.25",
            installed=True,
            healthy=True,
            import_error=None,
            module_name="xformers",
            entry_point=None,
            supported_operations=frozenset(["attention.causal", "attention.full"]),
            capabilities_schema_version="1.0",
        )

        json_str = json.dumps(original.to_dict())
        d = json.loads(json_str)
        restored = BackendSpec.from_dict(d)

        assert restored.backend_id == original.backend_id
        assert restored.version == original.version
        assert restored.installed == original.installed
