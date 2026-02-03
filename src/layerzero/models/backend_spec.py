"""
LayerZero Backend Specification

Dataclass describing an installed backend library's status and version.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BackendSpec:
    """Backend/library specification.

    Describes an installed backend library's status, version, and capabilities.
    Immutable (frozen) for hashability and thread safety.

    Attributes:
        backend_id: Backend identifier (e.g., "flash_attn")
        version: Version string (e.g., "2.5.6")
        installed: Whether the backend is installed
        healthy: Whether the backend is healthy (no repeated failures)
        import_error: Error message if import failed
        module_name: Python module name
        entry_point: Entry point for plugin registration
        supported_operations: Set of supported operation IDs
        capabilities_schema_version: Capabilities descriptor schema version
    """

    backend_id: str
    version: str
    installed: bool
    healthy: bool
    import_error: str | None
    module_name: str
    entry_point: str | None
    supported_operations: frozenset[str]
    capabilities_schema_version: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for JSON compatibility.

        Returns:
            Dict with backend spec fields.
        """
        return {
            "backend_id": self.backend_id,
            "version": self.version,
            "installed": self.installed,
            "healthy": self.healthy,
            "import_error": self.import_error,
            "module_name": self.module_name,
            "entry_point": self.entry_point,
            "supported_operations": list(self.supported_operations),
            "capabilities_schema_version": self.capabilities_schema_version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "BackendSpec":
        """Deserialize from dictionary.

        Args:
            d: Dict with backend spec fields.

        Returns:
            New BackendSpec instance.
        """
        return cls(
            backend_id=d["backend_id"],
            version=d["version"],
            installed=d["installed"],
            healthy=d["healthy"],
            import_error=d.get("import_error"),
            module_name=d["module_name"],
            entry_point=d.get("entry_point"),
            supported_operations=frozenset(d.get("supported_operations", [])),
            capabilities_schema_version=d.get("capabilities_schema_version", "1.0"),
        )

    @classmethod
    def probe(cls, backend_id: str, module_name: str) -> "BackendSpec":
        """Probe backend by attempting import.

        Attempts to import the module and extract version information.
        Never raises - returns a BackendSpec with installed=False on failure.

        Args:
            backend_id: Backend identifier
            module_name: Python module name to import

        Returns:
            BackendSpec describing the backend's status.
        """
        try:
            import importlib
            module = importlib.import_module(module_name)

            # Try to get version
            version = getattr(module, "__version__", "unknown")

            # Try to get supported operations from capabilities
            supported_ops: frozenset[str] = frozenset()
            if hasattr(module, "LAYERZERO_CAPABILITIES"):
                caps = module.LAYERZERO_CAPABILITIES
                if isinstance(caps, dict) and "ops" in caps:
                    supported_ops = frozenset(caps["ops"].keys())

            return cls(
                backend_id=backend_id,
                version=version,
                installed=True,
                healthy=True,
                import_error=None,
                module_name=module_name,
                entry_point=None,
                supported_operations=supported_ops,
                capabilities_schema_version="1.0",
            )

        except ImportError as e:
            return cls(
                backend_id=backend_id,
                version="unknown",
                installed=False,
                healthy=False,
                import_error=str(e),
                module_name=module_name,
                entry_point=None,
                supported_operations=frozenset(),
                capabilities_schema_version="1.0",
            )
        except Exception as e:
            return cls(
                backend_id=backend_id,
                version="unknown",
                installed=False,
                healthy=False,
                import_error=f"Unexpected error: {e}",
                module_name=module_name,
                entry_point=None,
                supported_operations=frozenset(),
                capabilities_schema_version="1.0",
            )

    def mark_unhealthy(self, error: str) -> "BackendSpec":
        """Return a new BackendSpec marked as unhealthy.

        Args:
            error: Error message describing the failure.

        Returns:
            New BackendSpec with healthy=False.
        """
        # Create dict, modify, and reconstruct (frozen dataclass)
        d = self.to_dict()
        d["healthy"] = False
        d["import_error"] = error
        return BackendSpec.from_dict(d)
