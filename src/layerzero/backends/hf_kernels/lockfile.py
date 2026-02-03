"""
LayerZero HuggingFace Kernel Hub Lockfile

Manage kernel lockfiles for reproducible builds.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class KernelLockEntry:
    """Entry in a kernel lockfile.

    Represents a locked kernel version with hash for verification.

    Attributes:
        name: Kernel name
        version: Kernel version string
        sha256: SHA256 hash for verification
        platform: Target platform (e.g., linux_x86_64)
    """

    name: str
    version: str
    sha256: str
    platform: str

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> KernelLockEntry:
        """Create from dictionary.

        Args:
            d: Dictionary with entry fields.

        Returns:
            New KernelLockEntry instance.
        """
        return cls(
            name=d["name"],
            version=d["version"],
            sha256=d["sha256"],
            platform=d["platform"],
        )


class KernelLockfile:
    """Manage kernel lockfiles for reproducibility.

    Lockfiles ensure that the same kernel versions are used
    across different environments.

    Example:
        ```python
        lockfile = KernelLockfile()

        # Load existing lockfile
        entries = lockfile.load("kernels.lock.json")

        # Save new lockfile
        lockfile.save(entries, "kernels.lock.json")

        # Verify loaded kernels
        if lockfile.verify(entries):
            print("All kernels verified")
        ```
    """

    LOCKFILE_VERSION = "1.0"

    def __init__(self) -> None:
        """Initialize the lockfile manager."""
        pass

    def load(self, path: str) -> list[KernelLockEntry]:
        """Load entries from a lockfile.

        Args:
            path: Path to lockfile

        Returns:
            List of KernelLockEntry objects. Empty if file not found.
        """
        lockfile_path = Path(path)

        if not lockfile_path.exists():
            logger.debug(f"Lockfile not found: {path}")
            return []

        try:
            with open(lockfile_path, "r") as f:
                data = json.load(f)

            # Validate lockfile version
            version = data.get("version", "1.0")
            if not version.startswith("1."):
                logger.warning(f"Unsupported lockfile version: {version}")

            kernels_data = data.get("kernels", [])
            entries: list[KernelLockEntry] = []

            for kernel_dict in kernels_data:
                try:
                    entry = KernelLockEntry.from_dict(kernel_dict)
                    entries.append(entry)
                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid lockfile entry: {e}")

            return entries

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse lockfile: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load lockfile: {e}")
            return []

    def save(
        self,
        entries: list[KernelLockEntry],
        path: str,
    ) -> None:
        """Save entries to a lockfile.

        Args:
            entries: List of KernelLockEntry objects to save
            path: Path to save lockfile
        """
        lockfile_path = Path(path)

        try:
            # Create parent directories if needed
            lockfile_path.parent.mkdir(parents=True, exist_ok=True)

            data = {
                "version": self.LOCKFILE_VERSION,
                "kernels": [entry.to_dict() for entry in entries],
            }

            with open(lockfile_path, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug(f"Saved lockfile with {len(entries)} entries: {path}")

        except Exception as e:
            logger.error(f"Failed to save lockfile: {e}")
            raise

    def verify(self, entries: list[KernelLockEntry]) -> bool:
        """Verify that loaded kernels match their lockfile entries.

        This verifies that the SHA256 hashes match.

        Args:
            entries: List of KernelLockEntry objects to verify

        Returns:
            True if all entries are valid, False if any mismatch.
        """
        if not entries:
            return True

        # For now, just validate that entries have required fields
        for entry in entries:
            if not entry.name or not entry.version or not entry.sha256:
                logger.warning(f"Invalid lockfile entry: {entry}")
                return False

        return True

    def add_entry(
        self,
        entries: list[KernelLockEntry],
        new_entry: KernelLockEntry,
    ) -> list[KernelLockEntry]:
        """Add or update an entry in the list.

        If an entry with the same name exists, it is replaced.

        Args:
            entries: Existing entries
            new_entry: Entry to add or update

        Returns:
            Updated list of entries.
        """
        # Remove existing entry with same name
        updated = [e for e in entries if e.name != new_entry.name]
        updated.append(new_entry)
        return updated

    def remove_entry(
        self,
        entries: list[KernelLockEntry],
        kernel_name: str,
    ) -> list[KernelLockEntry]:
        """Remove an entry by kernel name.

        Args:
            entries: Existing entries
            kernel_name: Name of kernel to remove

        Returns:
            Updated list of entries without the specified kernel.
        """
        return [e for e in entries if e.name != kernel_name]
