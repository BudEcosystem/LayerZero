"""Tests for HuggingFace Kernel Hub lockfile handling."""
from __future__ import annotations

from pathlib import Path
import tempfile
import json
import pytest

from layerzero.backends.hf_kernels.lockfile import (
    KernelLockEntry,
    KernelLockfile,
)


class TestKernelLockEntry:
    """Test KernelLockEntry dataclass."""

    def test_entry_creation(self) -> None:
        """KernelLockEntry can be created."""
        entry = KernelLockEntry(
            name="flash_attn",
            version="2.6.0",
            sha256="abc123def456",
            platform="linux_x86_64",
        )
        assert entry.name == "flash_attn"
        assert entry.version == "2.6.0"

    def test_entry_has_all_fields(self) -> None:
        """Entry has all required fields."""
        entry = KernelLockEntry(
            name="test_kernel",
            version="1.0.0",
            sha256="hash123",
            platform="linux_x86_64",
        )
        assert hasattr(entry, "name")
        assert hasattr(entry, "version")
        assert hasattr(entry, "sha256")
        assert hasattr(entry, "platform")

    def test_entry_to_dict(self) -> None:
        """Entry can be converted to dict."""
        entry = KernelLockEntry(
            name="test_kernel",
            version="1.0.0",
            sha256="hash123",
            platform="linux_x86_64",
        )
        d = entry.to_dict()
        assert isinstance(d, dict)
        assert d["name"] == "test_kernel"
        assert d["version"] == "1.0.0"

    def test_entry_from_dict(self) -> None:
        """Entry can be created from dict."""
        d = {
            "name": "test_kernel",
            "version": "1.0.0",
            "sha256": "hash123",
            "platform": "linux_x86_64",
        }
        entry = KernelLockEntry.from_dict(d)
        assert entry.name == "test_kernel"
        assert entry.version == "1.0.0"


class TestKernelLockfile:
    """Test KernelLockfile class."""

    def test_lockfile_instantiation(self) -> None:
        """KernelLockfile can be instantiated."""
        lockfile = KernelLockfile()
        assert lockfile is not None

    def test_lockfile_has_load_method(self) -> None:
        """Lockfile has load method."""
        lockfile = KernelLockfile()
        assert hasattr(lockfile, "load")
        assert callable(lockfile.load)

    def test_lockfile_has_save_method(self) -> None:
        """Lockfile has save method."""
        lockfile = KernelLockfile()
        assert hasattr(lockfile, "save")
        assert callable(lockfile.save)

    def test_lockfile_has_verify_method(self) -> None:
        """Lockfile has verify method."""
        lockfile = KernelLockfile()
        assert hasattr(lockfile, "verify")
        assert callable(lockfile.verify)


class TestLockfileOperations:
    """Test lockfile load/save operations."""

    def test_load_valid_lockfile(
        self,
        temp_lockfile: Path,
    ) -> None:
        """Load valid lockfile."""
        lockfile = KernelLockfile()
        entries = lockfile.load(str(temp_lockfile))
        assert isinstance(entries, list)
        assert len(entries) == 2

    def test_load_entries_have_correct_type(
        self,
        temp_lockfile: Path,
    ) -> None:
        """Loaded entries are KernelLockEntry instances."""
        lockfile = KernelLockfile()
        entries = lockfile.load(str(temp_lockfile))
        for entry in entries:
            assert isinstance(entry, KernelLockEntry)

    def test_load_nonexistent_lockfile(self) -> None:
        """Loading nonexistent lockfile returns empty list."""
        lockfile = KernelLockfile()
        entries = lockfile.load("/nonexistent/lockfile.json")
        assert entries == []

    def test_save_lockfile(self, tmp_path: Path) -> None:
        """Save lockfile to disk."""
        lockfile = KernelLockfile()
        entries = [
            KernelLockEntry(
                name="test_kernel",
                version="1.0.0",
                sha256="hash123",
                platform="linux_x86_64",
            ),
        ]

        output_path = tmp_path / "test_lockfile.json"
        lockfile.save(entries, str(output_path))

        assert output_path.exists()

    def test_round_trip_lockfile(self, tmp_path: Path) -> None:
        """Save and load produces identical entries."""
        lockfile = KernelLockfile()
        original_entries = [
            KernelLockEntry(
                name="flash_attn",
                version="2.6.0",
                sha256="abc123def456",
                platform="linux_x86_64",
            ),
            KernelLockEntry(
                name="triton_kernels",
                version="1.0.0",
                sha256="789ghi012jkl",
                platform="linux_x86_64",
            ),
        ]

        output_path = tmp_path / "roundtrip_lockfile.json"
        lockfile.save(original_entries, str(output_path))
        loaded_entries = lockfile.load(str(output_path))

        assert len(loaded_entries) == len(original_entries)
        for orig, loaded in zip(original_entries, loaded_entries):
            assert orig.name == loaded.name
            assert orig.version == loaded.version
            assert orig.sha256 == loaded.sha256


class TestLockfileVerification:
    """Test lockfile verification."""

    def test_verify_returns_bool(self) -> None:
        """verify returns boolean."""
        lockfile = KernelLockfile()
        entries = [
            KernelLockEntry(
                name="test",
                version="1.0.0",
                sha256="hash",
                platform="linux_x86_64",
            ),
        ]
        result = lockfile.verify(entries)
        assert isinstance(result, bool)

    def test_verify_empty_entries(self) -> None:
        """Empty entries verify as True."""
        lockfile = KernelLockfile()
        result = lockfile.verify([])
        assert result is True
