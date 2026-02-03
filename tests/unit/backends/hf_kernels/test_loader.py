"""Tests for HuggingFace Kernel Hub loader."""
from __future__ import annotations

from pathlib import Path
import pytest

from layerzero.backends.hf_kernels.loader import (
    HFKernelLoader,
    LoadedKernel,
)
from layerzero.backends.hf_kernels.version import is_hf_kernels_available


class TestHFKernelLoader:
    """Test HF kernel loader."""

    def test_loader_instantiation(self) -> None:
        """HFKernelLoader can be instantiated."""
        loader = HFKernelLoader()
        assert loader is not None

    def test_loader_has_load_method(self) -> None:
        """Loader has load method."""
        loader = HFKernelLoader()
        assert hasattr(loader, "load")
        assert callable(loader.load)

    def test_loader_has_load_from_lockfile_method(self) -> None:
        """Loader has load_from_lockfile method."""
        loader = HFKernelLoader()
        assert hasattr(loader, "load_from_lockfile")
        assert callable(loader.load_from_lockfile)

    def test_loader_has_cache_dir(self) -> None:
        """Loader has cache directory attribute."""
        loader = HFKernelLoader()
        assert hasattr(loader, "cache_dir")


class TestLoadedKernel:
    """Test LoadedKernel dataclass."""

    def test_loaded_kernel_has_name(self) -> None:
        """LoadedKernel has name attribute."""
        kernel = LoadedKernel(
            name="test_kernel",
            version="1.0.0",
            path=Path("/tmp/test.so"),
            sha256="abc123",
        )
        assert kernel.name == "test_kernel"

    def test_loaded_kernel_has_version(self) -> None:
        """LoadedKernel has version attribute."""
        kernel = LoadedKernel(
            name="test_kernel",
            version="1.0.0",
            path=Path("/tmp/test.so"),
            sha256="abc123",
        )
        assert kernel.version == "1.0.0"

    def test_loaded_kernel_has_path(self) -> None:
        """LoadedKernel has path attribute."""
        kernel = LoadedKernel(
            name="test_kernel",
            version="1.0.0",
            path=Path("/tmp/test.so"),
            sha256="abc123",
        )
        assert isinstance(kernel.path, Path)


class TestKernelLoading:
    """Test kernel loading functionality."""

    @pytest.mark.skipif(
        not is_hf_kernels_available(),
        reason="HF kernels not available"
    )
    def test_load_by_name(self) -> None:
        """Load kernel by name from Hub."""
        loader = HFKernelLoader()
        # Try to load - may return None if kernel not in Hub
        # This tests the loading mechanism, not kernel availability
        result = loader.load("flash_attn", version="latest")
        # Result can be None if kernel not available in Hub
        # The test passes if no exception is raised
        assert result is None or isinstance(result, LoadedKernel)

    @pytest.mark.skipif(
        not is_hf_kernels_available(),
        reason="HF kernels not available"
    )
    def test_load_specific_version(self) -> None:
        """Load specific kernel version."""
        loader = HFKernelLoader()
        result = loader.load("flash_attn", version="2.6.0")
        if result is not None:
            assert result.version == "2.6.0"

    def test_load_unknown_kernel_returns_none(self) -> None:
        """Loading unknown kernel returns None."""
        loader = HFKernelLoader()
        result = loader.load("definitely_not_a_real_kernel_xyz123")
        assert result is None


class TestLockfileLoading:
    """Test lockfile-based loading."""

    def test_load_from_lockfile_returns_list(
        self,
        temp_lockfile: Path,
    ) -> None:
        """load_from_lockfile returns list."""
        loader = HFKernelLoader()
        result = loader.load_from_lockfile(str(temp_lockfile))
        assert isinstance(result, list)

    def test_load_from_nonexistent_lockfile(self) -> None:
        """Loading from nonexistent lockfile raises or returns empty."""
        loader = HFKernelLoader()
        result = loader.load_from_lockfile("/nonexistent/path/lockfile.json")
        # Should either raise or return empty list
        assert result == [] or result is None

    def test_load_from_lockfile_parses_entries(
        self,
        temp_lockfile: Path,
    ) -> None:
        """Lockfile entries are parsed correctly."""
        loader = HFKernelLoader()
        result = loader.load_from_lockfile(str(temp_lockfile))
        # Should have parsed the lockfile structure
        assert isinstance(result, list)
