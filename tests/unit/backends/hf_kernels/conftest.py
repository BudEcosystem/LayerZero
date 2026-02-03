"""Pytest fixtures for HuggingFace Kernel Hub tests."""
from __future__ import annotations

import pytest
import tempfile
import json
from pathlib import Path


@pytest.fixture
def temp_lockfile() -> Path:
    """Create a temporary lockfile for testing."""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".json",
        delete=False,
    ) as f:
        lockfile_data = {
            "version": "1.0",
            "kernels": [
                {
                    "name": "flash_attn",
                    "version": "2.6.0",
                    "sha256": "abc123def456",
                    "platform": "linux_x86_64",
                },
                {
                    "name": "triton_kernels",
                    "version": "1.0.0",
                    "sha256": "789ghi012jkl",
                    "platform": "linux_x86_64",
                },
            ],
        }
        json.dump(lockfile_data, f)
        return Path(f.name)


@pytest.fixture
def mock_kernel_dir(tmp_path: Path) -> Path:
    """Create a mock kernel directory for testing."""
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()

    # Create mock kernel files
    (kernel_dir / "flash_attn.so").touch()
    (kernel_dir / "triton_kernels.so").touch()

    return kernel_dir


@pytest.fixture
def sample_lockfile_entries() -> list[dict]:
    """Sample lockfile entries for testing."""
    return [
        {
            "name": "flash_attn",
            "version": "2.6.0",
            "sha256": "abc123def456789",
            "platform": "linux_x86_64",
        },
        {
            "name": "custom_attention",
            "version": "1.2.3",
            "sha256": "def789abc012345",
            "platform": "linux_x86_64",
        },
    ]
