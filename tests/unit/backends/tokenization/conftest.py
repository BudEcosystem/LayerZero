"""Pytest fixtures for tokenization tests."""
from __future__ import annotations

from pathlib import Path
import tempfile
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for tokenization testing."""
    return [
        "Hello, world!",
        "The quick brown fox jumps over the lazy dog.",
        "GPT-4 is a large language model.",
        "Unicode test: 你好世界 🌍",
        "",  # Empty string edge case
    ]


@pytest.fixture
def simple_text() -> str:
    """Simple text for basic encoding tests."""
    return "Hello, world!"


@pytest.fixture
def unicode_text() -> str:
    """Unicode text for normalization tests."""
    return "café résumé naïve"


@pytest.fixture
def temp_model_dir() -> Generator[Path, None, None]:
    """Temporary directory for model files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_vocab() -> dict[str, int]:
    """Mock vocabulary for testing."""
    return {
        "<s>": 0,
        "</s>": 1,
        "<unk>": 2,
        "<pad>": 3,
        "hello": 4,
        "world": 5,
        "the": 6,
        "quick": 7,
        "brown": 8,
        "fox": 9,
    }


@pytest.fixture
def mock_merges() -> list[tuple[str, str]]:
    """Mock BPE merges for testing."""
    return [
        ("h", "e"),
        ("he", "l"),
        ("hel", "l"),
        ("hell", "o"),
        ("w", "o"),
        ("wo", "r"),
        ("wor", "l"),
        ("worl", "d"),
    ]
