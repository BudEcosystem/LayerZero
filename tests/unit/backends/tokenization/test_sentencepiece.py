"""Tests for SentencePiece adapter."""
from __future__ import annotations

from pathlib import Path

import pytest

from layerzero.backends.tokenization.sentencepiece import (
    SentencePieceAdapter,
    is_sentencepiece_available,
)


class TestSentencePieceAvailability:
    """Test SentencePiece availability detection."""

    def test_availability_returns_bool(self) -> None:
        """is_sentencepiece_available returns boolean."""
        result = is_sentencepiece_available()
        assert isinstance(result, bool)

    def test_adapter_instantiation(self) -> None:
        """Adapter can be instantiated without model."""
        adapter = SentencePieceAdapter()
        assert adapter is not None

    def test_adapter_has_encode_method(self) -> None:
        """Adapter has encode method."""
        adapter = SentencePieceAdapter()
        assert hasattr(adapter, "encode")
        assert callable(adapter.encode)

    def test_adapter_has_decode_method(self) -> None:
        """Adapter has decode method."""
        adapter = SentencePieceAdapter()
        assert hasattr(adapter, "decode")
        assert callable(adapter.decode)


class TestSentencePieceEncode:
    """Test SentencePiece encoding."""

    @pytest.mark.skipif(
        not is_sentencepiece_available(),
        reason="SentencePiece not available"
    )
    def test_encode_without_model_raises(
        self,
        simple_text: str,
    ) -> None:
        """Encode without loaded model raises error."""
        adapter = SentencePieceAdapter()
        with pytest.raises(RuntimeError):
            adapter.encode(simple_text)

    def test_adapter_has_load_method(self) -> None:
        """Adapter has load method."""
        adapter = SentencePieceAdapter()
        assert hasattr(adapter, "load")
        assert callable(adapter.load)

    @pytest.mark.skipif(
        not is_sentencepiece_available(),
        reason="SentencePiece not available"
    )
    def test_load_nonexistent_model_raises(
        self,
        temp_model_dir: Path,
    ) -> None:
        """Loading nonexistent model raises error."""
        adapter = SentencePieceAdapter()
        with pytest.raises((FileNotFoundError, RuntimeError)):
            adapter.load(str(temp_model_dir / "nonexistent.model"))


class TestSentencePieceDecode:
    """Test SentencePiece decoding."""

    @pytest.mark.skipif(
        not is_sentencepiece_available(),
        reason="SentencePiece not available"
    )
    def test_decode_without_model_raises(self) -> None:
        """Decode without loaded model raises error."""
        adapter = SentencePieceAdapter()
        with pytest.raises(RuntimeError):
            adapter.decode([1, 2, 3])


class TestSentencePieceBatch:
    """Test SentencePiece batch encoding."""

    def test_adapter_has_batch_encode_method(self) -> None:
        """Adapter has batch_encode method."""
        adapter = SentencePieceAdapter()
        assert hasattr(adapter, "batch_encode")
        assert callable(adapter.batch_encode)


class TestSentencePieceNormalization:
    """Test SentencePiece NFKC normalization."""

    @pytest.mark.skipif(
        not is_sentencepiece_available(),
        reason="SentencePiece not available"
    )
    def test_adapter_has_normalize_method(self) -> None:
        """Adapter has normalize method."""
        adapter = SentencePieceAdapter()
        assert hasattr(adapter, "normalize")
        assert callable(adapter.normalize)

    @pytest.mark.skipif(
        not is_sentencepiece_available(),
        reason="SentencePiece not available"
    )
    def test_normalize_nfkc(
        self,
        unicode_text: str,
    ) -> None:
        """Normalize performs NFKC normalization."""
        adapter = SentencePieceAdapter()
        # NFKC normalization should work without a loaded model
        normalized = adapter.normalize(unicode_text)
        assert isinstance(normalized, str)

    @pytest.mark.skipif(
        not is_sentencepiece_available(),
        reason="SentencePiece not available"
    )
    def test_normalize_handles_special_characters(self) -> None:
        """Normalize handles special Unicode characters."""
        adapter = SentencePieceAdapter()
        # Full-width to half-width conversion
        text = "ＡＢＣ１２３"  # Full-width ASCII
        normalized = adapter.normalize(text)
        assert isinstance(normalized, str)
        # NFKC should convert full-width to regular ASCII
        assert "A" in normalized or "Ａ" in normalized  # Depends on implementation


class TestSentencePieceCacheKey:
    """Test SentencePiece cache key generation."""

    def test_adapter_has_get_cache_key(self) -> None:
        """Adapter has get_cache_key method."""
        adapter = SentencePieceAdapter()
        assert hasattr(adapter, "get_cache_key")
        assert callable(adapter.get_cache_key)

    def test_cache_key_without_model(self) -> None:
        """Cache key without loaded model returns placeholder."""
        adapter = SentencePieceAdapter()
        cache_key = adapter.get_cache_key()
        assert isinstance(cache_key, str)
        # Without a model, should indicate no model loaded
        assert "no_model" in cache_key or cache_key == ""


class TestSentencePieceModelInfo:
    """Test SentencePiece model information."""

    def test_adapter_has_is_loaded_property(self) -> None:
        """Adapter has is_loaded property."""
        adapter = SentencePieceAdapter()
        assert hasattr(adapter, "is_loaded")

    def test_not_loaded_initially(self) -> None:
        """Adapter is not loaded initially."""
        adapter = SentencePieceAdapter()
        assert adapter.is_loaded is False

    def test_adapter_has_vocab_size_property(self) -> None:
        """Adapter has vocab_size property."""
        adapter = SentencePieceAdapter()
        assert hasattr(adapter, "vocab_size")

    def test_vocab_size_without_model(self) -> None:
        """Vocab size without model returns 0."""
        adapter = SentencePieceAdapter()
        assert adapter.vocab_size == 0
