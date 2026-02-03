"""Tests for HuggingFace Tokenizers adapter."""
from __future__ import annotations

import pytest

from layerzero.backends.tokenization.hf_tokenizers import (
    HFTokenizerAdapter,
    is_hf_tokenizers_available,
)


class TestHFTokenizersAvailability:
    """Test HuggingFace Tokenizers availability detection."""

    def test_availability_returns_bool(self) -> None:
        """is_hf_tokenizers_available returns boolean."""
        result = is_hf_tokenizers_available()
        assert isinstance(result, bool)

    def test_adapter_instantiation(self) -> None:
        """Adapter can be instantiated."""
        adapter = HFTokenizerAdapter()
        assert adapter is not None

    def test_adapter_has_encode_method(self) -> None:
        """Adapter has encode method."""
        adapter = HFTokenizerAdapter()
        assert hasattr(adapter, "encode")
        assert callable(adapter.encode)

    def test_adapter_has_decode_method(self) -> None:
        """Adapter has decode method."""
        adapter = HFTokenizerAdapter()
        assert hasattr(adapter, "decode")
        assert callable(adapter.decode)


class TestHFTokenizersEncode:
    """Test HuggingFace Tokenizers encoding."""

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_encode_returns_list_of_ints(
        self,
        simple_text: str,
    ) -> None:
        """Encode returns list of integers."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        result = adapter.encode(simple_text)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_encode_empty_string(self) -> None:
        """Encode handles empty string."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        result = adapter.encode("")
        assert isinstance(result, list)


class TestHFTokenizersDecode:
    """Test HuggingFace Tokenizers decoding."""

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_decode_returns_string(self) -> None:
        """Decode returns string."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        ids = [101, 7592, 102]  # [CLS] hello [SEP]
        result = adapter.decode(ids)
        assert isinstance(result, str)

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_encode_decode_roundtrip(
        self,
        simple_text: str,
    ) -> None:
        """Encode then decode recovers original text (approximately)."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        ids = adapter.encode(simple_text)
        decoded = adapter.decode(ids)
        # BERT adds special tokens and lowercases, so we check normalized form
        assert "hello" in decoded.lower()


class TestHFTokenizersBatch:
    """Test HuggingFace Tokenizers batch encoding."""

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_batch_encode_returns_list_of_lists(
        self,
        sample_texts: list[str],
    ) -> None:
        """Batch encode returns list of lists."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        # Filter empty strings for batch encoding
        texts = [t for t in sample_texts if t]
        result = adapter.batch_encode(texts)
        assert isinstance(result, list)
        assert all(isinstance(x, list) for x in result)
        assert len(result) == len(texts)

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_batch_encode_empty_list(self) -> None:
        """Batch encode handles empty list."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        result = adapter.batch_encode([])
        assert result == []


class TestHFTokenizersOffsets:
    """Test HuggingFace Tokenizers offset mapping."""

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_encode_with_offsets(
        self,
        simple_text: str,
    ) -> None:
        """Encode with offsets returns ids and offset mapping."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        ids, offsets = adapter.encode_with_offsets(simple_text)
        assert isinstance(ids, list)
        assert isinstance(offsets, list)
        assert len(ids) == len(offsets)
        # Each offset is a tuple of (start, end)
        for offset in offsets:
            assert isinstance(offset, tuple)
            assert len(offset) == 2


class TestHFTokenizersSpecialTokens:
    """Test HuggingFace Tokenizers special token handling."""

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_get_special_tokens(self) -> None:
        """Adapter exposes special tokens."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        special_tokens = adapter.get_special_tokens()
        assert isinstance(special_tokens, dict)
        # BERT should have at least CLS and SEP
        assert "cls_token" in special_tokens or "bos_token" in special_tokens

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_vocab_size(self) -> None:
        """Adapter reports vocabulary size."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        vocab_size = adapter.vocab_size
        assert isinstance(vocab_size, int)
        assert vocab_size > 0


class TestHFTokenizersCacheKey:
    """Test HuggingFace Tokenizers cache key generation."""

    def test_adapter_has_get_cache_key(self) -> None:
        """Adapter has get_cache_key method."""
        adapter = HFTokenizerAdapter()
        assert hasattr(adapter, "get_cache_key")
        assert callable(adapter.get_cache_key)

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_cache_key_returns_string(self) -> None:
        """Cache key returns non-empty string."""
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        cache_key = adapter.get_cache_key()
        assert isinstance(cache_key, str)
        assert len(cache_key) > 0

    @pytest.mark.skipif(
        not is_hf_tokenizers_available(),
        reason="HuggingFace tokenizers not available"
    )
    def test_same_tokenizer_same_cache_key(self) -> None:
        """Same tokenizer produces same cache key."""
        adapter1 = HFTokenizerAdapter(pretrained="bert-base-uncased")
        adapter2 = HFTokenizerAdapter(pretrained="bert-base-uncased")
        assert adapter1.get_cache_key() == adapter2.get_cache_key()
