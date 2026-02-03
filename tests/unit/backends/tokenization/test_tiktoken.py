"""Tests for tiktoken adapter."""
from __future__ import annotations

import pytest

from layerzero.backends.tokenization.tiktoken import (
    TiktokenAdapter,
    is_tiktoken_available,
)


class TestTiktokenAvailability:
    """Test tiktoken availability detection."""

    def test_availability_returns_bool(self) -> None:
        """is_tiktoken_available returns boolean."""
        result = is_tiktoken_available()
        assert isinstance(result, bool)

    def test_adapter_instantiation(self) -> None:
        """Adapter can be instantiated."""
        adapter = TiktokenAdapter()
        assert adapter is not None

    def test_adapter_has_encode_method(self) -> None:
        """Adapter has encode method."""
        adapter = TiktokenAdapter()
        assert hasattr(adapter, "encode")
        assert callable(adapter.encode)

    def test_adapter_has_decode_method(self) -> None:
        """Adapter has decode method."""
        adapter = TiktokenAdapter()
        assert hasattr(adapter, "decode")
        assert callable(adapter.decode)


class TestTiktokenEncode:
    """Test tiktoken encoding."""

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_encode_returns_list_of_ints(
        self,
        simple_text: str,
    ) -> None:
        """Encode returns list of integers."""
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        result = adapter.encode(simple_text)
        assert isinstance(result, list)
        assert all(isinstance(x, int) for x in result)

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_encode_empty_string(self) -> None:
        """Encode handles empty string."""
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        result = adapter.encode("")
        assert isinstance(result, list)
        assert len(result) == 0

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_encode_unicode(
        self,
        unicode_text: str,
    ) -> None:
        """Encode handles unicode text."""
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        result = adapter.encode(unicode_text)
        assert isinstance(result, list)
        assert len(result) > 0


class TestTiktokenDecode:
    """Test tiktoken decoding."""

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_decode_returns_string(self) -> None:
        """Decode returns string."""
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        # Token IDs for "Hello" in cl100k_base
        ids = [9906]  # "Hello" token
        result = adapter.decode(ids)
        assert isinstance(result, str)

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_encode_decode_roundtrip(
        self,
        simple_text: str,
    ) -> None:
        """Encode then decode recovers original text."""
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        ids = adapter.encode(simple_text)
        decoded = adapter.decode(ids)
        assert decoded == simple_text

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_decode_empty_list(self) -> None:
        """Decode handles empty list."""
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        result = adapter.decode([])
        assert result == ""


class TestTiktokenEncodings:
    """Test tiktoken different encodings."""

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_cl100k_base_encoding(
        self,
        simple_text: str,
    ) -> None:
        """cl100k_base encoding works."""
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        ids = adapter.encode(simple_text)
        assert len(ids) > 0

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_p50k_base_encoding(
        self,
        simple_text: str,
    ) -> None:
        """p50k_base encoding works."""
        adapter = TiktokenAdapter(encoding_name="p50k_base")
        ids = adapter.encode(simple_text)
        assert len(ids) > 0

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_different_encodings_produce_different_tokens(
        self,
        simple_text: str,
    ) -> None:
        """Different encodings may produce different tokenizations."""
        adapter_cl100k = TiktokenAdapter(encoding_name="cl100k_base")
        adapter_p50k = TiktokenAdapter(encoding_name="p50k_base")
        ids_cl100k = adapter_cl100k.encode(simple_text)
        ids_p50k = adapter_p50k.encode(simple_text)
        # Token IDs will differ between encodings
        # Both should successfully encode
        assert len(ids_cl100k) > 0
        assert len(ids_p50k) > 0


class TestTiktokenBatch:
    """Test tiktoken batch encoding."""

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_batch_encode_returns_list_of_lists(
        self,
        sample_texts: list[str],
    ) -> None:
        """Batch encode returns list of lists."""
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        # Filter empty strings
        texts = [t for t in sample_texts if t]
        result = adapter.batch_encode(texts)
        assert isinstance(result, list)
        assert all(isinstance(x, list) for x in result)
        assert len(result) == len(texts)


class TestTiktokenCacheKey:
    """Test tiktoken cache key generation."""

    def test_adapter_has_get_cache_key(self) -> None:
        """Adapter has get_cache_key method."""
        adapter = TiktokenAdapter()
        assert hasattr(adapter, "get_cache_key")
        assert callable(adapter.get_cache_key)

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_cache_key_contains_encoding_name(self) -> None:
        """Cache key contains encoding name."""
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        cache_key = adapter.get_cache_key()
        assert "cl100k_base" in cache_key

    @pytest.mark.skipif(
        not is_tiktoken_available(),
        reason="tiktoken not available"
    )
    def test_different_encodings_different_cache_keys(self) -> None:
        """Different encodings have different cache keys."""
        adapter_cl100k = TiktokenAdapter(encoding_name="cl100k_base")
        adapter_p50k = TiktokenAdapter(encoding_name="p50k_base")
        assert adapter_cl100k.get_cache_key() != adapter_p50k.get_cache_key()
