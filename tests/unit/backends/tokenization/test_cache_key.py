"""Tests for tokenizer cache key generation."""
from __future__ import annotations

import pytest

from layerzero.backends.tokenization.cache_key import (
    generate_tokenizer_cache_key,
    hash_vocab,
    hash_merges,
    hash_added_tokens,
)


class TestVocabHash:
    """Test vocabulary hashing."""

    def test_hash_vocab_returns_string(
        self,
        mock_vocab: dict[str, int],
    ) -> None:
        """hash_vocab returns string."""
        result = hash_vocab(mock_vocab)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_vocab_deterministic(
        self,
        mock_vocab: dict[str, int],
    ) -> None:
        """Same vocab produces same hash."""
        hash1 = hash_vocab(mock_vocab)
        hash2 = hash_vocab(mock_vocab)
        assert hash1 == hash2

    def test_hash_vocab_different_for_different_vocabs(
        self,
        mock_vocab: dict[str, int],
    ) -> None:
        """Different vocabs produce different hashes."""
        vocab2 = {**mock_vocab, "extra": 100}
        hash1 = hash_vocab(mock_vocab)
        hash2 = hash_vocab(vocab2)
        assert hash1 != hash2

    def test_hash_vocab_empty(self) -> None:
        """hash_vocab handles empty vocabulary."""
        result = hash_vocab({})
        assert isinstance(result, str)

    def test_hash_vocab_order_independent(self) -> None:
        """Vocab hash is independent of insertion order."""
        vocab1 = {"a": 1, "b": 2, "c": 3}
        vocab2 = {"c": 3, "a": 1, "b": 2}
        assert hash_vocab(vocab1) == hash_vocab(vocab2)


class TestMergesHash:
    """Test BPE merges hashing."""

    def test_hash_merges_returns_string(
        self,
        mock_merges: list[tuple[str, str]],
    ) -> None:
        """hash_merges returns string."""
        result = hash_merges(mock_merges)
        assert isinstance(result, str)
        assert len(result) > 0

    def test_hash_merges_deterministic(
        self,
        mock_merges: list[tuple[str, str]],
    ) -> None:
        """Same merges produce same hash."""
        hash1 = hash_merges(mock_merges)
        hash2 = hash_merges(mock_merges)
        assert hash1 == hash2

    def test_hash_merges_order_matters(
        self,
        mock_merges: list[tuple[str, str]],
    ) -> None:
        """Different merge order produces different hash."""
        reversed_merges = list(reversed(mock_merges))
        hash1 = hash_merges(mock_merges)
        hash2 = hash_merges(reversed_merges)
        assert hash1 != hash2

    def test_hash_merges_empty(self) -> None:
        """hash_merges handles empty list."""
        result = hash_merges([])
        assert isinstance(result, str)

    def test_hash_merges_none(self) -> None:
        """hash_merges handles None."""
        result = hash_merges(None)
        assert result is None or result == ""


class TestAddedTokensHash:
    """Test added tokens hashing."""

    def test_hash_added_tokens_returns_string(self) -> None:
        """hash_added_tokens returns string."""
        added_tokens = {"<mask>": 50264, "<extra_id_0>": 50265}
        result = hash_added_tokens(added_tokens)
        assert isinstance(result, str)

    def test_hash_added_tokens_deterministic(self) -> None:
        """Same added tokens produce same hash."""
        added_tokens = {"<mask>": 50264}
        hash1 = hash_added_tokens(added_tokens)
        hash2 = hash_added_tokens(added_tokens)
        assert hash1 == hash2

    def test_hash_added_tokens_empty(self) -> None:
        """hash_added_tokens handles empty dict."""
        result = hash_added_tokens({})
        assert isinstance(result, str)

    def test_hash_added_tokens_none(self) -> None:
        """hash_added_tokens handles None."""
        result = hash_added_tokens(None)
        assert result is None or result == ""


class TestGenerateCacheKey:
    """Test full cache key generation."""

    def test_generate_cache_key_returns_string(self) -> None:
        """generate_tokenizer_cache_key returns string."""
        result = generate_tokenizer_cache_key(
            vocab_hash="abc123",
            normalizer_id="nfkc",
            merges_hash=None,
            added_tokens_hash=None,
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_generate_cache_key_includes_vocab_hash(self) -> None:
        """Cache key includes vocab hash."""
        vocab_hash = "uniquevocabhash123"
        result = generate_tokenizer_cache_key(
            vocab_hash=vocab_hash,
            normalizer_id="nfkc",
            merges_hash=None,
            added_tokens_hash=None,
        )
        assert vocab_hash in result

    def test_generate_cache_key_includes_normalizer(self) -> None:
        """Cache key includes normalizer ID."""
        result = generate_tokenizer_cache_key(
            vocab_hash="abc123",
            normalizer_id="nfkc_casefold",
            merges_hash=None,
            added_tokens_hash=None,
        )
        assert "nfkc_casefold" in result

    def test_generate_cache_key_with_merges(self) -> None:
        """Cache key includes merges hash when provided."""
        merges_hash = "merges123"
        result = generate_tokenizer_cache_key(
            vocab_hash="abc123",
            normalizer_id="nfkc",
            merges_hash=merges_hash,
            added_tokens_hash=None,
        )
        assert merges_hash in result

    def test_generate_cache_key_with_added_tokens(self) -> None:
        """Cache key includes added tokens hash when provided."""
        added_hash = "added456"
        result = generate_tokenizer_cache_key(
            vocab_hash="abc123",
            normalizer_id="nfkc",
            merges_hash=None,
            added_tokens_hash=added_hash,
        )
        assert added_hash in result

    def test_generate_cache_key_different_normalizers(self) -> None:
        """Different normalizers produce different cache keys."""
        key1 = generate_tokenizer_cache_key(
            vocab_hash="abc123",
            normalizer_id="nfkc",
            merges_hash=None,
            added_tokens_hash=None,
        )
        key2 = generate_tokenizer_cache_key(
            vocab_hash="abc123",
            normalizer_id="nfc",
            merges_hash=None,
            added_tokens_hash=None,
        )
        assert key1 != key2


class TestCacheKeyInvalidation:
    """Test cache key invalidation scenarios."""

    def test_vocab_change_invalidates_cache(
        self,
        mock_vocab: dict[str, int],
    ) -> None:
        """Changing vocab produces different cache key."""
        hash1 = hash_vocab(mock_vocab)
        hash2 = hash_vocab({**mock_vocab, "newtoken": 999})

        key1 = generate_tokenizer_cache_key(
            vocab_hash=hash1,
            normalizer_id="nfkc",
            merges_hash=None,
            added_tokens_hash=None,
        )
        key2 = generate_tokenizer_cache_key(
            vocab_hash=hash2,
            normalizer_id="nfkc",
            merges_hash=None,
            added_tokens_hash=None,
        )
        assert key1 != key2

    def test_merges_change_invalidates_cache(
        self,
        mock_merges: list[tuple[str, str]],
    ) -> None:
        """Changing merges produces different cache key."""
        hash1 = hash_merges(mock_merges)
        hash2 = hash_merges(mock_merges + [("x", "y")])

        key1 = generate_tokenizer_cache_key(
            vocab_hash="abc",
            normalizer_id="nfkc",
            merges_hash=hash1,
            added_tokens_hash=None,
        )
        key2 = generate_tokenizer_cache_key(
            vocab_hash="abc",
            normalizer_id="nfkc",
            merges_hash=hash2,
            added_tokens_hash=None,
        )
        assert key1 != key2

    def test_added_tokens_change_invalidates_cache(self) -> None:
        """Changing added tokens produces different cache key."""
        hash1 = hash_added_tokens({"<mask>": 100})
        hash2 = hash_added_tokens({"<mask>": 100, "<extra>": 101})

        key1 = generate_tokenizer_cache_key(
            vocab_hash="abc",
            normalizer_id="nfkc",
            merges_hash=None,
            added_tokens_hash=hash1,
        )
        key2 = generate_tokenizer_cache_key(
            vocab_hash="abc",
            normalizer_id="nfkc",
            merges_hash=None,
            added_tokens_hash=hash2,
        )
        assert key1 != key2

    def test_normalizer_change_invalidates_cache(self) -> None:
        """Changing normalizer produces different cache key."""
        key1 = generate_tokenizer_cache_key(
            vocab_hash="abc",
            normalizer_id="nfkc",
            merges_hash=None,
            added_tokens_hash=None,
        )
        key2 = generate_tokenizer_cache_key(
            vocab_hash="abc",
            normalizer_id="lowercase",
            merges_hash=None,
            added_tokens_hash=None,
        )
        assert key1 != key2
