"""
Base Tokenizer Adapter

Abstract base class for all tokenizer adapters providing a unified API.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


class BaseTokenizerAdapter(ABC):
    """Base class for tokenizer adapters.

    Provides a unified interface for encoding and decoding text
    across different tokenizer backends (HuggingFace, tiktoken,
    SentencePiece).

    Thread Safety:
        Implementations must be thread-safe. The underlying tokenizer
        libraries (tokenizers, tiktoken, sentencepiece) all provide
        thread-safe encode/decode operations.

    Example:
        ```python
        adapter = SomeTokenizerAdapter()
        ids = adapter.encode("Hello, world!")
        text = adapter.decode(ids)
        ```
    """

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token IDs.
        """

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs to decode.

        Returns:
            Decoded text string.
        """

    @abstractmethod
    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Batch encode multiple texts.

        More efficient than calling encode() in a loop for backends
        that support parallel batch encoding.

        Args:
            texts: List of input texts to tokenize.

        Returns:
            List of token ID lists, one per input text.
        """

    @abstractmethod
    def get_cache_key(self) -> str:
        """Get tokenizer identity cache key.

        Returns a stable string that uniquely identifies this
        tokenizer's configuration. Used for caching tokenized
        outputs.

        The cache key should include:
        - Vocabulary hash
        - Normalizer configuration
        - BPE merges hash (if applicable)
        - Added tokens hash (if applicable)
        - Special tokens configuration

        Returns:
            Cache key string.
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Get vocabulary size.

        Returns:
            Number of tokens in the vocabulary.
        """

    def batch_decode(self, id_lists: list[list[int]]) -> list[str]:
        """Batch decode multiple token ID sequences.

        Default implementation calls decode() in a loop.
        Subclasses can override for more efficient batch decoding.

        Args:
            id_lists: List of token ID lists to decode.

        Returns:
            List of decoded text strings.
        """
        return [self.decode(ids) for ids in id_lists]

    def count_tokens(self, text: str) -> int:
        """Count number of tokens in text.

        Convenience method that encodes and returns the length.

        Args:
            text: Input text.

        Returns:
            Number of tokens.
        """
        return len(self.encode(text))

    def truncate(
        self,
        text: str,
        max_tokens: int,
        *,
        add_special_tokens: bool = True,
    ) -> str:
        """Truncate text to maximum number of tokens.

        Args:
            text: Input text.
            max_tokens: Maximum number of tokens.
            add_special_tokens: Whether to account for special tokens.

        Returns:
            Truncated and decoded text.
        """
        ids = self.encode(text)
        if len(ids) <= max_tokens:
            return text
        truncated_ids = ids[:max_tokens]
        return self.decode(truncated_ids)
