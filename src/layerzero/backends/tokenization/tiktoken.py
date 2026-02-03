"""
tiktoken Adapter

Adapter for OpenAI's tiktoken tokenizer library.
Supports multiple encoding schemes (cl100k_base, p50k_base, etc.).
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from layerzero.backends.tokenization.base import BaseTokenizerAdapter
from layerzero.backends.tokenization.cache_key import generate_tiktoken_cache_key

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_tiktoken = None
_tiktoken_available: bool | None = None


def _get_tiktoken() -> Any:
    """Lazy import of tiktoken library."""
    global _tiktoken
    if _tiktoken is None:
        try:
            import tiktoken
            _tiktoken = tiktoken
        except ImportError:
            _tiktoken = False
    return _tiktoken if _tiktoken else None


def is_tiktoken_available() -> bool:
    """Check if tiktoken is available.

    Returns:
        True if tiktoken library is installed.
    """
    global _tiktoken_available
    if _tiktoken_available is None:
        _tiktoken_available = _get_tiktoken() is not None
    return _tiktoken_available


class TiktokenAdapter(BaseTokenizerAdapter):
    """Adapter for tiktoken (OpenAI).

    Provides a unified interface to OpenAI's tiktoken library
    for tokenizing text compatible with OpenAI models.

    Thread Safety:
        Thread-safe. tiktoken's Encoding objects handle
        concurrent access safely.

    Example:
        ```python
        adapter = TiktokenAdapter(encoding_name="cl100k_base")
        ids = adapter.encode("Hello, world!")
        text = adapter.decode(ids)
        ```

    Supported Encodings:
        - cl100k_base: GPT-4, ChatGPT, text-embedding-ada-002
        - p50k_base: Codex, code-davinci-002
        - p50k_edit: text-davinci-edit-001
        - r50k_base: GPT-3, davinci
        - gpt2: GPT-2

    Note:
        For model-specific tokenization, use tiktoken.encoding_for_model()
        directly or specify the model-specific encoding name.
    """

    def __init__(
        self,
        encoding_name: str = "cl100k_base",
    ) -> None:
        """Initialize the adapter.

        Args:
            encoding_name: Name of the tiktoken encoding to use.
                Defaults to "cl100k_base" (GPT-4/ChatGPT encoding).
        """
        self._encoding_name = encoding_name
        self._encoding = None
        self._available = is_tiktoken_available()

        if self._available:
            self._load_encoding()

    def _load_encoding(self) -> None:
        """Load the tiktoken encoding."""
        tiktoken = _get_tiktoken()
        if tiktoken is None:
            return

        try:
            self._encoding = tiktoken.get_encoding(self._encoding_name)
        except Exception as e:
            logger.warning(f"Failed to load tiktoken encoding: {e}")
            self._encoding = None

    @property
    def is_loaded(self) -> bool:
        """Check if encoding is loaded."""
        return self._encoding is not None

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._encoding is None:
            return 0
        return self._encoding.n_vocab

    @property
    def encoding_name(self) -> str:
        """Get encoding name."""
        return self._encoding_name

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token IDs.

        Raises:
            RuntimeError: If encoding is not loaded.
        """
        if self._encoding is None:
            if not self._available:
                raise RuntimeError(
                    "tiktoken not installed. "
                    "Install with: pip install tiktoken"
                )
            raise RuntimeError("Encoding not loaded")

        return self._encoding.encode(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text.

        Raises:
            RuntimeError: If encoding is not loaded.
        """
        if self._encoding is None:
            if not self._available:
                raise RuntimeError(
                    "tiktoken not installed. "
                    "Install with: pip install tiktoken"
                )
            raise RuntimeError("Encoding not loaded")

        return self._encoding.decode(ids)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Batch encode multiple texts.

        tiktoken supports efficient batch encoding via
        encode_batch method.

        Args:
            texts: List of texts to encode.

        Returns:
            List of token ID lists.

        Raises:
            RuntimeError: If encoding is not loaded.
        """
        if not texts:
            return []

        if self._encoding is None:
            if not self._available:
                raise RuntimeError(
                    "tiktoken not installed. "
                    "Install with: pip install tiktoken"
                )
            raise RuntimeError("Encoding not loaded")

        # tiktoken has encode_batch for efficiency
        return self._encoding.encode_batch(texts)

    def encode_with_special_tokens(
        self,
        text: str,
        allowed_special: set[str] | str = "all",
    ) -> list[int]:
        """Encode text allowing special tokens.

        By default, tiktoken raises an error if special tokens
        (like <|endoftext|>) appear in the text. This method
        allows encoding them.

        Args:
            text: Input text to tokenize.
            allowed_special: Set of allowed special tokens,
                or "all" to allow all.

        Returns:
            List of token IDs.

        Raises:
            RuntimeError: If encoding is not loaded.
        """
        if self._encoding is None:
            if not self._available:
                raise RuntimeError(
                    "tiktoken not installed. "
                    "Install with: pip install tiktoken"
                )
            raise RuntimeError("Encoding not loaded")

        return self._encoding.encode(
            text,
            allowed_special=allowed_special,
        )

    def get_special_tokens(self) -> dict[str, int]:
        """Get special tokens mapping.

        Returns:
            Dictionary mapping special token strings to IDs.
        """
        if self._encoding is None:
            return {}

        try:
            return dict(self._encoding._special_tokens)
        except Exception:
            return {}

    def get_cache_key(self) -> str:
        """Get tokenizer identity cache key.

        For tiktoken, the cache key is based on the encoding name
        since encodings are immutable and identified by name.

        Returns:
            Cache key string.
        """
        return generate_tiktoken_cache_key(self._encoding_name)

    def decode_single_token_bytes(self, token_id: int) -> bytes:
        """Decode a single token ID to bytes.

        Useful for debugging or analyzing individual tokens.

        Args:
            token_id: Single token ID.

        Returns:
            Raw bytes for the token.

        Raises:
            RuntimeError: If encoding is not loaded.
        """
        if self._encoding is None:
            if not self._available:
                raise RuntimeError(
                    "tiktoken not installed. "
                    "Install with: pip install tiktoken"
                )
            raise RuntimeError("Encoding not loaded")

        return self._encoding.decode_single_token_bytes(token_id)

    def __repr__(self) -> str:
        """Return string representation."""
        return f"TiktokenAdapter(encoding_name={self._encoding_name!r})"
