"""
HuggingFace Tokenizers Adapter

Adapter for the HuggingFace tokenizers library (Rust-based, fast).
Supports offset mapping for span alignment and batch encoding.
"""
from __future__ import annotations

import logging
from typing import Any, TYPE_CHECKING

from layerzero.backends.tokenization.base import BaseTokenizerAdapter
from layerzero.backends.tokenization.cache_key import (
    generate_tokenizer_cache_key,
    hash_vocab,
    hash_merges,
    hash_added_tokens,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_tokenizers = None
_tokenizers_available: bool | None = None


def _get_tokenizers() -> Any:
    """Lazy import of tokenizers library."""
    global _tokenizers
    if _tokenizers is None:
        try:
            import tokenizers
            _tokenizers = tokenizers
        except ImportError:
            _tokenizers = False
    return _tokenizers if _tokenizers else None


def is_hf_tokenizers_available() -> bool:
    """Check if HuggingFace tokenizers is available.

    Returns:
        True if tokenizers library is installed.
    """
    global _tokenizers_available
    if _tokenizers_available is None:
        _tokenizers_available = _get_tokenizers() is not None
    return _tokenizers_available


class HFTokenizerAdapter(BaseTokenizerAdapter):
    """Adapter for HuggingFace Tokenizers (Rust-based).

    Provides a unified interface to HuggingFace's fast tokenizers
    with support for offset mapping and batch encoding.

    Thread Safety:
        Thread-safe. The underlying Rust tokenizer handles
        concurrent access safely.

    Example:
        ```python
        adapter = HFTokenizerAdapter(pretrained="bert-base-uncased")
        ids = adapter.encode("Hello, world!")
        text = adapter.decode(ids)

        # With offset mapping
        ids, offsets = adapter.encode_with_offsets("Hello, world!")
        ```
    """

    def __init__(
        self,
        tokenizer_path: str | None = None,
        pretrained: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            tokenizer_path: Path to tokenizer.json file.
            pretrained: Name of pretrained tokenizer to load from Hub.

        Note:
            If neither argument is provided, the adapter will be created
            in an uninitialized state. Encoding/decoding will fail until
            a tokenizer is loaded.
        """
        self._tokenizer = None
        self._tokenizer_path = tokenizer_path
        self._pretrained = pretrained
        self._available = is_hf_tokenizers_available()
        self._cache_key: str | None = None

        if self._available and (tokenizer_path or pretrained):
            self._load_tokenizer()

    def _load_tokenizer(self) -> None:
        """Load the tokenizer from path or pretrained name."""
        tokenizers = _get_tokenizers()
        if tokenizers is None:
            return

        try:
            if self._tokenizer_path:
                self._tokenizer = tokenizers.Tokenizer.from_file(
                    self._tokenizer_path
                )
            elif self._pretrained:
                self._tokenizer = tokenizers.Tokenizer.from_pretrained(
                    self._pretrained
                )
        except Exception as e:
            logger.warning(f"Failed to load tokenizer: {e}")
            self._tokenizer = None

    @property
    def is_loaded(self) -> bool:
        """Check if tokenizer is loaded."""
        return self._tokenizer is not None

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._tokenizer is None:
            return 0
        return self._tokenizer.get_vocab_size()

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token IDs.

        Raises:
            RuntimeError: If tokenizer is not loaded.
        """
        if self._tokenizer is None:
            if not self._available:
                raise RuntimeError(
                    "HuggingFace tokenizers not installed. "
                    "Install with: pip install tokenizers"
                )
            raise RuntimeError("Tokenizer not loaded")

        encoding = self._tokenizer.encode(text)
        return encoding.ids

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text.

        Raises:
            RuntimeError: If tokenizer is not loaded.
        """
        if self._tokenizer is None:
            if not self._available:
                raise RuntimeError(
                    "HuggingFace tokenizers not installed. "
                    "Install with: pip install tokenizers"
                )
            raise RuntimeError("Tokenizer not loaded")

        return self._tokenizer.decode(ids)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Batch encode multiple texts.

        Uses parallel batch encoding for efficiency.

        Args:
            texts: List of texts to encode.

        Returns:
            List of token ID lists.

        Raises:
            RuntimeError: If tokenizer is not loaded.
        """
        if not texts:
            return []

        if self._tokenizer is None:
            if not self._available:
                raise RuntimeError(
                    "HuggingFace tokenizers not installed. "
                    "Install with: pip install tokenizers"
                )
            raise RuntimeError("Tokenizer not loaded")

        encodings = self._tokenizer.encode_batch(texts)
        return [enc.ids for enc in encodings]

    def encode_with_offsets(
        self,
        text: str,
    ) -> tuple[list[int], list[tuple[int, int]]]:
        """Encode text with offset mapping.

        Returns both token IDs and character offset pairs for
        each token, enabling span alignment between original
        text and tokens.

        Args:
            text: Input text to tokenize.

        Returns:
            Tuple of (token IDs, offset pairs).
            Each offset pair is (start, end) character indices.

        Raises:
            RuntimeError: If tokenizer is not loaded.
        """
        if self._tokenizer is None:
            if not self._available:
                raise RuntimeError(
                    "HuggingFace tokenizers not installed. "
                    "Install with: pip install tokenizers"
                )
            raise RuntimeError("Tokenizer not loaded")

        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        offsets = list(encoding.offsets)
        return ids, offsets

    def get_special_tokens(self) -> dict[str, str | None]:
        """Get special tokens mapping.

        Returns:
            Dictionary of special token names to values.
        """
        if self._tokenizer is None:
            return {}

        special_tokens: dict[str, str | None] = {}

        # Try to get common special tokens
        try:
            model = self._tokenizer.model
            if hasattr(model, "unk_token"):
                special_tokens["unk_token"] = model.unk_token
        except Exception:
            pass

        # Check added tokens for special tokens
        try:
            added_tokens = self._tokenizer.get_added_tokens_decoder()
            for token_id, token in added_tokens.items():
                content = token.content if hasattr(token, "content") else str(token)
                if content.startswith("[") or content.startswith("<"):
                    # Likely a special token
                    if "cls" in content.lower():
                        special_tokens["cls_token"] = content
                    elif "sep" in content.lower():
                        special_tokens["sep_token"] = content
                    elif "pad" in content.lower():
                        special_tokens["pad_token"] = content
                    elif "mask" in content.lower():
                        special_tokens["mask_token"] = content
                    elif "bos" in content.lower() or content == "<s>":
                        special_tokens["bos_token"] = content
                    elif "eos" in content.lower() or content == "</s>":
                        special_tokens["eos_token"] = content
        except Exception:
            pass

        return special_tokens

    def get_cache_key(self) -> str:
        """Get tokenizer identity cache key.

        Returns:
            Cache key string based on vocabulary, normalizer,
            and other configuration.
        """
        if self._cache_key is not None:
            return self._cache_key

        if self._tokenizer is None:
            return ""

        try:
            # Get vocabulary hash
            vocab = self._tokenizer.get_vocab()
            vocab_h = hash_vocab(vocab)

            # Get normalizer ID
            normalizer = self._tokenizer.normalizer
            if normalizer is not None:
                normalizer_id = type(normalizer).__name__
            else:
                normalizer_id = "none"

            # Get merges hash (for BPE models)
            merges_h = None
            try:
                model = self._tokenizer.model
                if hasattr(model, "merges"):
                    merges = model.merges
                    if merges:
                        merges_h = hash_merges([tuple(m.split()) for m in merges])
            except Exception:
                pass

            # Get added tokens hash
            added_h = None
            try:
                added_tokens = self._tokenizer.get_added_tokens_decoder()
                if added_tokens:
                    added_dict = {
                        str(k): getattr(v, "content", str(v))
                        for k, v in added_tokens.items()
                    }
                    added_h = hash_added_tokens(
                        {v: int(k) for k, v in added_dict.items()}
                    )
            except Exception:
                pass

            self._cache_key = generate_tokenizer_cache_key(
                vocab_hash=vocab_h,
                normalizer_id=normalizer_id,
                merges_hash=merges_h,
                added_tokens_hash=added_h,
            )
            return self._cache_key

        except Exception as e:
            logger.debug(f"Failed to generate cache key: {e}")
            return ""

    def __repr__(self) -> str:
        """Return string representation."""
        if self._pretrained:
            return f"HFTokenizerAdapter(pretrained={self._pretrained!r})"
        if self._tokenizer_path:
            return f"HFTokenizerAdapter(path={self._tokenizer_path!r})"
        return "HFTokenizerAdapter()"
