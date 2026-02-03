"""
SentencePiece Adapter

Adapter for Google's SentencePiece tokenizer library.
Used by many legacy models (LLaMA, T5, etc.).
"""
from __future__ import annotations

import logging
import unicodedata
from typing import Any, TYPE_CHECKING

from layerzero.backends.tokenization.base import BaseTokenizerAdapter
from layerzero.backends.tokenization.cache_key import (
    generate_sentencepiece_cache_key,
    hash_file,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# Lazy import for optional dependency
_sentencepiece = None
_sentencepiece_available: bool | None = None


def _get_sentencepiece() -> Any:
    """Lazy import of sentencepiece library."""
    global _sentencepiece
    if _sentencepiece is None:
        try:
            import sentencepiece
            _sentencepiece = sentencepiece
        except ImportError:
            _sentencepiece = False
    return _sentencepiece if _sentencepiece else None


def is_sentencepiece_available() -> bool:
    """Check if SentencePiece is available.

    Returns:
        True if sentencepiece library is installed.
    """
    global _sentencepiece_available
    if _sentencepiece_available is None:
        _sentencepiece_available = _get_sentencepiece() is not None
    return _sentencepiece_available


class SentencePieceAdapter(BaseTokenizerAdapter):
    """Adapter for SentencePiece.

    Provides a unified interface to Google's SentencePiece library
    for subword tokenization used by many legacy models.

    Thread Safety:
        Thread-safe. SentencePieceProcessor handles concurrent
        access safely.

    Example:
        ```python
        adapter = SentencePieceAdapter()
        adapter.load("model.spm")
        ids = adapter.encode("Hello, world!")
        text = adapter.decode(ids)
        ```

    Note:
        Unlike HuggingFace and tiktoken adapters, SentencePiece
        requires explicitly loading a model file before use.
    """

    def __init__(
        self,
        model_path: str | None = None,
    ) -> None:
        """Initialize the adapter.

        Args:
            model_path: Optional path to .model file.
                If provided, loads the model immediately.
        """
        self._model_path: str | None = None
        self._processor = None
        self._available = is_sentencepiece_available()
        self._model_hash: str | None = None

        if model_path and self._available:
            self.load(model_path)

    def load(self, model_path: str) -> None:
        """Load a SentencePiece model.

        Args:
            model_path: Path to .model file.

        Raises:
            RuntimeError: If SentencePiece is not installed.
            FileNotFoundError: If model file does not exist.
        """
        if not self._available:
            raise RuntimeError(
                "SentencePiece not installed. "
                "Install with: pip install sentencepiece"
            )

        sp = _get_sentencepiece()
        if sp is None:
            raise RuntimeError("SentencePiece import failed")

        self._processor = sp.SentencePieceProcessor()

        try:
            self._processor.Load(model_path)
            self._model_path = model_path
            self._model_hash = hash_file(model_path)
        except Exception as e:
            self._processor = None
            self._model_path = None
            self._model_hash = None
            if "No such file" in str(e) or "not found" in str(e).lower():
                raise FileNotFoundError(f"Model file not found: {model_path}")
            raise RuntimeError(f"Failed to load SentencePiece model: {e}")

    @property
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self._processor is not None

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        if self._processor is None:
            return 0
        return self._processor.GetPieceSize()

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Args:
            text: Input text to tokenize.

        Returns:
            List of token IDs.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._processor is None:
            raise RuntimeError(
                "SentencePiece model not loaded. "
                "Call load(model_path) first."
            )

        return self._processor.EncodeAsIds(text)

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text.

        Args:
            ids: List of token IDs.

        Returns:
            Decoded text.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._processor is None:
            raise RuntimeError(
                "SentencePiece model not loaded. "
                "Call load(model_path) first."
            )

        return self._processor.DecodeIds(ids)

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Batch encode multiple texts.

        SentencePiece processes each text sequentially in this
        implementation. For truly parallel batch encoding,
        consider using the C++ API directly.

        Args:
            texts: List of texts to encode.

        Returns:
            List of token ID lists.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if not texts:
            return []

        if self._processor is None:
            raise RuntimeError(
                "SentencePiece model not loaded. "
                "Call load(model_path) first."
            )

        return [self._processor.EncodeAsIds(text) for text in texts]

    def encode_as_pieces(self, text: str) -> list[str]:
        """Encode text to subword pieces.

        Returns the actual subword strings instead of IDs.

        Args:
            text: Input text to tokenize.

        Returns:
            List of subword piece strings.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._processor is None:
            raise RuntimeError(
                "SentencePiece model not loaded. "
                "Call load(model_path) first."
            )

        return self._processor.EncodeAsPieces(text)

    def normalize(self, text: str) -> str:
        """Normalize text using NFKC.

        SentencePiece models typically expect NFKC-normalized input.
        This method provides standalone normalization without a
        loaded model.

        Args:
            text: Input text to normalize.

        Returns:
            NFKC-normalized text.
        """
        return unicodedata.normalize("NFKC", text)

    def get_special_tokens(self) -> dict[str, int | None]:
        """Get special token IDs.

        Returns:
            Dictionary mapping special token names to IDs.
        """
        if self._processor is None:
            return {}

        special_tokens: dict[str, int | None] = {}

        try:
            # Standard SentencePiece special tokens
            special_tokens["bos_token_id"] = self._processor.bos_id()
            special_tokens["eos_token_id"] = self._processor.eos_id()
            special_tokens["pad_token_id"] = self._processor.pad_id()
            special_tokens["unk_token_id"] = self._processor.unk_id()
        except Exception:
            pass

        # Filter out -1 (not set) values
        return {k: v for k, v in special_tokens.items() if v != -1}

    def piece_to_id(self, piece: str) -> int:
        """Convert a piece string to its ID.

        Args:
            piece: Subword piece string.

        Returns:
            Token ID.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._processor is None:
            raise RuntimeError(
                "SentencePiece model not loaded. "
                "Call load(model_path) first."
            )

        return self._processor.PieceToId(piece)

    def id_to_piece(self, token_id: int) -> str:
        """Convert a token ID to its piece string.

        Args:
            token_id: Token ID.

        Returns:
            Subword piece string.

        Raises:
            RuntimeError: If model is not loaded.
        """
        if self._processor is None:
            raise RuntimeError(
                "SentencePiece model not loaded. "
                "Call load(model_path) first."
            )

        return self._processor.IdToPiece(token_id)

    def get_cache_key(self) -> str:
        """Get tokenizer identity cache key.

        For SentencePiece, the cache key is based on the model
        file hash to detect any changes.

        Returns:
            Cache key string, or indicator if no model loaded.
        """
        if self._model_hash is None:
            return "no_model"

        # Get normalizer spec from model config
        normalizer_spec = "default"
        if self._processor is not None:
            try:
                # Check if normalization is enabled
                normalizer_spec = "nfkc"  # SentencePiece typically uses NFKC
            except Exception:
                pass

        return generate_sentencepiece_cache_key(
            model_hash=self._model_hash,
            normalizer_spec=normalizer_spec,
        )

    def __repr__(self) -> str:
        """Return string representation."""
        if self._model_path:
            return f"SentencePieceAdapter(model_path={self._model_path!r})"
        return "SentencePieceAdapter()"
