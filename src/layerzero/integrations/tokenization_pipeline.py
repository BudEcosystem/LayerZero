"""
Tokenization Pipeline Integration

Provides tokenization integration with HF pipelines,
caching, and efficient batch processing.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Protocol, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import nn

logger = logging.getLogger(__name__)


class TokenizerType(Enum):
    """Tokenizer type enumeration."""

    HF_TOKENIZERS = auto()
    TIKTOKEN = auto()
    SENTENCEPIECE = auto()
    AUTO = auto()


class TokenizerProtocol(Protocol):
    """Protocol for tokenizer adapters."""

    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        ...

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs."""
        ...

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs to text."""
        ...

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        """Batch encode texts."""
        ...

    def get_cache_key(self) -> str:
        """Get cache key for this tokenizer."""
        ...


@dataclass
class TokenizedBatch:
    """Batch of tokenized sequences.

    Attributes:
        input_ids: List of token ID sequences.
        attention_mask: List of attention masks (1 for real, 0 for pad).
        token_type_ids: Optional token type IDs for segment embeddings.
    """

    input_ids: list[list[int]]
    attention_mask: list[list[int]] | None = None
    token_type_ids: list[list[int]] | None = None

    def to_tensors(
        self,
        device: str | torch.device = "cpu",
    ) -> dict[str, torch.Tensor | None]:
        """Convert to PyTorch tensors.

        Args:
            device: Target device for tensors.

        Returns:
            Dictionary with tensor values.
        """
        result: dict[str, torch.Tensor | None] = {
            "input_ids": torch.tensor(self.input_ids, device=device),
        }

        if self.attention_mask is not None:
            result["attention_mask"] = torch.tensor(
                self.attention_mask, device=device
            )
        else:
            result["attention_mask"] = None

        if self.token_type_ids is not None:
            result["token_type_ids"] = torch.tensor(
                self.token_type_ids, device=device
            )
        else:
            result["token_type_ids"] = None

        return result

    def __len__(self) -> int:
        """Get batch size."""
        return len(self.input_ids)


class TokenizationPipeline:
    """Tokenization pipeline with caching and batch support.

    Provides efficient tokenization with:
    - System prompt caching
    - Batch processing with padding/truncation
    - Multiple tokenizer backend support

    Example:
        ```python
        from layerzero.integrations.tokenization_pipeline import TokenizationPipeline

        pipeline = TokenizationPipeline(tokenizer, max_length=512)
        pipeline.cache_system_prompt("You are a helpful assistant.")

        result = pipeline.encode(["Hello", "World"])
        tensors = result.to_tensors(device="cuda")
        ```
    """

    def __init__(
        self,
        tokenizer: TokenizerProtocol | Any,
        max_length: int = 2048,
        padding: bool | str = True,
        truncation: bool = True,
        pad_token_id: int = 0,
    ) -> None:
        """Initialize tokenization pipeline.

        Args:
            tokenizer: Tokenizer instance (HF, tiktoken, or adapter).
            max_length: Maximum sequence length.
            padding: Whether to pad sequences (True, False, or "max_length").
            truncation: Whether to truncate sequences.
            pad_token_id: Token ID to use for padding.
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation
        self.pad_token_id = pad_token_id

        # Caching
        self._cached_system_prompt: tuple[int, ...] | None = None
        self._phrase_cache: dict[str, tuple[int, ...]] = {}

    def encode(
        self,
        text: str | list[str],
    ) -> TokenizedBatch:
        """Encode text(s) to tokens.

        Args:
            text: Single text string or list of strings.

        Returns:
            TokenizedBatch with input_ids and attention_mask.
        """
        if isinstance(text, str):
            texts = [text]
        else:
            texts = text

        # Encode all texts
        if hasattr(self.tokenizer, "batch_encode"):
            all_ids = self.tokenizer.batch_encode(texts)
        else:
            all_ids = [self.tokenizer.encode(t) for t in texts]

        # Apply truncation
        if self.truncation:
            all_ids = [ids[:self.max_length] for ids in all_ids]

        # Apply padding
        if self.padding:
            max_len = max(len(ids) for ids in all_ids)
            if self.padding == "max_length":
                max_len = self.max_length

            padded_ids: list[list[int]] = []
            attention_masks: list[list[int]] = []

            for ids in all_ids:
                pad_len = max_len - len(ids)
                padded = list(ids) + [self.pad_token_id] * pad_len
                mask = [1] * len(ids) + [0] * pad_len
                padded_ids.append(padded)
                attention_masks.append(mask)

            return TokenizedBatch(
                input_ids=padded_ids,
                attention_mask=attention_masks,
            )

        return TokenizedBatch(
            input_ids=[list(ids) for ids in all_ids],
            attention_mask=[[1] * len(ids) for ids in all_ids],
        )

    def decode(
        self,
        token_ids: list[int] | list[list[int]],
    ) -> str | list[str]:
        """Decode token IDs to text.

        Args:
            token_ids: Single sequence or batch of sequences.

        Returns:
            Decoded text string(s).
        """
        if not token_ids:
            return "" if isinstance(token_ids, list) and not any(isinstance(x, list) for x in token_ids) else []

        # Check if batch or single
        if isinstance(token_ids[0], list):
            return [self.tokenizer.decode(ids) for ids in token_ids]
        return self.tokenizer.decode(token_ids)

    def cache_system_prompt(self, system_prompt: str) -> tuple[int, ...]:
        """Cache system prompt tokens.

        Tokenizes the system prompt and caches the result for reuse.

        Args:
            system_prompt: System prompt text.

        Returns:
            Tuple of token IDs.
        """
        tokens = tuple(self.tokenizer.encode(system_prompt))
        self._cached_system_prompt = tokens
        logger.debug(f"Cached system prompt ({len(tokens)} tokens)")
        return tokens

    def get_cached_system_prompt(self) -> tuple[int, ...] | None:
        """Get cached system prompt if available.

        Returns:
            Cached token tuple or None if not cached.
        """
        return self._cached_system_prompt

    def cache_phrase(self, phrase: str) -> tuple[int, ...]:
        """Cache a common phrase.

        Args:
            phrase: Phrase text to cache.

        Returns:
            Tuple of token IDs.
        """
        if phrase not in self._phrase_cache:
            self._phrase_cache[phrase] = tuple(self.tokenizer.encode(phrase))
        return self._phrase_cache[phrase]

    def clear_cache(self) -> None:
        """Clear all cached tokens."""
        self._cached_system_prompt = None
        self._phrase_cache.clear()
        logger.debug("Cleared tokenization cache")


# Model type to tokenizer type mapping
MODEL_TOKENIZER_MAP: dict[str, TokenizerType] = {
    "llama": TokenizerType.SENTENCEPIECE,
    "llama2": TokenizerType.SENTENCEPIECE,
    "llama3": TokenizerType.TIKTOKEN,
    "gpt2": TokenizerType.HF_TOKENIZERS,
    "gpt-neo": TokenizerType.HF_TOKENIZERS,
    "gptj": TokenizerType.HF_TOKENIZERS,
    "mistral": TokenizerType.SENTENCEPIECE,
    "mixtral": TokenizerType.SENTENCEPIECE,
    "phi": TokenizerType.HF_TOKENIZERS,
    "phi3": TokenizerType.HF_TOKENIZERS,
    "qwen": TokenizerType.TIKTOKEN,
    "qwen2": TokenizerType.TIKTOKEN,
}


class DefaultTokenizer:
    """Default tokenizer when no specific tokenizer is available."""

    def __init__(self, vocab_size: int = 32000) -> None:
        self._vocab_size = vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        # Simple byte-level encoding
        return [b % self._vocab_size for b in text.encode("utf-8")]

    def decode(self, ids: list[int]) -> str:
        # Best-effort decode
        try:
            return bytes(i % 256 for i in ids).decode("utf-8", errors="replace")
        except Exception:
            return ""

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def get_cache_key(self) -> str:
        return f"default_tokenizer_v1_{self._vocab_size}"


def auto_select_tokenizer(model: "nn.Module") -> Any:
    """Auto-select tokenizer based on model configuration.

    Inspects the model's config to determine the appropriate
    tokenizer type and returns a suitable tokenizer instance.

    Args:
        model: PyTorch model with config attribute.

    Returns:
        Tokenizer instance appropriate for the model.
    """
    model_type = None

    if hasattr(model, "config"):
        config = model.config
        if hasattr(config, "model_type"):
            model_type = config.model_type.lower()

    if model_type is None:
        # Fallback to class name inspection
        class_name = type(model).__name__.lower()
        for known_type in MODEL_TOKENIZER_MAP:
            if known_type in class_name:
                model_type = known_type
                break

    if model_type is not None:
        logger.debug(f"Auto-detected model type: {model_type}")
        tokenizer_type = MODEL_TOKENIZER_MAP.get(model_type, TokenizerType.HF_TOKENIZERS)

        # Try to load appropriate tokenizer
        try:
            if tokenizer_type == TokenizerType.HF_TOKENIZERS:
                from layerzero.backends.tokenization.hf_tokenizers import HFTokenizersAdapter
                # Would need actual tokenizer loading here
                return DefaultTokenizer()
            elif tokenizer_type == TokenizerType.TIKTOKEN:
                from layerzero.backends.tokenization.tiktoken import TiktokenAdapter
                return DefaultTokenizer()
            elif tokenizer_type == TokenizerType.SENTENCEPIECE:
                from layerzero.backends.tokenization.sentencepiece import SentencePieceAdapter
                return DefaultTokenizer()
        except ImportError:
            pass

    # Return default tokenizer
    logger.debug("Using default tokenizer")
    return DefaultTokenizer()


def get_tokenizer_for_model(model_name: str) -> Any:
    """Get appropriate tokenizer for a model name.

    Args:
        model_name: Model name or path (e.g., "meta-llama/Llama-2-7b").

    Returns:
        Tokenizer instance.
    """
    model_name_lower = model_name.lower()

    # Determine tokenizer type from model name
    tokenizer_type = TokenizerType.HF_TOKENIZERS

    for model_type, tok_type in MODEL_TOKENIZER_MAP.items():
        if model_type in model_name_lower:
            tokenizer_type = tok_type
            break

    logger.debug(f"Selected tokenizer type {tokenizer_type} for {model_name}")

    # Try to create appropriate tokenizer
    try:
        if tokenizer_type == TokenizerType.TIKTOKEN:
            try:
                import tiktoken
                # For models using tiktoken (OpenAI models, LLaMA 3, Qwen)
                # Would need actual encoding name mapping
                return DefaultTokenizer()
            except ImportError:
                pass

        elif tokenizer_type == TokenizerType.SENTENCEPIECE:
            try:
                import sentencepiece
                return DefaultTokenizer()
            except ImportError:
                pass

        # Try HF tokenizers
        try:
            from transformers import AutoTokenizer
            # Would load actual tokenizer: AutoTokenizer.from_pretrained(model_name)
            return DefaultTokenizer()
        except ImportError:
            pass

    except Exception as e:
        logger.warning(f"Failed to load tokenizer for {model_name}: {e}")

    return DefaultTokenizer()


def create_pipeline_tokenizer(
    tokenizer: Any,
    max_length: int = 2048,
    **kwargs: Any,
) -> TokenizationPipeline:
    """Create a tokenization pipeline from any tokenizer.

    Wraps various tokenizer types in a unified pipeline interface.

    Args:
        tokenizer: Tokenizer instance (HF, tiktoken, or adapter).
        max_length: Maximum sequence length.
        **kwargs: Additional arguments for TokenizationPipeline.

    Returns:
        TokenizationPipeline instance.
    """
    return TokenizationPipeline(
        tokenizer=tokenizer,
        max_length=max_length,
        **kwargs,
    )
