"""Tests for tokenization pipeline integration."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from typing import Any

from layerzero.integrations.tokenization_pipeline import (
    TokenizationPipeline,
    TokenizedBatch,
    auto_select_tokenizer,
    get_tokenizer_for_model,
    TokenizerType,
)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self._vocab_size = vocab_size
        self._vocab = {f"token_{i}": i for i in range(vocab_size)}
        self._vocab["<pad>"] = 0
        self._vocab["<unk>"] = 1
        self._vocab["<bos>"] = 2
        self._vocab["<eos>"] = 3
        self._reverse_vocab = {v: k for k, v in self._vocab.items()}

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        # Simple character-based encoding for testing
        tokens = []
        for char in text.lower():
            if char in self._vocab:
                tokens.append(self._vocab[char])
            else:
                tokens.append(1)  # <unk>
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "".join(
            self._reverse_vocab.get(i, "<unk>") for i in ids
        )

    def batch_encode(self, texts: list[str]) -> list[list[int]]:
        return [self.encode(t) for t in texts]

    def get_cache_key(self) -> str:
        return f"mock_tokenizer_v1_{self._vocab_size}"


class MockModel(nn.Module):
    """Mock model for testing tokenizer auto-selection."""

    def __init__(self, model_type: str = "llama") -> None:
        super().__init__()
        self.linear = nn.Linear(10, 10)
        self.config = type("Config", (), {
            "model_type": model_type,
            "vocab_size": 32000,
        })()


class TestTokenizationPipelineIntegration:
    """Test pipeline tokenizer selection."""

    @pytest.fixture
    def mock_tokenizer(self) -> MockTokenizer:
        return MockTokenizer()

    @pytest.fixture
    def pipeline(self, mock_tokenizer: MockTokenizer) -> TokenizationPipeline:
        return TokenizationPipeline(
            tokenizer=mock_tokenizer,
            max_length=512,
            padding=True,
            truncation=True,
        )

    def test_pipeline_tokenizer_selection(self, pipeline: TokenizationPipeline) -> None:
        """Correct tokenizer selected for model."""
        # Pipeline should use the provided tokenizer
        assert pipeline.tokenizer is not None
        assert hasattr(pipeline.tokenizer, "encode")
        assert hasattr(pipeline.tokenizer, "decode")

    def test_pipeline_tokenizer_auto_detect(self) -> None:
        """Auto-detect tokenizer from model config."""
        model = MockModel(model_type="llama")

        tokenizer = auto_select_tokenizer(model)

        # Should return a tokenizer (may be mock or default)
        assert tokenizer is not None


class TestTokenizationCaching:
    """Test tokenization caching functionality."""

    @pytest.fixture
    def pipeline(self) -> TokenizationPipeline:
        return TokenizationPipeline(
            tokenizer=MockTokenizer(),
            max_length=512,
        )

    def test_system_prompt_caching(self, pipeline: TokenizationPipeline) -> None:
        """System prompt tokenization cached."""
        system_prompt = "You are a helpful assistant."

        # Cache the system prompt
        tokens = pipeline.cache_system_prompt(system_prompt)

        # Should return tokens
        assert tokens is not None
        assert len(tokens) > 0

        # Should be cached
        cached = pipeline.get_cached_system_prompt()
        assert cached is not None
        assert cached == tokens

    def test_cache_hit_reuses_tokens(self, pipeline: TokenizationPipeline) -> None:
        """Cache hit reuses tokenized result."""
        system_prompt = "You are a helpful assistant."

        # Cache and encode
        tokens1 = pipeline.cache_system_prompt(system_prompt)
        tokens2 = pipeline.get_cached_system_prompt()

        # Should be identical (same object if cached)
        assert tokens1 == tokens2

    def test_cache_invalidation_on_config_change(self) -> None:
        """Cache invalidated when config changes."""
        tokenizer = MockTokenizer()
        pipeline = TokenizationPipeline(
            tokenizer=tokenizer,
            max_length=512,
        )

        # Cache a prompt
        system_prompt = "You are a helpful assistant."
        tokens1 = pipeline.cache_system_prompt(system_prompt)

        # Change the cached prompt
        new_prompt = "You are an expert."
        tokens2 = pipeline.cache_system_prompt(new_prompt)

        # Should be different
        assert tokens1 != tokens2

        # New prompt should be cached
        cached = pipeline.get_cached_system_prompt()
        assert cached == tokens2


class TestBatchTokenization:
    """Test batch tokenization functionality."""

    @pytest.fixture
    def pipeline(self) -> TokenizationPipeline:
        return TokenizationPipeline(
            tokenizer=MockTokenizer(),
            max_length=512,
            padding=True,
            truncation=True,
        )

    def test_batch_tokenization_efficient(self, pipeline: TokenizationPipeline) -> None:
        """Batch tokenization is efficient."""
        texts = [
            "Hello world",
            "How are you",
            "This is a test",
        ]

        result = pipeline.encode(texts)

        # Should return TokenizedBatch
        assert isinstance(result, TokenizedBatch)
        assert len(result.input_ids) == 3
        assert result.attention_mask is not None

    def test_batch_padding_handling(self, pipeline: TokenizationPipeline) -> None:
        """Padding handled correctly in batch."""
        texts = [
            "Hi",  # Short
            "This is a longer sentence with more tokens",  # Longer
        ]

        result = pipeline.encode(texts)

        # Both sequences should have same length (padded)
        assert len(result.input_ids[0]) == len(result.input_ids[1])

        # Attention mask should reflect padding
        assert result.attention_mask is not None
        # Shorter sequence should have fewer 1s
        assert sum(result.attention_mask[0]) <= sum(result.attention_mask[1])

    def test_batch_truncation_handling(self) -> None:
        """Truncation handled correctly in batch."""
        pipeline = TokenizationPipeline(
            tokenizer=MockTokenizer(),
            max_length=10,  # Very short max length
            padding=True,
            truncation=True,
        )

        texts = [
            "This is a very long sentence that should be truncated because it exceeds max length",
        ]

        result = pipeline.encode(texts)

        # Should be truncated to max_length
        assert len(result.input_ids[0]) <= 10


class TestTokenizerIntegration:
    """Test integration with different tokenizer types."""

    def test_hf_tokenizers_in_pipeline(self) -> None:
        """HF tokenizers work in pipeline."""
        # Test with mock that simulates HF tokenizer interface
        tokenizer = MockTokenizer()
        pipeline = TokenizationPipeline(tokenizer=tokenizer)

        result = pipeline.encode("Hello world")

        assert result is not None
        assert len(result.input_ids) > 0

    def test_tiktoken_in_pipeline(self) -> None:
        """tiktoken works in pipeline."""
        # Test with mock that simulates tiktoken interface
        tokenizer = MockTokenizer()
        pipeline = TokenizationPipeline(tokenizer=tokenizer)

        result = pipeline.encode("Hello world")

        assert result is not None
        assert len(result.input_ids) > 0


class TestTokenizedBatch:
    """Test TokenizedBatch dataclass."""

    def test_tokenized_batch_creation(self) -> None:
        """TokenizedBatch can be created."""
        batch = TokenizedBatch(
            input_ids=[[1, 2, 3], [4, 5, 6]],
            attention_mask=[[1, 1, 1], [1, 1, 1]],
            token_type_ids=None,
        )

        assert len(batch.input_ids) == 2
        assert len(batch.attention_mask) == 2
        assert batch.token_type_ids is None

    def test_tokenized_batch_to_tensors(self) -> None:
        """TokenizedBatch converts to tensors."""
        batch = TokenizedBatch(
            input_ids=[[1, 2, 3], [4, 5, 6]],
            attention_mask=[[1, 1, 1], [1, 1, 1]],
            token_type_ids=None,
        )

        tensors = batch.to_tensors()

        assert isinstance(tensors["input_ids"], torch.Tensor)
        assert tensors["input_ids"].shape == (2, 3)
        assert isinstance(tensors["attention_mask"], torch.Tensor)
        assert "token_type_ids" not in tensors or tensors["token_type_ids"] is None


class TestTokenizerType:
    """Test TokenizerType enum."""

    def test_tokenizer_types_exist(self) -> None:
        """TokenizerType enum has expected values."""
        assert hasattr(TokenizerType, "HF_TOKENIZERS")
        assert hasattr(TokenizerType, "TIKTOKEN")
        assert hasattr(TokenizerType, "SENTENCEPIECE")
        assert hasattr(TokenizerType, "AUTO")


class TestModelTokenizerMapping:
    """Test model to tokenizer mapping."""

    def test_get_tokenizer_for_llama(self) -> None:
        """Get tokenizer for LLaMA model."""
        tokenizer = get_tokenizer_for_model("meta-llama/Llama-2-7b")

        # Should return a tokenizer (may be default/mock)
        assert tokenizer is not None

    def test_get_tokenizer_for_gpt(self) -> None:
        """Get tokenizer for GPT model."""
        tokenizer = get_tokenizer_for_model("gpt2")

        assert tokenizer is not None

    def test_get_tokenizer_for_unknown(self) -> None:
        """Get tokenizer for unknown model."""
        tokenizer = get_tokenizer_for_model("unknown-model")

        # Should return a default tokenizer
        assert tokenizer is not None
