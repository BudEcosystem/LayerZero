"""
LayerZero Tokenization Backend Adapters

Unified tokenization API across multiple tokenizer libraries:
- HuggingFace Tokenizers (Rust-based, fast)
- tiktoken (OpenAI's tokenizer)
- SentencePiece (legacy models)
"""
from layerzero.backends.tokenization.base import BaseTokenizerAdapter
from layerzero.backends.tokenization.hf_tokenizers import (
    HFTokenizerAdapter,
    is_hf_tokenizers_available,
)
from layerzero.backends.tokenization.tiktoken import (
    TiktokenAdapter,
    is_tiktoken_available,
)
from layerzero.backends.tokenization.sentencepiece import (
    SentencePieceAdapter,
    is_sentencepiece_available,
)
from layerzero.backends.tokenization.cache_key import (
    generate_tokenizer_cache_key,
    hash_vocab,
    hash_merges,
    hash_added_tokens,
)

__all__ = [
    "BaseTokenizerAdapter",
    "HFTokenizerAdapter",
    "is_hf_tokenizers_available",
    "TiktokenAdapter",
    "is_tiktoken_available",
    "SentencePieceAdapter",
    "is_sentencepiece_available",
    "generate_tokenizer_cache_key",
    "hash_vocab",
    "hash_merges",
    "hash_added_tokens",
]
