"""
Tokenizer Cache Key Generation

Generates stable cache keys for tokenizer identity based on:
- Vocabulary hash
- Normalizer configuration
- BPE merges (if applicable)
- Added tokens
"""
from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass


def hash_vocab(vocab: dict[str, int]) -> str:
    """Generate hash for vocabulary mapping.

    The hash is order-independent since vocabularies are
    semantically unordered mappings.

    Args:
        vocab: Token to ID mapping.

    Returns:
        SHA256 hash of the vocabulary (first 16 hex chars).
    """
    if not vocab:
        return "empty"

    # Sort by token for deterministic ordering
    sorted_items = sorted(vocab.items(), key=lambda x: x[0])
    vocab_str = json.dumps(sorted_items, ensure_ascii=False, sort_keys=True)
    hash_bytes = hashlib.sha256(vocab_str.encode("utf-8")).digest()
    return hash_bytes[:8].hex()


def hash_merges(
    merges: list[tuple[str, str]] | None,
) -> str | None:
    """Generate hash for BPE merges.

    Order matters for merges since they define priority.

    Args:
        merges: List of BPE merge pairs in priority order.

    Returns:
        SHA256 hash of merges (first 16 hex chars), or None if no merges.
    """
    if merges is None:
        return None

    if not merges:
        return "empty"

    # Merges are ordered, so we preserve the order
    merges_str = json.dumps(merges, ensure_ascii=False)
    hash_bytes = hashlib.sha256(merges_str.encode("utf-8")).digest()
    return hash_bytes[:8].hex()


def hash_added_tokens(
    added_tokens: dict[str, int] | None,
) -> str | None:
    """Generate hash for added tokens.

    Args:
        added_tokens: Added token to ID mapping.

    Returns:
        SHA256 hash of added tokens (first 16 hex chars), or None if no added tokens.
    """
    if added_tokens is None:
        return None

    if not added_tokens:
        return "empty"

    # Sort for deterministic ordering
    sorted_items = sorted(added_tokens.items(), key=lambda x: x[0])
    tokens_str = json.dumps(sorted_items, ensure_ascii=False, sort_keys=True)
    hash_bytes = hashlib.sha256(tokens_str.encode("utf-8")).digest()
    return hash_bytes[:8].hex()


def generate_tokenizer_cache_key(
    vocab_hash: str,
    normalizer_id: str,
    merges_hash: str | None,
    added_tokens_hash: str | None,
) -> str:
    """Generate stable cache key for tokenizer.

    Combines all tokenizer configuration components into a single
    cache key string that uniquely identifies the tokenizer.

    Args:
        vocab_hash: Hash of vocabulary mapping.
        normalizer_id: Identifier for text normalizer (e.g., "nfkc", "lowercase").
        merges_hash: Hash of BPE merges, or None if not applicable.
        added_tokens_hash: Hash of added tokens, or None if not applicable.

    Returns:
        Combined cache key string.

    Example:
        ```python
        key = generate_tokenizer_cache_key(
            vocab_hash="a1b2c3d4",
            normalizer_id="nfkc",
            merges_hash="e5f6g7h8",
            added_tokens_hash=None,
        )
        # Returns: "vocab:a1b2c3d4|norm:nfkc|merges:e5f6g7h8"
        ```
    """
    parts = [
        f"vocab:{vocab_hash}",
        f"norm:{normalizer_id}",
    ]

    if merges_hash is not None:
        parts.append(f"merges:{merges_hash}")

    if added_tokens_hash is not None:
        parts.append(f"added:{added_tokens_hash}")

    return "|".join(parts)


def generate_tiktoken_cache_key(
    encoding_name: str,
    plugin_version: str | None = None,
) -> str:
    """Generate cache key for tiktoken encoding.

    tiktoken encodings are identified by name and optionally
    by plugin version for custom encodings.

    Args:
        encoding_name: Name of the encoding (e.g., "cl100k_base").
        plugin_version: Version of tiktoken extension plugin, if any.

    Returns:
        Cache key string.
    """
    if plugin_version:
        return f"tiktoken:{encoding_name}:v{plugin_version}"
    return f"tiktoken:{encoding_name}"


def generate_sentencepiece_cache_key(
    model_hash: str,
    normalizer_spec: str = "default",
) -> str:
    """Generate cache key for SentencePiece model.

    Args:
        model_hash: SHA256 hash of the .model file.
        normalizer_spec: Normalizer specification string.

    Returns:
        Cache key string.
    """
    return f"sentencepiece:{model_hash}|norm:{normalizer_spec}"


def hash_file(file_path: str) -> str:
    """Generate SHA256 hash of a file.

    Reads file in chunks for memory efficiency with large files.

    Args:
        file_path: Path to the file.

    Returns:
        SHA256 hash (first 16 hex chars).

    Raises:
        FileNotFoundError: If file does not exist.
    """
    hasher = hashlib.sha256()
    chunk_size = 65536  # 64KB chunks

    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            hasher.update(chunk)

    return hasher.digest()[:8].hex()
