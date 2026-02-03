"""
Shape manifest for JIT warmup.

This module provides:
- ShapeSignature: Describes a kernel shape for warmup
- ShapeManifest: Collection of shapes to warmup
- Bucketing utilities for sequence length and batch size
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch

# Default buckets for common sequence lengths (powers of 2)
DEFAULT_SEQ_BUCKETS: tuple[int, ...] = (128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)

# Default buckets for batch sizes
DEFAULT_BATCH_BUCKETS: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64)

# Map torch dtype to string for serialization
DTYPE_TO_STR: dict[torch.dtype, str] = {
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.float32: "float32",
    torch.float64: "float64",
    torch.int8: "int8",
    torch.int16: "int16",
    torch.int32: "int32",
    torch.int64: "int64",
}

STR_TO_DTYPE: dict[str, torch.dtype] = {v: k for k, v in DTYPE_TO_STR.items()}


def bucket_seq_len(seq_len: int) -> int:
    """Bucket sequence length to nearest power of 2.

    Args:
        seq_len: Actual sequence length.

    Returns:
        Bucketed sequence length (power of 2 >= seq_len).

    Example:
        >>> bucket_seq_len(500)
        512
        >>> bucket_seq_len(2000)
        2048
    """
    if seq_len <= 0:
        return 1
    # Round up to nearest power of 2
    return 1 << (seq_len - 1).bit_length()


def bucket_batch_size(batch_size: int) -> int:
    """Bucket batch size to nearest power of 2.

    Args:
        batch_size: Actual batch size.

    Returns:
        Bucketed batch size (power of 2 >= batch_size).

    Example:
        >>> bucket_batch_size(3)
        4
        >>> bucket_batch_size(10)
        16
    """
    if batch_size <= 0:
        return 1
    if batch_size == 1:
        return 1
    # Round up to nearest power of 2
    return 1 << (batch_size - 1).bit_length()


@dataclass(frozen=True)
class ShapeSignature:
    """Describes a kernel shape for warmup.

    This identifies a unique combination of parameters that requires
    JIT compilation. Two identical signatures will produce the same
    compiled kernel.

    Attributes:
        operation: Operation name (e.g., "attention.causal", "norm.rms").
        dtype: Data type for computation.
        batch_size_bucket: Bucketed batch size.
        seq_len_bucket: Bucketed sequence length.
        head_dim: Head dimension for attention.
        num_heads: Number of attention heads.
        num_kv_heads: Number of KV heads for GQA (optional).
        layout: Memory layout ("BSHD" or "BHSD").

    Example:
        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
            num_kv_heads=8,
        )
    """

    operation: str
    dtype: torch.dtype
    batch_size_bucket: int
    seq_len_bucket: int
    head_dim: int
    num_heads: int
    num_kv_heads: int | None = None
    layout: str = "BSHD"

    def to_key(self) -> str:
        """Generate unique cache key for this shape.

        Returns:
            String key uniquely identifying this shape.
        """
        parts = [
            self.operation,
            DTYPE_TO_STR.get(self.dtype, str(self.dtype)),
            str(self.batch_size_bucket),
            str(self.seq_len_bucket),
            str(self.head_dim),
            str(self.num_heads),
            str(self.num_kv_heads) if self.num_kv_heads else "none",
            self.layout,
        ]
        return ":".join(parts)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization.

        Returns:
            Dictionary representation.
        """
        return {
            "operation": self.operation,
            "dtype": DTYPE_TO_STR.get(self.dtype, str(self.dtype)),
            "batch_size_bucket": self.batch_size_bucket,
            "seq_len_bucket": self.seq_len_bucket,
            "head_dim": self.head_dim,
            "num_heads": self.num_heads,
            "num_kv_heads": self.num_kv_heads,
            "layout": self.layout,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ShapeSignature":
        """Create from dictionary.

        Args:
            data: Dictionary with shape fields.

        Returns:
            ShapeSignature instance.
        """
        dtype_str = data["dtype"]
        dtype = STR_TO_DTYPE.get(dtype_str)
        if dtype is None:
            # Try parsing torch.dtype string
            dtype = getattr(torch, dtype_str.replace("torch.", ""), torch.float16)

        return cls(
            operation=data["operation"],
            dtype=dtype,
            batch_size_bucket=data["batch_size_bucket"],
            seq_len_bucket=data["seq_len_bucket"],
            head_dim=data["head_dim"],
            num_heads=data["num_heads"],
            num_kv_heads=data.get("num_kv_heads"),
            layout=data.get("layout", "BSHD"),
        )


@dataclass
class ShapeEntry:
    """Internal entry in ShapeManifest with metadata."""

    signature: ShapeSignature
    critical: bool = False
    frequency: float = 1.0


@dataclass
class ShapeManifest:
    """Collection of shapes to warmup.

    Tracks shapes that require JIT compilation warmup, with support for
    marking critical shapes and persistence to disk.

    Attributes:
        model_config_hash: Hash of model config for version tracking.

    Example:
        manifest = ShapeManifest()
        manifest.add_shape(sig, critical=True)
        manifest.save(Path("manifest.json"))

        loaded = ShapeManifest.load(Path("manifest.json"))
    """

    model_config_hash: str | None = None
    _shapes: dict[str, ShapeEntry] = field(default_factory=dict)

    def __len__(self) -> int:
        """Get number of shapes in manifest."""
        return len(self._shapes)

    def __iter__(self) -> Iterator[ShapeSignature]:
        """Iterate over shape signatures."""
        return iter(entry.signature for entry in self._shapes.values())

    def __contains__(self, sig: ShapeSignature) -> bool:
        """Check if shape is in manifest."""
        return sig.to_key() in self._shapes

    def add_shape(
        self,
        sig: ShapeSignature,
        critical: bool = False,
        frequency: float = 1.0,
    ) -> None:
        """Add shape to manifest.

        If shape already exists, updates critical/frequency if new values
        are higher priority.

        Args:
            sig: Shape signature to add.
            critical: Whether this is a critical shape.
            frequency: Expected frequency (for prioritization).
        """
        key = sig.to_key()
        if key in self._shapes:
            # Update if new entry is higher priority
            existing = self._shapes[key]
            if critical and not existing.critical:
                self._shapes[key] = ShapeEntry(sig, critical=True, frequency=frequency)
            elif frequency > existing.frequency:
                self._shapes[key] = ShapeEntry(
                    sig,
                    critical=existing.critical or critical,
                    frequency=frequency,
                )
        else:
            self._shapes[key] = ShapeEntry(sig, critical=critical, frequency=frequency)

    def get_shapes(self, critical_only: bool = False) -> list[ShapeSignature]:
        """Get shapes from manifest.

        Args:
            critical_only: If True, only return critical shapes.

        Returns:
            List of shape signatures.
        """
        if critical_only:
            return [
                entry.signature
                for entry in self._shapes.values()
                if entry.critical
            ]
        return [entry.signature for entry in self._shapes.values()]

    def get_ordered_shapes(self, critical_first: bool = True) -> list[ShapeSignature]:
        """Get shapes ordered by priority.

        Args:
            critical_first: If True, critical shapes come first.

        Returns:
            Ordered list of shape signatures.
        """
        entries = list(self._shapes.values())

        if critical_first:
            # Sort by: critical (desc), frequency (desc)
            entries.sort(key=lambda e: (e.critical, e.frequency), reverse=True)
        else:
            # Sort by frequency only
            entries.sort(key=lambda e: e.frequency, reverse=True)

        return [entry.signature for entry in entries]

    def save(self, path: Path) -> None:
        """Save manifest to JSON file.

        Args:
            path: Path to save manifest.
        """
        data = {
            "version": "1.0",
            "model_config_hash": self.model_config_hash,
            "shapes": [
                {
                    "signature": entry.signature.to_dict(),
                    "critical": entry.critical,
                    "frequency": entry.frequency,
                }
                for entry in self._shapes.values()
            ],
        }

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "ShapeManifest":
        """Load manifest from JSON file.

        Args:
            path: Path to load manifest from.

        Returns:
            Loaded ShapeManifest.

        Raises:
            FileNotFoundError: If file doesn't exist.
        """
        with open(path) as f:
            data = json.load(f)

        manifest = cls(model_config_hash=data.get("model_config_hash"))

        for shape_data in data.get("shapes", []):
            sig = ShapeSignature.from_dict(shape_data["signature"])
            manifest.add_shape(
                sig,
                critical=shape_data.get("critical", False),
                frequency=shape_data.get("frequency", 1.0),
            )

        return manifest

    @classmethod
    def from_model_config(
        cls,
        config: dict[str, Any],
        batch_buckets: list[int] | None = None,
        seq_buckets: list[int] | None = None,
        dtypes: list[torch.dtype] | None = None,
        operations: list[str] | None = None,
    ) -> "ShapeManifest":
        """Create manifest from model configuration.

        Generates shapes for common combinations of batch size and
        sequence length based on model config.

        Args:
            config: Model configuration dictionary.
            batch_buckets: Batch size buckets to warmup.
            seq_buckets: Sequence length buckets to warmup.
            dtypes: Data types to warmup.
            operations: Operations to warmup.

        Returns:
            ShapeManifest with generated shapes.
        """
        # Compute config hash for versioning
        config_str = json.dumps(config, sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:16]

        manifest = cls(model_config_hash=config_hash)

        # Extract model parameters
        num_heads = config.get("num_attention_heads", 32)
        num_kv_heads = config.get("num_key_value_heads", num_heads)
        head_dim = config.get("head_dim")
        if head_dim is None:
            hidden_size = config.get("hidden_size", 4096)
            head_dim = hidden_size // num_heads

        max_seq = config.get("max_position_embeddings", 8192)

        # Use defaults if not specified
        batch_buckets = batch_buckets or list(DEFAULT_BATCH_BUCKETS)
        seq_buckets = seq_buckets or [
            s for s in DEFAULT_SEQ_BUCKETS if s <= max_seq
        ]
        dtypes = dtypes or [torch.float16, torch.bfloat16]
        operations = operations or ["attention.causal"]

        # Generate shapes
        for op in operations:
            for dtype in dtypes:
                for batch in batch_buckets:
                    for seq in seq_buckets:
                        # Mark small shapes as critical (fast to compile, commonly used)
                        critical = batch <= 4 and seq <= 2048

                        sig = ShapeSignature(
                            operation=op,
                            dtype=dtype,
                            batch_size_bucket=batch,
                            seq_len_bucket=seq,
                            head_dim=head_dim,
                            num_heads=num_heads,
                            num_kv_heads=num_kv_heads if num_kv_heads != num_heads else None,
                        )

                        # Frequency decreases with larger shapes
                        frequency = 1.0 / (math.log2(batch + 1) + math.log2(seq + 1))
                        manifest.add_shape(sig, critical=critical, frequency=frequency)

        return manifest
