"""Tests for ShapeManifest class."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch


class TestShapeSignature:
    """Tests for ShapeSignature dataclass."""

    def test_signature_creation(self) -> None:
        """Test creating shape signature."""
        from layerzero.warmup.shape_manifest import ShapeSignature

        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
            num_kv_heads=8,
        )

        assert sig.operation == "attention.causal"
        assert sig.dtype == torch.float16
        assert sig.head_dim == 128

    def test_signature_immutable(self) -> None:
        """Test signature is frozen (immutable)."""
        from layerzero.warmup.shape_manifest import ShapeSignature

        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )

        with pytest.raises((AttributeError, TypeError)):
            sig.operation = "attention.paged"  # type: ignore

    def test_signature_hashable(self) -> None:
        """Test signature is hashable."""
        from layerzero.warmup.shape_manifest import ShapeSignature

        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )

        # Should be usable in sets and as dict keys
        sig_set = {sig}
        assert sig in sig_set

    def test_signature_to_key(self) -> None:
        """Test signature generates consistent key."""
        from layerzero.warmup.shape_manifest import ShapeSignature

        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
            num_kv_heads=8,
        )

        key = sig.to_key()
        assert "attention.causal" in key
        assert "1024" in key
        assert "128" in key

    def test_signature_equality(self) -> None:
        """Test signature equality comparison."""
        from layerzero.warmup.shape_manifest import ShapeSignature

        sig1 = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )
        sig2 = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )
        sig3 = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=2048,  # Different
            head_dim=128,
            num_heads=32,
        )

        assert sig1 == sig2
        assert sig1 != sig3

    def test_signature_serialization(self) -> None:
        """Test signature can be serialized to JSON."""
        from layerzero.warmup.shape_manifest import ShapeSignature

        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
            num_kv_heads=8,
        )

        data = sig.to_dict()
        assert isinstance(data, dict)
        assert data["operation"] == "attention.causal"

        # Roundtrip
        sig2 = ShapeSignature.from_dict(data)
        assert sig == sig2


class TestShapeManifest:
    """Tests for ShapeManifest class."""

    def test_manifest_creation(self) -> None:
        """Test creating empty manifest."""
        from layerzero.warmup.shape_manifest import ShapeManifest

        manifest = ShapeManifest()
        assert len(manifest) == 0

    def test_manifest_add_shape(self) -> None:
        """Test adding shapes to manifest."""
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        manifest = ShapeManifest()
        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )

        manifest.add_shape(sig)
        assert len(manifest) == 1

    def test_manifest_add_critical_shape(self) -> None:
        """Test adding critical shapes."""
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        manifest = ShapeManifest()

        sig1 = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )
        sig2 = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=8,
            seq_len_bucket=2048,
            head_dim=128,
            num_heads=32,
        )

        manifest.add_shape(sig1, critical=True)
        manifest.add_shape(sig2, critical=False)

        critical = manifest.get_shapes(critical_only=True)
        all_shapes = manifest.get_shapes(critical_only=False)

        assert len(critical) == 1
        assert len(all_shapes) == 2

    def test_manifest_deduplicate(self) -> None:
        """Test manifest deduplicates shapes."""
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        manifest = ShapeManifest()
        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )

        manifest.add_shape(sig)
        manifest.add_shape(sig)  # Duplicate

        assert len(manifest) == 1

    def test_manifest_save_load(self, temp_cache_dir: Path) -> None:
        """Test manifest save and load."""
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        manifest = ShapeManifest()
        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )
        manifest.add_shape(sig, critical=True)

        path = temp_cache_dir / "manifest.json"
        manifest.save(path)

        assert path.exists()

        loaded = ShapeManifest.load(path)
        assert len(loaded) == 1
        assert loaded.get_shapes(critical_only=True)[0] == sig

    def test_manifest_from_model_config(self, sample_model_config: dict) -> None:
        """Test creating manifest from model config."""
        from layerzero.warmup.shape_manifest import ShapeManifest

        manifest = ShapeManifest.from_model_config(sample_model_config)

        # Should generate shapes for common batch sizes and seq lengths
        shapes = manifest.get_shapes()
        assert len(shapes) > 0

        # Check shapes have correct head_dim from config
        for shape in shapes:
            assert shape.head_dim == sample_model_config["head_dim"]
            assert shape.num_heads == sample_model_config["num_attention_heads"]

    def test_manifest_from_model_config_buckets(self, sample_model_config: dict) -> None:
        """Test manifest generates bucketed shapes."""
        from layerzero.warmup.shape_manifest import ShapeManifest

        manifest = ShapeManifest.from_model_config(
            sample_model_config,
            batch_buckets=[1, 4, 8],
            seq_buckets=[512, 1024, 2048, 4096],
        )

        shapes = manifest.get_shapes()
        seq_lens = {s.seq_len_bucket for s in shapes}

        assert 512 in seq_lens or 1024 in seq_lens

    def test_manifest_version_hash(self) -> None:
        """Test manifest version tied to config hash."""
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        manifest = ShapeManifest(model_config_hash="abc123")
        assert manifest.model_config_hash == "abc123"

    def test_manifest_iteration(self) -> None:
        """Test iterating over manifest."""
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        manifest = ShapeManifest()
        sig1 = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )
        sig2 = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=8,
            seq_len_bucket=2048,
            head_dim=128,
            num_heads=32,
        )

        manifest.add_shape(sig1)
        manifest.add_shape(sig2)

        shapes = list(manifest)
        assert len(shapes) == 2


class TestBucketing:
    """Tests for shape bucketing utilities."""

    def test_bucket_seq_len(self) -> None:
        """Test sequence length bucketing."""
        from layerzero.warmup.shape_manifest import bucket_seq_len

        # Powers of 2 bucketing
        assert bucket_seq_len(100) == 128
        assert bucket_seq_len(500) == 512
        assert bucket_seq_len(1000) == 1024
        assert bucket_seq_len(2000) == 2048
        assert bucket_seq_len(3000) == 4096

    def test_bucket_batch_size(self) -> None:
        """Test batch size bucketing."""
        from layerzero.warmup.shape_manifest import bucket_batch_size

        assert bucket_batch_size(1) == 1
        assert bucket_batch_size(3) == 4
        assert bucket_batch_size(5) == 8
        assert bucket_batch_size(10) == 16

    def test_default_seq_buckets(self) -> None:
        """Test default sequence length buckets."""
        from layerzero.warmup.shape_manifest import DEFAULT_SEQ_BUCKETS

        assert 512 in DEFAULT_SEQ_BUCKETS
        assert 1024 in DEFAULT_SEQ_BUCKETS
        assert 2048 in DEFAULT_SEQ_BUCKETS

    def test_default_batch_buckets(self) -> None:
        """Test default batch size buckets."""
        from layerzero.warmup.shape_manifest import DEFAULT_BATCH_BUCKETS

        assert 1 in DEFAULT_BATCH_BUCKETS
        assert 8 in DEFAULT_BATCH_BUCKETS
