"""
LayerZero JIT Warmup Module

Provides JIT compilation warmup protocol to prevent latency spikes
in production from first-time kernel compilation.

Main components:
- WarmupConfig: Configuration for warmup behavior
- ShapeManifest: Track shapes for warmup
- JITWarmupProtocol: Execute warmup protocol
- WarmupReport: Results from warmup execution

Usage:
    from layerzero.warmup import WarmupConfig, JITWarmupProtocol, ShapeManifest

    config = WarmupConfig(timeout_ms=30000.0, blocking=True)
    protocol = JITWarmupProtocol(config)

    manifest = ShapeManifest.from_model_config(model_config)
    report = protocol.warmup(manifest)

    if not report.success_rate == 1.0:
        print(f"Warmup had {report.failed_shapes} failures")
"""
from __future__ import annotations

from layerzero.warmup.config import (
    ShapeWarmupResult,
    WarmupConfig,
    WarmupReport,
)
from layerzero.warmup.protocol import (
    JITWarmupProtocol,
    WarmupStatus,
)
from layerzero.warmup.shape_manifest import (
    DEFAULT_BATCH_BUCKETS,
    DEFAULT_SEQ_BUCKETS,
    ShapeManifest,
    ShapeSignature,
    bucket_batch_size,
    bucket_seq_len,
)

__all__ = [
    # Config
    "WarmupConfig",
    "WarmupReport",
    "ShapeWarmupResult",
    # Protocol
    "JITWarmupProtocol",
    "WarmupStatus",
    # Manifest
    "ShapeManifest",
    "ShapeSignature",
    "bucket_seq_len",
    "bucket_batch_size",
    "DEFAULT_SEQ_BUCKETS",
    "DEFAULT_BATCH_BUCKETS",
]
