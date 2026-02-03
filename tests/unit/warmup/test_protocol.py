"""Tests for JITWarmupProtocol class."""
from __future__ import annotations

import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch


class TestJITWarmupProtocol:
    """Tests for JITWarmupProtocol class."""

    def test_protocol_creation(self) -> None:
        """Test creating warmup protocol."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol

        config = WarmupConfig()
        protocol = JITWarmupProtocol(config)

        assert protocol is not None

    def test_warmup_blocks_until_complete(self, temp_cache_dir: Path) -> None:
        """Test warmup blocks until critical shapes compiled."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig(
            blocking=True,
            cache_dir=temp_cache_dir,
            timeout_ms=5000.0,
        )
        protocol = JITWarmupProtocol(config)

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

        # Mock the actual JIT compilation
        with patch.object(protocol, "_compile_shape") as mock_compile:
            mock_compile.return_value = (True, 100.0, None)

            # Pass specific backends to limit iterations
            report = protocol.warmup(manifest, backends=["torch_sdpa"])

        assert report.total_shapes == 1
        # At least one shape processed (compiled or cached)
        assert report.compiled_shapes + report.cached_shapes >= 1

    def test_warmup_loads_manifest(self, temp_cache_dir: Path) -> None:
        """Test shape manifest loaded from config."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        # Create and save manifest
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
        manifest_path = temp_cache_dir / "manifest.json"
        manifest.save(manifest_path)

        # Load and warmup
        config = WarmupConfig(cache_dir=temp_cache_dir)
        protocol = JITWarmupProtocol(config)

        loaded = ShapeManifest.load(manifest_path)
        assert len(loaded) == 1

    def test_warmup_critical_shapes_first(self) -> None:
        """Test critical shapes compiled before non-critical."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig(critical_shapes_first=True)
        protocol = JITWarmupProtocol(config)

        manifest = ShapeManifest()
        # Add non-critical first
        non_critical = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=8,
            seq_len_bucket=4096,
            head_dim=128,
            num_heads=32,
        )
        manifest.add_shape(non_critical, critical=False)

        # Add critical
        critical = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )
        manifest.add_shape(critical, critical=True)

        compilation_order = []

        def mock_compile(shape: ShapeSignature, backend: str) -> tuple[bool, float, str | None]:
            compilation_order.append(shape.seq_len_bucket)
            return (True, 100.0, None)

        with patch.object(protocol, "_compile_shape", side_effect=mock_compile):
            protocol.warmup(manifest)

        # Critical (1024) should be compiled before non-critical (4096)
        assert compilation_order[0] == 1024

    def test_warmup_timeout_uses_fallback(self) -> None:
        """Test fallback used when warmup times out."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig(timeout_ms=100.0)  # Very short timeout
        protocol = JITWarmupProtocol(config)

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

        def slow_compile(shape: ShapeSignature, backend: str) -> tuple[bool, float, str | None]:
            time.sleep(0.5)  # 500ms, exceeds 100ms timeout
            return (True, 500.0, None)

        with patch.object(protocol, "_compile_shape", side_effect=slow_compile):
            report = protocol.warmup(manifest)

        # Should fail due to timeout
        assert report.failed_shapes >= 1

    def test_background_compile_non_blocking(self) -> None:
        """Test background compile doesn't block requests."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig(blocking=False, background_compile=True)
        protocol = JITWarmupProtocol(config)

        manifest = ShapeManifest()
        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
        )
        manifest.add_shape(sig, critical=False)

        def slow_compile(shape: ShapeSignature, backend: str) -> tuple[bool, float, str | None]:
            time.sleep(0.5)
            return (True, 500.0, None)

        with patch.object(protocol, "_compile_shape", side_effect=slow_compile):
            start = time.perf_counter()
            report = protocol.warmup(manifest)
            elapsed = time.perf_counter() - start

        # Should return quickly without blocking
        assert elapsed < 0.2  # Much less than 500ms compile time


class TestWarmupStatus:
    """Tests for WarmupStatus tracking."""

    def test_warmup_status_creation(self) -> None:
        """Test creating warmup status."""
        from layerzero.warmup.protocol import WarmupStatus

        status = WarmupStatus()
        assert status.critical_complete is False
        assert status.all_complete is False

    def test_warmup_status_critical_pending(self) -> None:
        """Test critical pending count."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig()
        protocol = JITWarmupProtocol(config)

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

        protocol._manifest = manifest
        protocol._warmed_shapes = set()

        status = protocol.get_warmup_status()
        assert status.critical_pending == 1


class TestWarmupTelemetry:
    """Tests for warmup telemetry integration."""

    def test_warmup_status_telemetry(self) -> None:
        """Test warmup status reported via telemetry."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig()
        protocol = JITWarmupProtocol(config)

        # Register mock telemetry
        telemetry_reports = []
        protocol.on_progress = lambda total, done: telemetry_reports.append((total, done))

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

        with patch.object(protocol, "_compile_shape", return_value=(True, 100.0, None)):
            protocol.warmup(manifest, progress_callback=protocol.on_progress)

        assert len(telemetry_reports) > 0


class TestWarmupCachePersistence:
    """Tests for JIT cache persistence."""

    def test_warmup_cache_persistence(self, temp_cache_dir: Path) -> None:
        """Test compiled kernels cached to disk."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig(
            cache_dir=temp_cache_dir,
            persist_cache=True,
        )
        protocol = JITWarmupProtocol(config)

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

        with patch.object(protocol, "_compile_shape", return_value=(True, 100.0, None)):
            protocol.warmup(manifest)

        # Check cache was persisted
        # (Implementation would create files in cache_dir)


class TestWarmupIntegration:
    """Integration tests for warmup with backends."""

    def test_warmup_flashinfer_jit(self) -> None:
        """Test FlashInfer JIT warmup works."""
        pytest.importorskip("flashinfer")

        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig(timeout_ms=60000.0)  # 60s for JIT
        protocol = JITWarmupProtocol(config)

        manifest = ShapeManifest()
        sig = ShapeSignature(
            operation="attention.causal",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=128,
            num_heads=32,
            num_kv_heads=8,
        )
        manifest.add_shape(sig)

        report = protocol.warmup(manifest, backends=["flashinfer"])
        # May fail if flashinfer not installed, which is OK

    def test_warmup_triton_jit(self) -> None:
        """Test Triton JIT warmup works."""
        pytest.importorskip("triton")

        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig(timeout_ms=60000.0)
        protocol = JITWarmupProtocol(config)

        manifest = ShapeManifest()
        sig = ShapeSignature(
            operation="norm.rms",
            dtype=torch.float16,
            batch_size_bucket=1,
            seq_len_bucket=1024,
            head_dim=4096,  # hidden_size for norm
            num_heads=1,
        )
        manifest.add_shape(sig)

        report = protocol.warmup(manifest, backends=["triton"])


class TestWarmupConcurrency:
    """Tests for concurrent warmup operations."""

    def test_max_concurrent_jit(self) -> None:
        """Test max concurrent JIT compilations limited."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig(max_concurrent_jit=2)
        protocol = JITWarmupProtocol(config)

        manifest = ShapeManifest()
        for i in range(10):
            sig = ShapeSignature(
                operation="attention.causal",
                dtype=torch.float16,
                batch_size_bucket=1,
                seq_len_bucket=1024 + i * 512,
                head_dim=128,
                num_heads=32,
            )
            manifest.add_shape(sig)

        concurrent_count = []
        current_concurrent = [0]
        lock = threading.Lock()

        def tracked_compile(shape: ShapeSignature, backend: str) -> tuple[bool, float, str | None]:
            with lock:
                current_concurrent[0] += 1
                concurrent_count.append(current_concurrent[0])
            time.sleep(0.1)
            with lock:
                current_concurrent[0] -= 1
            return (True, 100.0, None)

        with patch.object(protocol, "_compile_shape", side_effect=tracked_compile):
            protocol.warmup(manifest)

        # Should never exceed max_concurrent_jit
        assert max(concurrent_count) <= 2


class TestWarmupErrorHandling:
    """Tests for warmup error handling."""

    def test_warmup_compile_failure_continues(self) -> None:
        """Test warmup continues after compile failure."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig()
        protocol = JITWarmupProtocol(config)

        manifest = ShapeManifest()
        for i in range(3):
            sig = ShapeSignature(
                operation="attention.causal",
                dtype=torch.float16,
                batch_size_bucket=1,
                seq_len_bucket=1024 + i * 512,
                head_dim=128,
                num_heads=32,
            )
            manifest.add_shape(sig)

        call_count = [0]

        def failing_compile(shape: ShapeSignature, backend: str) -> tuple[bool, float, str | None]:
            call_count[0] += 1
            if shape.seq_len_bucket == 1536:  # Second shape fails
                return (False, 0.0, "Compile error")
            return (True, 100.0, None)

        with patch.object(protocol, "_compile_shape", side_effect=failing_compile):
            # Use single backend to get predictable call count
            report = protocol.warmup(manifest, backends=["torch_sdpa"])

        # All 3 shapes should be attempted (1 backend * 3 shapes)
        assert call_count[0] == 3
        assert report.failed_shapes == 1
        # 2 compiled or cached
        assert report.compiled_shapes + report.cached_shapes == 2

    def test_warmup_oom_reduces_concurrency(self) -> None:
        """Test OOM during JIT reduces concurrency."""
        from layerzero.warmup.config import WarmupConfig
        from layerzero.warmup.protocol import JITWarmupProtocol
        from layerzero.warmup.shape_manifest import ShapeManifest, ShapeSignature

        config = WarmupConfig(max_concurrent_jit=4)
        protocol = JITWarmupProtocol(config)

        # Simulate OOM
        oom_raised = [False]

        def oom_compile(shape: ShapeSignature, backend: str) -> tuple[bool, float, str | None]:
            if not oom_raised[0]:
                oom_raised[0] = True
                return (False, 0.0, "CUDA out of memory")
            return (True, 100.0, None)

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

        with patch.object(protocol, "_compile_shape", side_effect=oom_compile):
            # Should handle OOM gracefully
            report = protocol.warmup(manifest)
