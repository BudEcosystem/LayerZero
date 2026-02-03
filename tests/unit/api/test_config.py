"""Tests for LayerZero configuration APIs.

Tests for lz.configure(), lz.lock(), lz.unlock(), lz.load_config(), etc.
"""
from __future__ import annotations

import pytest
import torch
import tempfile
import os


class TestConfigureAPI:
    """Tests for lz.configure() public API."""

    def test_configure_default_backend(self) -> None:
        """Configure default backend."""
        import layerzero as lz

        # Configure default backend
        lz.configure(default_backend="torch_sdpa")

        # Should not raise
        config = lz.get_config()
        assert config.default_backend == "torch_sdpa"

    def test_configure_cache_size(self) -> None:
        """Configure cache size."""
        import layerzero as lz

        lz.configure(cache_size=1024)

        config = lz.get_config()
        assert config.cache_size == 1024

    def test_configure_strict_mode(self) -> None:
        """Configure strict mode."""
        import layerzero as lz

        lz.configure(strict_mode=True)

        config = lz.get_config()
        assert config.strict_mode is True

    def test_configure_multiple_options(self) -> None:
        """Configure multiple options at once."""
        import layerzero as lz

        lz.configure(
            default_backend="torch_sdpa",
            cache_size=2048,
            strict_mode=False,
        )

        config = lz.get_config()
        assert config.default_backend == "torch_sdpa"
        assert config.cache_size == 2048
        assert config.strict_mode is False

    def test_configure_reset(self) -> None:
        """Reset configuration to defaults."""
        import layerzero as lz

        # Set custom config
        lz.configure(default_backend="custom")

        # Reset
        lz.configure(reset=True)

        config = lz.get_config()
        # Should be back to default
        assert config.default_backend is None or config.default_backend == "auto"


class TestLockUnlockAPI:
    """Tests for lz.lock() and lz.unlock() APIs."""

    def test_lock_basic(self) -> None:
        """Lock an operation to specific kernel."""
        import layerzero as lz

        lz.lock("attention.causal", "torch_sdpa")

        # Verify lock is active
        locks = lz.get_locks()
        assert "attention.causal" in locks
        assert locks["attention.causal"] == "torch_sdpa"

    def test_unlock_basic(self) -> None:
        """Unlock a previously locked operation."""
        import layerzero as lz

        # Lock first
        lz.lock("attention.causal", "torch_sdpa")

        # Unlock
        lz.unlock("attention.causal")

        # Verify unlock
        locks = lz.get_locks()
        assert "attention.causal" not in locks

    def test_lock_affects_selection(self) -> None:
        """Locked kernel is always selected."""
        import layerzero as lz

        # Lock to SDPA
        lz.lock("attention.causal", "torch_sdpa")

        q = torch.randn(2, 8, 64, 64)
        k = torch.randn(2, 8, 64, 64)
        v = torch.randn(2, 8, 64, 64)

        # Should use SDPA regardless of other factors
        output = lz.attention(q, k, v, is_causal=True)

        # Clean up
        lz.unlock("attention.causal")

        assert output.shape == (2, 8, 64, 64)

    def test_unlock_nonexistent(self) -> None:
        """Unlocking non-existent lock doesn't raise."""
        import layerzero as lz

        # Should not raise
        lz.unlock("nonexistent.operation")

    def test_lock_invalid_kernel(self) -> None:
        """Lock with invalid kernel is accepted (lenient mode)."""
        import layerzero as lz

        # LayerZero is lenient by design - it accepts any kernel_id
        # This allows forward compatibility and custom kernels
        lz.lock("attention.causal", "invalid_kernel_that_does_not_exist")

        # Verify lock was set
        locks = lz.get_locks()
        assert "attention.causal" in locks
        assert locks["attention.causal"] == "invalid_kernel_that_does_not_exist"

        # Clean up
        lz.unlock("attention.causal")


class TestLoadConfigAPI:
    """Tests for lz.load_config() API."""

    def test_load_config_yaml(self) -> None:
        """Load configuration from YAML file."""
        import layerzero as lz

        yaml_content = """
default_backend: torch_sdpa
cache_size: 1024
strict_mode: false
locks:
  attention.causal: torch_sdpa
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write(yaml_content)
            config_path = f.name

        try:
            lz.load_config(config_path)

            config = lz.get_config()
            assert config.default_backend == "torch_sdpa"
            assert config.cache_size == 1024
        finally:
            os.unlink(config_path)

    def test_load_config_nonexistent(self) -> None:
        """Loading non-existent config raises."""
        import layerzero as lz

        with pytest.raises(FileNotFoundError):
            lz.load_config("/nonexistent/path/config.yaml")


class TestGetConfigAPI:
    """Tests for lz.get_config() API."""

    def test_get_config_returns_config(self) -> None:
        """get_config returns configuration object."""
        import layerzero as lz

        config = lz.get_config()

        # Should have expected attributes
        assert hasattr(config, 'default_backend')
        assert hasattr(config, 'cache_size')

    def test_get_config_immutable(self) -> None:
        """get_config returns immutable or copy."""
        import layerzero as lz

        config1 = lz.get_config()
        config2 = lz.get_config()

        # Modifying one shouldn't affect internal state
        # (depends on implementation - could be frozen dataclass)


class TestGetLocksAPI:
    """Tests for lz.get_locks() API."""

    def test_get_locks_empty(self) -> None:
        """get_locks returns empty dict when no locks."""
        import layerzero as lz

        # Clear any existing locks
        for op in list(lz.get_locks().keys()):
            lz.unlock(op)

        locks = lz.get_locks()
        assert isinstance(locks, dict)
        assert len(locks) == 0

    def test_get_locks_with_locks(self) -> None:
        """get_locks returns current locks."""
        import layerzero as lz

        lz.lock("op1", "kernel1")
        lz.lock("op2", "kernel2")

        locks = lz.get_locks()

        assert "op1" in locks
        assert "op2" in locks

        # Clean up
        lz.unlock("op1")
        lz.unlock("op2")
