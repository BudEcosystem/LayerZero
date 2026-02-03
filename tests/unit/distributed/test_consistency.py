"""Tests for distributed version and selection consistency."""
from __future__ import annotations

import hashlib
import pytest
from unittest.mock import MagicMock, patch, call

import torch

from layerzero.distributed.consistency import (
    ConsistencyConfig,
    ConsistencyMode,
    SelectionHash,
    VersionChecker,
    SelectionSynchronizer,
    ConsistencyError,
    DistributedContext,
    get_distributed_context,
    is_distributed,
)
from layerzero.reasons import (
    VERSION_MISMATCH,
    SELECTION_HASH_MISMATCH,
    BROADCAST_FAILED,
    Reason,
    ReasonCategory,
)


class TestConsistencyConfig:
    """Tests for ConsistencyConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = ConsistencyConfig()

        assert config.mode == ConsistencyMode.STRICT
        assert config.broadcast_from_rank0 is True
        assert config.hash_algorithm == "sha256"
        assert config.fail_on_mismatch_training is True
        assert config.fallback_on_mismatch_inference is True

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = ConsistencyConfig(
            mode=ConsistencyMode.RELAXED,
            broadcast_from_rank0=False,
            hash_algorithm="md5",
            fail_on_mismatch_training=False,
            fallback_on_mismatch_inference=False,
        )

        assert config.mode == ConsistencyMode.RELAXED
        assert config.broadcast_from_rank0 is False
        assert config.hash_algorithm == "md5"
        assert config.fail_on_mismatch_training is False
        assert config.fallback_on_mismatch_inference is False

    def test_config_immutable(self) -> None:
        """Config is immutable."""
        config = ConsistencyConfig()

        with pytest.raises(AttributeError):
            config.mode = ConsistencyMode.RELAXED


class TestConsistencyMode:
    """Tests for ConsistencyMode enum."""

    def test_strict_mode(self) -> None:
        """STRICT mode value."""
        assert ConsistencyMode.STRICT.value == "strict"

    def test_relaxed_mode(self) -> None:
        """RELAXED mode value."""
        assert ConsistencyMode.RELAXED.value == "relaxed"

    def test_disabled_mode(self) -> None:
        """DISABLED mode value."""
        assert ConsistencyMode.DISABLED.value == "disabled"


class TestSelectionHash:
    """Tests for SelectionHash."""

    def test_hash_creation(self) -> None:
        """SelectionHash stores hash value."""
        hash_val = hashlib.sha256(b"test").hexdigest()
        sel_hash = SelectionHash(hash_value=hash_val, kernel_id="kernel1")

        assert sel_hash.hash_value == hash_val
        assert sel_hash.kernel_id == "kernel1"

    def test_hash_equality(self) -> None:
        """SelectionHash equality comparison."""
        hash1 = SelectionHash(hash_value="abc123", kernel_id="kernel1")
        hash2 = SelectionHash(hash_value="abc123", kernel_id="kernel1")
        hash3 = SelectionHash(hash_value="def456", kernel_id="kernel1")

        assert hash1 == hash2
        assert hash1 != hash3

    def test_compute_hash(self) -> None:
        """SelectionHash.compute creates hash from kernel info."""
        kernel_id = "test_kernel"
        operation = "attention"
        config = {"batch_size": 4, "seq_len": 2048}

        hash_obj = SelectionHash.compute(
            kernel_id=kernel_id,
            operation=operation,
            config=config,
        )

        assert hash_obj.kernel_id == kernel_id
        assert len(hash_obj.hash_value) == 64  # SHA256 hex digest length

    def test_hash_deterministic(self) -> None:
        """Hash is deterministic for same inputs."""
        hash1 = SelectionHash.compute(
            kernel_id="kernel1",
            operation="attention",
            config={"a": 1, "b": 2},
        )
        hash2 = SelectionHash.compute(
            kernel_id="kernel1",
            operation="attention",
            config={"a": 1, "b": 2},
        )

        assert hash1.hash_value == hash2.hash_value


class TestDistributedContext:
    """Tests for DistributedContext."""

    def test_context_non_distributed(self, mock_dist_unavailable) -> None:
        """Context when not distributed."""
        ctx = get_distributed_context()

        assert ctx.rank == 0
        assert ctx.world_size == 1
        assert ctx.is_distributed is False

    def test_context_distributed_rank0(self, mock_rank_0) -> None:
        """Context for rank 0 in distributed setting."""
        ctx = get_distributed_context()

        assert ctx.rank == 0
        assert ctx.world_size == 4
        assert ctx.is_distributed is True
        assert ctx.is_rank0 is True

    def test_context_distributed_rank1(self, mock_rank_1) -> None:
        """Context for rank 1 in distributed setting."""
        ctx = get_distributed_context()

        assert ctx.rank == 1
        assert ctx.world_size == 4
        assert ctx.is_distributed is True
        assert ctx.is_rank0 is False

    def test_is_distributed_helper(self, mock_dist_available) -> None:
        """is_distributed helper function."""
        assert is_distributed() is True

    def test_is_distributed_when_unavailable(self, mock_dist_unavailable) -> None:
        """is_distributed when torch.distributed unavailable."""
        assert is_distributed() is False


class TestVersionChecker:
    """Tests for VersionChecker."""

    def test_version_check_same(self, mock_rank_0) -> None:
        """Version check passes when versions match."""
        checker = VersionChecker()

        # Mock broadcast to return same version
        with patch.object(checker, '_broadcast_version', return_value="1.0.0"):
            passed, reason = checker.check_version("1.0.0")

        assert passed is True
        assert reason is None

    def test_version_broadcast_from_rank0(self, mock_rank_0) -> None:
        """Version is broadcast from rank 0."""
        checker = VersionChecker()

        with patch('torch.distributed.broadcast_object_list') as mock_broadcast:
            checker._broadcast_version("1.0.0")
            mock_broadcast.assert_called_once()

    def test_version_received_on_other_ranks(self, mock_rank_1) -> None:
        """Non-root ranks receive version from rank 0."""
        checker = VersionChecker()

        with patch('torch.distributed.broadcast_object_list') as mock_broadcast:
            # Simulate receiving version from rank 0
            mock_broadcast.side_effect = lambda obj_list, src: obj_list.__setitem__(0, "1.0.0")
            version = checker._broadcast_version("1.0.0")
            assert version == "1.0.0"

    def test_version_mismatch_training_fails(self, mock_rank_1) -> None:
        """Version mismatch fails in training mode."""
        config = ConsistencyConfig(fail_on_mismatch_training=True)
        checker = VersionChecker(config=config)

        with patch.object(checker, '_broadcast_version', return_value="2.0.0"):
            passed, reason = checker.check_version("1.0.0", is_training=True)

        assert passed is False
        assert reason is not None
        assert reason.code == VERSION_MISMATCH
        assert reason.category == ReasonCategory.DISTRIBUTED

    def test_version_mismatch_inference_fallback(self, mock_rank_1) -> None:
        """Version mismatch uses fallback in inference mode."""
        config = ConsistencyConfig(
            fail_on_mismatch_training=True,
            fallback_on_mismatch_inference=True,
        )
        checker = VersionChecker(config=config)

        with patch.object(checker, '_broadcast_version', return_value="2.0.0"):
            passed, reason = checker.check_version("1.0.0", is_training=False)

        # Inference mode should use fallback (pass but warn)
        assert passed is True
        assert reason is None  # Fallback means no error

    def test_version_check_non_distributed(self, mock_dist_unavailable) -> None:
        """Version check passes when not distributed."""
        checker = VersionChecker()

        passed, reason = checker.check_version("1.0.0")

        assert passed is True
        assert reason is None


class TestSelectionSynchronizer:
    """Tests for SelectionSynchronizer."""

    def test_selection_synchronized(self, mock_rank_0) -> None:
        """Selection synchronized across ranks."""
        sync = SelectionSynchronizer()

        selection_hash = SelectionHash(hash_value="abc123", kernel_id="kernel1")

        with patch.object(sync, '_broadcast_hash', return_value=selection_hash):
            result = sync.synchronize(selection_hash)

        assert result == selection_hash

    def test_selection_hash_compared(self, mock_rank_1) -> None:
        """Selection hash is compared across ranks."""
        sync = SelectionSynchronizer()

        local_hash = SelectionHash(hash_value="abc123", kernel_id="kernel1")
        remote_hash = SelectionHash(hash_value="abc123", kernel_id="kernel1")

        with patch.object(sync, '_broadcast_hash', return_value=remote_hash):
            result = sync.synchronize(local_hash)

        assert result == local_hash

    def test_selection_broadcast_on_mismatch(self, mock_rank_1) -> None:
        """Selection is broadcast from rank 0 when ranks disagree."""
        config = ConsistencyConfig(broadcast_from_rank0=True)
        sync = SelectionSynchronizer(config=config)

        local_hash = SelectionHash(hash_value="abc123", kernel_id="kernel1")
        rank0_hash = SelectionHash(hash_value="def456", kernel_id="kernel1")

        with patch.object(sync, '_broadcast_hash', return_value=rank0_hash):
            result = sync.synchronize(local_hash)

        # Should use rank 0's selection
        assert result == rank0_hash

    def test_selection_mismatch_error_strict(self, mock_rank_1) -> None:
        """Selection mismatch raises error in strict mode with no broadcast."""
        config = ConsistencyConfig(
            mode=ConsistencyMode.STRICT,
            broadcast_from_rank0=False,
        )
        sync = SelectionSynchronizer(config=config)

        local_hash = SelectionHash(hash_value="abc123", kernel_id="kernel1")
        remote_hash = SelectionHash(hash_value="def456", kernel_id="kernel1")

        with patch.object(sync, '_broadcast_hash', return_value=remote_hash):
            with pytest.raises(ConsistencyError) as exc_info:
                sync.synchronize(local_hash)

        assert "mismatch" in str(exc_info.value).lower()

    def test_synchronize_non_distributed(self, mock_dist_unavailable) -> None:
        """Synchronize returns local hash when not distributed."""
        sync = SelectionSynchronizer()

        local_hash = SelectionHash(hash_value="abc123", kernel_id="kernel1")
        result = sync.synchronize(local_hash)

        assert result == local_hash

    def test_broadcast_failure_handled(self, mock_rank_0) -> None:
        """Broadcast failure is handled gracefully."""
        sync = SelectionSynchronizer()

        local_hash = SelectionHash(hash_value="abc123", kernel_id="kernel1")

        with patch('torch.distributed.broadcast_object_list', side_effect=RuntimeError("Broadcast failed")):
            with pytest.raises(ConsistencyError) as exc_info:
                sync.synchronize(local_hash)

        assert "broadcast" in str(exc_info.value).lower()


class TestConsistencyIntegration:
    """Integration tests for consistency module."""

    def test_full_consistency_check_distributed(self, mock_rank_0) -> None:
        """Full consistency check in distributed setting."""
        checker = VersionChecker()
        sync = SelectionSynchronizer()

        # Check version
        with patch.object(checker, '_broadcast_version', return_value="1.0.0"):
            version_ok, _ = checker.check_version("1.0.0")

        assert version_ok is True

        # Synchronize selection
        selection_hash = SelectionHash.compute(
            kernel_id="kernel1",
            operation="attention",
            config={"batch_size": 4},
        )

        with patch.object(sync, '_broadcast_hash', return_value=selection_hash):
            result = sync.synchronize(selection_hash)

        assert result == selection_hash

    def test_relaxed_mode_allows_differences(self, mock_rank_1) -> None:
        """Relaxed mode allows minor differences."""
        config = ConsistencyConfig(mode=ConsistencyMode.RELAXED)
        sync = SelectionSynchronizer(config=config)

        local_hash = SelectionHash(hash_value="abc123", kernel_id="kernel1")
        remote_hash = SelectionHash(hash_value="def456", kernel_id="kernel1")

        # In relaxed mode, should use rank 0's selection without error
        with patch.object(sync, '_broadcast_hash', return_value=remote_hash):
            result = sync.synchronize(local_hash)

        assert result == remote_hash

    def test_disabled_mode_skips_checks(self, mock_rank_1) -> None:
        """Disabled mode skips all consistency checks."""
        config = ConsistencyConfig(mode=ConsistencyMode.DISABLED)
        sync = SelectionSynchronizer(config=config)

        local_hash = SelectionHash(hash_value="abc123", kernel_id="kernel1")

        # Should return local hash without any broadcast
        result = sync.synchronize(local_hash)

        assert result == local_hash
