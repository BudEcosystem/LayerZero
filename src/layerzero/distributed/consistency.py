"""
Distributed version and selection consistency.

This module provides:
- ConsistencyConfig: Configuration for consistency checks
- ConsistencyMode: Strict/relaxed/disabled modes
- SelectionHash: Hash for selection verification
- VersionChecker: Checks LayerZero version across ranks
- SelectionSynchronizer: Synchronizes selection across ranks
- DistributedContext: Context for distributed execution
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass
from enum import Enum, unique
from typing import TYPE_CHECKING, Any

import torch

from layerzero.reasons import (
    BROADCAST_FAILED,
    SELECTION_HASH_MISMATCH,
    VERSION_MISMATCH,
    Reason,
    ReasonCategory,
)

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ConsistencyError(Exception):
    """Error raised when consistency check fails."""

    pass


@unique
class ConsistencyMode(str, Enum):
    """Consistency check modes.

    Attributes:
        STRICT: Require exact match, fail on mismatch.
        RELAXED: Use rank 0's selection on mismatch.
        DISABLED: Skip all consistency checks.
    """

    STRICT = "strict"
    RELAXED = "relaxed"
    DISABLED = "disabled"


@dataclass(frozen=True)
class ConsistencyConfig:
    """Configuration for distributed consistency.

    Attributes:
        mode: Consistency mode (strict, relaxed, disabled).
        broadcast_from_rank0: Broadcast selection from rank 0 on mismatch.
        hash_algorithm: Hash algorithm for selection verification.
        fail_on_mismatch_training: Fail on mismatch during training.
        fallback_on_mismatch_inference: Use fallback on mismatch during inference.
    """

    mode: ConsistencyMode = ConsistencyMode.STRICT
    broadcast_from_rank0: bool = True
    hash_algorithm: str = "sha256"
    fail_on_mismatch_training: bool = True
    fallback_on_mismatch_inference: bool = True


@dataclass(frozen=True)
class DistributedContext:
    """Context for distributed execution.

    Attributes:
        rank: Current process rank.
        world_size: Total number of processes.
        is_distributed: Whether running in distributed mode.
    """

    rank: int
    world_size: int
    is_distributed: bool

    @property
    def is_rank0(self) -> bool:
        """Check if this is rank 0."""
        return self.rank == 0


def is_distributed() -> bool:
    """Check if running in distributed mode.

    Returns:
        True if torch.distributed is available and initialized.
    """
    if not torch.distributed.is_available():
        return False
    if not torch.distributed.is_initialized():
        return False
    return True


def get_distributed_context() -> DistributedContext:
    """Get current distributed context.

    Returns:
        DistributedContext with rank and world size info.
    """
    if not is_distributed():
        return DistributedContext(
            rank=0,
            world_size=1,
            is_distributed=False,
        )

    return DistributedContext(
        rank=torch.distributed.get_rank(),
        world_size=torch.distributed.get_world_size(),
        is_distributed=True,
    )


@dataclass(frozen=True)
class SelectionHash:
    """Hash for selection verification.

    Used to verify that all ranks have made the same selection
    decision for a given kernel.

    Attributes:
        hash_value: SHA-256 hash of selection parameters.
        kernel_id: Kernel identifier for this selection.
    """

    hash_value: str
    kernel_id: str

    @classmethod
    def compute(
        cls,
        kernel_id: str,
        operation: str,
        config: dict[str, Any],
        algorithm: str = "sha256",
    ) -> SelectionHash:
        """Compute hash from selection parameters.

        Args:
            kernel_id: Kernel identifier.
            operation: Operation type (attention, matmul, etc.).
            config: Configuration dictionary.
            algorithm: Hash algorithm (default: sha256).

        Returns:
            SelectionHash with computed hash value.
        """
        # Create deterministic string representation
        data = {
            "kernel_id": kernel_id,
            "operation": operation,
            "config": config,
        }
        # Sort keys for deterministic ordering
        data_str = json.dumps(data, sort_keys=True)

        if algorithm == "sha256":
            hash_val = hashlib.sha256(data_str.encode()).hexdigest()
        elif algorithm == "md5":
            hash_val = hashlib.md5(data_str.encode()).hexdigest()
        else:
            hash_val = hashlib.sha256(data_str.encode()).hexdigest()

        return cls(hash_value=hash_val, kernel_id=kernel_id)


class VersionChecker:
    """Checks LayerZero version consistency across ranks.

    Ensures all ranks are using the same LayerZero version
    to prevent subtle bugs from version mismatches.

    Example:
        checker = VersionChecker()

        passed, reason = checker.check_version("1.0.0")
        if not passed:
            raise ConsistencyError(str(reason))
    """

    def __init__(self, config: ConsistencyConfig | None = None) -> None:
        """Initialize version checker.

        Args:
            config: Consistency configuration.
        """
        self._config = config or ConsistencyConfig()

    @property
    def config(self) -> ConsistencyConfig:
        """Get configuration."""
        return self._config

    def check_version(
        self,
        local_version: str,
        is_training: bool = False,
    ) -> tuple[bool, Reason | None]:
        """Check version consistency across ranks.

        Args:
            local_version: Local LayerZero version.
            is_training: Whether in training mode.

        Returns:
            Tuple of (passed, reason). Reason is None if passed.
        """
        if not is_distributed():
            return True, None

        if self._config.mode == ConsistencyMode.DISABLED:
            return True, None

        try:
            # Get version from rank 0
            rank0_version = self._broadcast_version(local_version)

            if local_version != rank0_version:
                logger.warning(
                    "Version mismatch: local=%s, rank0=%s",
                    local_version,
                    rank0_version,
                )

                if is_training and self._config.fail_on_mismatch_training:
                    reason = Reason(
                        code=VERSION_MISMATCH,
                        message=(
                            f"LayerZero version mismatch: local {local_version} "
                            f"vs rank 0 {rank0_version}"
                        ),
                        category=ReasonCategory.DISTRIBUTED,
                    )
                    return False, reason

                if not is_training and self._config.fallback_on_mismatch_inference:
                    # Inference mode with fallback - proceed with warning
                    logger.warning(
                        "Proceeding with version mismatch in inference mode"
                    )
                    return True, None

            return True, None

        except Exception as e:
            logger.error("Version check failed: %s", e)
            reason = Reason(
                code=BROADCAST_FAILED,
                message=f"Version broadcast failed: {e}",
                category=ReasonCategory.DISTRIBUTED,
            )
            return False, reason

    def _broadcast_version(self, local_version: str) -> str:
        """Broadcast version from rank 0.

        Args:
            local_version: Local version string.

        Returns:
            Version string from rank 0.
        """
        ctx = get_distributed_context()

        # Create list for broadcast
        obj_list = [local_version if ctx.is_rank0 else ""]

        # Broadcast from rank 0
        torch.distributed.broadcast_object_list(obj_list, src=0)

        return obj_list[0]


class SelectionSynchronizer:
    """Synchronizes kernel selection across distributed ranks.

    Ensures all ranks use the same kernel selection to maintain
    consistent behavior in distributed training/inference.

    Example:
        sync = SelectionSynchronizer()

        local_hash = SelectionHash.compute(kernel_id="kernel1", ...)
        synced_hash = sync.synchronize(local_hash)
    """

    def __init__(self, config: ConsistencyConfig | None = None) -> None:
        """Initialize selection synchronizer.

        Args:
            config: Consistency configuration.
        """
        self._config = config or ConsistencyConfig()

    @property
    def config(self) -> ConsistencyConfig:
        """Get configuration."""
        return self._config

    def synchronize(self, local_hash: SelectionHash) -> SelectionHash:
        """Synchronize selection across ranks.

        Args:
            local_hash: Local selection hash.

        Returns:
            Synchronized selection hash (may be from rank 0).

        Raises:
            ConsistencyError: If synchronization fails in strict mode.
        """
        if not is_distributed():
            return local_hash

        if self._config.mode == ConsistencyMode.DISABLED:
            return local_hash

        try:
            # Get hash from rank 0
            rank0_hash = self._broadcast_hash(local_hash)

            if local_hash.hash_value != rank0_hash.hash_value:
                logger.warning(
                    "Selection hash mismatch for %s: local=%s, rank0=%s",
                    local_hash.kernel_id,
                    local_hash.hash_value[:16],
                    rank0_hash.hash_value[:16],
                )

                if self._config.broadcast_from_rank0:
                    # Use rank 0's selection
                    logger.info(
                        "Using rank 0 selection for %s",
                        local_hash.kernel_id,
                    )
                    return rank0_hash

                if self._config.mode == ConsistencyMode.STRICT:
                    raise ConsistencyError(
                        f"Selection hash mismatch for {local_hash.kernel_id}: "
                        f"local {local_hash.hash_value[:16]} vs "
                        f"rank0 {rank0_hash.hash_value[:16]}"
                    )

                # Relaxed mode - use rank 0's selection
                return rank0_hash

            return local_hash

        except ConsistencyError:
            raise
        except Exception as e:
            logger.error("Selection synchronization failed: %s", e)
            raise ConsistencyError(f"Selection broadcast failed: {e}") from e

    def _broadcast_hash(self, local_hash: SelectionHash) -> SelectionHash:
        """Broadcast selection hash from rank 0.

        Args:
            local_hash: Local selection hash.

        Returns:
            Selection hash from rank 0.
        """
        ctx = get_distributed_context()

        # Create list for broadcast
        if ctx.is_rank0:
            obj_list = [(local_hash.hash_value, local_hash.kernel_id)]
        else:
            obj_list = [("", "")]

        # Broadcast from rank 0
        torch.distributed.broadcast_object_list(obj_list, src=0)

        hash_value, kernel_id = obj_list[0]
        return SelectionHash(hash_value=hash_value, kernel_id=kernel_id)


# Global instances
_global_version_checker: VersionChecker | None = None
_global_synchronizer: SelectionSynchronizer | None = None


def get_global_version_checker() -> VersionChecker:
    """Get global version checker instance.

    Returns:
        Global VersionChecker instance.
    """
    global _global_version_checker
    if _global_version_checker is None:
        _global_version_checker = VersionChecker()
    return _global_version_checker


def get_global_synchronizer() -> SelectionSynchronizer:
    """Get global selection synchronizer instance.

    Returns:
        Global SelectionSynchronizer instance.
    """
    global _global_synchronizer
    if _global_synchronizer is None:
        _global_synchronizer = SelectionSynchronizer()
    return _global_synchronizer
