"""
ABI conflict detection for backends.

This module provides:
- ABIInfo: ABI information for a backend
- ConflictResult: Result of conflict detection
- ABIConflictDetector: Detects ABI conflicts
- detect_abi_conflict: Convenience function
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ABIInfo:
    """ABI information for a backend.

    Attributes:
        backend_id: Unique backend identifier.
        abi_version: ABI version string.
        torch_version: PyTorch version.
        cuda_version: CUDA version.
    """

    backend_id: str
    abi_version: str
    torch_version: str
    cuda_version: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dict representation.
        """
        return {
            "backend_id": self.backend_id,
            "abi_version": self.abi_version,
            "torch_version": self.torch_version,
            "cuda_version": self.cuda_version,
        }


@dataclass
class ConflictResult:
    """Result of ABI conflict detection.

    Attributes:
        has_conflict: Whether conflicts were detected.
        conflicting_backends: List of conflicting backend IDs.
        details: Details about the conflict.
    """

    has_conflict: bool
    conflicting_backends: list[str]
    details: str = ""


class ABIConflictDetector:
    """Detects ABI conflicts between backends.

    Tracks ABI information for registered backends and detects
    incompatibilities that require process isolation.

    Example:
        detector = ABIConflictDetector()

        detector.register(ABIInfo("backend_a", "1.0", "2.0.0", "11.8"))
        detector.register(ABIInfo("backend_b", "2.0", "2.0.0", "11.8"))

        result = detector.detect()
        if result.has_conflict:
            isolated = detector.get_backends_requiring_isolation()
    """

    def __init__(self) -> None:
        """Initialize ABI conflict detector."""
        self._backends: dict[str, ABIInfo] = {}
        self._conflicts: list[tuple[str, str]] = []

    def register(self, info: ABIInfo) -> None:
        """Register backend ABI information.

        Args:
            info: ABI information for backend.
        """
        self._backends[info.backend_id] = info
        logger.debug(
            "Registered backend %s with ABI %s",
            info.backend_id,
            info.abi_version,
        )

    def unregister(self, backend_id: str) -> None:
        """Unregister backend.

        Args:
            backend_id: Backend identifier.
        """
        if backend_id in self._backends:
            del self._backends[backend_id]
            # Clear related conflicts
            self._conflicts = [
                (a, b) for a, b in self._conflicts
                if a != backend_id and b != backend_id
            ]

    def detect(self) -> ConflictResult:
        """Detect ABI conflicts between registered backends.

        Returns:
            ConflictResult with detected conflicts.
        """
        self._conflicts.clear()
        conflicting_backends: set[str] = set()
        details_parts: list[str] = []

        backend_list = list(self._backends.values())

        for i, info_a in enumerate(backend_list):
            for info_b in backend_list[i+1:]:
                conflict = self._check_conflict(info_a, info_b)

                if conflict:
                    self._conflicts.append((info_a.backend_id, info_b.backend_id))
                    conflicting_backends.add(info_a.backend_id)
                    conflicting_backends.add(info_b.backend_id)
                    details_parts.append(conflict)

        has_conflict = len(self._conflicts) > 0

        if has_conflict:
            logger.warning(
                "ABI conflicts detected: %d conflicts between %d backends",
                len(self._conflicts),
                len(conflicting_backends),
            )

        return ConflictResult(
            has_conflict=has_conflict,
            conflicting_backends=list(conflicting_backends),
            details="; ".join(details_parts),
        )

    def get_backends_requiring_isolation(self) -> list[str]:
        """Get list of backends that require process isolation.

        Returns backends that have ABI conflicts with other backends.
        The strategy is to isolate the minority to minimize overhead.

        Returns:
            List of backend IDs requiring isolation.
        """
        if not self._conflicts:
            return []

        # Count conflicts per backend
        conflict_counts: dict[str, int] = {}
        for a, b in self._conflicts:
            conflict_counts[a] = conflict_counts.get(a, 0) + 1
            conflict_counts[b] = conflict_counts.get(b, 0) + 1

        # Return backends with fewer conflicts (minority isolation)
        if not conflict_counts:
            return []

        # Find the backend with most conflicts - it's likely the odd one out
        max_conflicts = max(conflict_counts.values())
        isolated = [
            bid for bid, count in conflict_counts.items()
            if count == max_conflicts
        ]

        # If all have same conflicts, isolate the one added later
        if len(isolated) == len(conflict_counts):
            # Return the last added backend
            return [list(conflict_counts.keys())[-1]]

        return isolated

    def _check_conflict(self, info_a: ABIInfo, info_b: ABIInfo) -> str | None:
        """Check for ABI conflict between two backends.

        Args:
            info_a: First backend ABI info.
            info_b: Second backend ABI info.

        Returns:
            Conflict description or None if no conflict.
        """
        # Check ABI version mismatch
        if info_a.abi_version != info_b.abi_version:
            return (
                f"ABI version mismatch: {info_a.backend_id}={info_a.abi_version} "
                f"vs {info_b.backend_id}={info_b.abi_version}"
            )

        # Check CUDA version mismatch (major version only)
        if info_a.cuda_version and info_b.cuda_version:
            cuda_major_a = info_a.cuda_version.split('.')[0]
            cuda_major_b = info_b.cuda_version.split('.')[0]

            if cuda_major_a != cuda_major_b:
                return (
                    f"CUDA major version mismatch: {info_a.backend_id}={info_a.cuda_version} "
                    f"vs {info_b.backend_id}={info_b.cuda_version}"
                )

        return None

    def clear(self) -> None:
        """Clear all registered backends and conflicts."""
        self._backends.clear()
        self._conflicts.clear()


def detect_abi_conflict(backends: list[ABIInfo]) -> ConflictResult:
    """Convenience function to detect ABI conflicts.

    Args:
        backends: List of backend ABI info.

    Returns:
        ConflictResult with detected conflicts.
    """
    detector = ABIConflictDetector()
    for info in backends:
        detector.register(info)
    return detector.detect()
