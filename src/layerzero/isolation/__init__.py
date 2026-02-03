"""
Process isolation module for backend management.

This module provides:
- SubprocessBackend: Isolated subprocess backend
- SubprocessBackendConfig: Configuration for subprocess
- IPCChannel: IPC communication channel
- IPCMessage: IPC message format
- ABIConflictDetector: Detects ABI conflicts between backends
- ABIInfo: ABI information for a backend
"""
from __future__ import annotations

from layerzero.isolation.subprocess_backend import (
    SubprocessBackend,
    SubprocessBackendConfig,
    SubprocessError,
    SubprocessState,
)
from layerzero.isolation.ipc import (
    IPCChannel,
    IPCConfig,
    IPCMessage,
    IPCMessageType,
    SharedMemoryChannel,
)
from layerzero.isolation.abi_detector import (
    ABIConflictDetector,
    ABIInfo,
    ConflictResult,
    detect_abi_conflict,
)

__all__ = [
    # Subprocess backend
    "SubprocessBackend",
    "SubprocessBackendConfig",
    "SubprocessError",
    "SubprocessState",
    # IPC
    "IPCChannel",
    "IPCConfig",
    "IPCMessage",
    "IPCMessageType",
    "SharedMemoryChannel",
    # ABI detection
    "ABIConflictDetector",
    "ABIInfo",
    "ConflictResult",
    "detect_abi_conflict",
]
