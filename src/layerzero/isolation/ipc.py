"""
IPC communication for subprocess backends.

This module provides:
- IPCConfig: Configuration for IPC
- IPCMessageType: Message type enum
- IPCMessage: Message format for IPC
- IPCChannel: Communication channel
- SharedMemoryChannel: Shared memory based channel
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from enum import Enum, unique
from multiprocessing.shared_memory import SharedMemory
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IPCConfig:
    """Configuration for IPC.

    Attributes:
        buffer_size: Size of communication buffer in bytes.
        timeout_ms: Timeout for operations in milliseconds.
        use_shared_memory: Use shared memory for communication.
    """

    buffer_size: int = 4096
    timeout_ms: int = 5000
    use_shared_memory: bool = True


@unique
class IPCMessageType(str, Enum):
    """Message types for IPC.

    Attributes:
        REQUEST: Request from parent to subprocess.
        RESPONSE: Response from subprocess to parent.
        ERROR: Error message.
        HEARTBEAT: Heartbeat message.
    """

    REQUEST = "request"
    RESPONSE = "response"
    ERROR = "error"
    HEARTBEAT = "heartbeat"


@dataclass
class IPCMessage:
    """Message format for IPC.

    Attributes:
        msg_type: Type of message.
        payload: Message payload.
        msg_id: Unique message identifier.
        timestamp: Message timestamp.
    """

    msg_type: IPCMessageType
    payload: dict[str, Any]
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = 0.0

    def serialize(self) -> bytes:
        """Serialize message to bytes.

        Returns:
            Serialized message bytes.
        """
        data = {
            "msg_type": self.msg_type.value,
            "payload": self.payload,
            "msg_id": self.msg_id,
            "timestamp": self.timestamp,
        }
        return json.dumps(data).encode('utf-8')

    @classmethod
    def deserialize(cls, data: bytes) -> IPCMessage:
        """Deserialize message from bytes.

        Args:
            data: Serialized message bytes.

        Returns:
            Deserialized IPCMessage.
        """
        obj = json.loads(data.decode('utf-8'))
        return cls(
            msg_type=IPCMessageType(obj["msg_type"]),
            payload=obj["payload"],
            msg_id=obj["msg_id"],
            timestamp=obj.get("timestamp", 0.0),
        )


class IPCChannel:
    """IPC communication channel.

    Provides send/receive interface for IPC communication
    with subprocess backends.

    Example:
        config = IPCConfig()
        channel = IPCChannel(config=config)

        msg = IPCMessage(IPCMessageType.REQUEST, {"op": "test"})
        channel.send(msg)
        response = channel.receive()
    """

    def __init__(self, config: IPCConfig | None = None) -> None:
        """Initialize IPC channel.

        Args:
            config: IPC configuration.
        """
        self._config = config or IPCConfig()
        self._closed = False
        self._buffer: list[IPCMessage] = []

    @property
    def config(self) -> IPCConfig:
        """Get configuration."""
        return self._config

    @property
    def is_closed(self) -> bool:
        """Check if channel is closed."""
        return self._closed

    def send(self, message: IPCMessage) -> None:
        """Send message to channel.

        Args:
            message: Message to send.

        Raises:
            RuntimeError: If channel is closed.
        """
        if self._closed:
            raise RuntimeError("Channel is closed")

        self._internal_send(message)

    def receive(self, timeout_ms: int | None = None) -> IPCMessage | None:
        """Receive message from channel.

        Args:
            timeout_ms: Timeout in milliseconds.

        Returns:
            Received message or None on timeout.

        Raises:
            RuntimeError: If channel is closed.
        """
        if self._closed:
            raise RuntimeError("Channel is closed")

        return self._internal_receive(timeout_ms)

    def close(self) -> None:
        """Close the channel."""
        self._closed = True
        logger.debug("IPC channel closed")

    def _internal_send(self, message: IPCMessage) -> None:
        """Internal send implementation.

        Args:
            message: Message to send.
        """
        # Default implementation uses internal buffer
        # Subclasses override for actual IPC
        logger.debug("Sending message: %s", message.msg_id)

    def _internal_receive(self, timeout_ms: int | None = None) -> IPCMessage | None:
        """Internal receive implementation.

        Args:
            timeout_ms: Timeout in milliseconds.

        Returns:
            Received message or None.
        """
        # Default implementation - subclasses override
        return None


class SharedMemoryChannel(IPCChannel):
    """Shared memory based IPC channel.

    Uses shared memory for efficient data transfer between
    processes without copying through kernel.

    Example:
        config = IPCConfig(use_shared_memory=True)
        channel = SharedMemoryChannel(config=config, name="backend_channel")
    """

    def __init__(
        self,
        config: IPCConfig | None = None,
        name: str | None = None,
    ) -> None:
        """Initialize shared memory channel.

        Args:
            config: IPC configuration.
            name: Shared memory name.
        """
        super().__init__(config=config)
        self._name = name or f"lz_ipc_{uuid.uuid4().hex[:8]}"
        self._shm: SharedMemory | None = None

        try:
            self._shm = SharedMemory(name=self._name, create=True, size=self._config.buffer_size)
            logger.debug("Created shared memory: %s", self._name)
        except Exception as e:
            logger.warning("Failed to create shared memory: %s", e)

    @property
    def name(self) -> str:
        """Get shared memory name."""
        return self._name

    def write(self, data: bytes) -> None:
        """Write data to shared memory.

        Args:
            data: Data to write.
        """
        if self._shm is None:
            raise RuntimeError("Shared memory not initialized")

        if len(data) > self._config.buffer_size:
            raise ValueError(f"Data size {len(data)} exceeds buffer size {self._config.buffer_size}")

        # Write length prefix and data
        length = len(data).to_bytes(4, 'big')
        self._shm.buf[:4] = length
        self._shm.buf[4:4+len(data)] = data

    def read(self) -> bytes:
        """Read data from shared memory.

        Returns:
            Read data bytes.
        """
        if self._shm is None:
            raise RuntimeError("Shared memory not initialized")

        # Read length prefix
        length = int.from_bytes(self._shm.buf[:4], 'big')
        if length == 0 or length > self._config.buffer_size - 4:
            return b""

        return bytes(self._shm.buf[4:4+length])

    def close(self) -> None:
        """Close and cleanup shared memory."""
        super().close()

        if self._shm is not None:
            try:
                self._shm.close()
                self._shm.unlink()
            except Exception as e:
                logger.warning("Error closing shared memory: %s", e)
            self._shm = None
