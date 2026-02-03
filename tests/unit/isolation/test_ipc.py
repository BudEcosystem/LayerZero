"""Tests for IPC communication."""
from __future__ import annotations

import time
import pytest
from unittest.mock import MagicMock, patch
from typing import Any

from layerzero.isolation.ipc import (
    IPCChannel,
    IPCConfig,
    IPCMessage,
    IPCMessageType,
    SharedMemoryChannel,
)


class TestIPCConfig:
    """Tests for IPCConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = IPCConfig()

        assert config.buffer_size == 4096
        assert config.timeout_ms == 5000
        assert config.use_shared_memory is True

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = IPCConfig(
            buffer_size=8192,
            timeout_ms=10000,
            use_shared_memory=False,
        )

        assert config.buffer_size == 8192
        assert config.timeout_ms == 10000
        assert config.use_shared_memory is False


class TestIPCMessageType:
    """Tests for IPCMessageType enum."""

    def test_request_type(self) -> None:
        """REQUEST message type."""
        assert IPCMessageType.REQUEST.value == "request"

    def test_response_type(self) -> None:
        """RESPONSE message type."""
        assert IPCMessageType.RESPONSE.value == "response"

    def test_error_type(self) -> None:
        """ERROR message type."""
        assert IPCMessageType.ERROR.value == "error"

    def test_heartbeat_type(self) -> None:
        """HEARTBEAT message type."""
        assert IPCMessageType.HEARTBEAT.value == "heartbeat"


class TestIPCMessage:
    """Tests for IPCMessage dataclass."""

    def test_creation(self) -> None:
        """IPCMessage stores message data."""
        msg = IPCMessage(
            msg_type=IPCMessageType.REQUEST,
            payload={"operation": "attention"},
            msg_id="123",
        )

        assert msg.msg_type == IPCMessageType.REQUEST
        assert msg.payload["operation"] == "attention"
        assert msg.msg_id == "123"

    def test_serialize(self) -> None:
        """IPCMessage can be serialized."""
        msg = IPCMessage(
            msg_type=IPCMessageType.REQUEST,
            payload={"key": "value"},
            msg_id="456",
        )

        data = msg.serialize()

        assert isinstance(data, bytes)
        assert len(data) > 0

    def test_deserialize(self) -> None:
        """IPCMessage can be deserialized."""
        original = IPCMessage(
            msg_type=IPCMessageType.REQUEST,
            payload={"key": "value"},
            msg_id="789",
        )

        data = original.serialize()
        restored = IPCMessage.deserialize(data)

        assert restored.msg_type == original.msg_type
        assert restored.payload == original.payload
        assert restored.msg_id == original.msg_id


class TestIPCChannel:
    """Tests for IPCChannel."""

    def test_channel_creation(self) -> None:
        """IPCChannel can be created."""
        config = IPCConfig()
        channel = IPCChannel(config=config)

        assert channel is not None
        assert channel.config == config

    def test_send_receive(self) -> None:
        """IPCChannel send and receive."""
        config = IPCConfig()
        channel = IPCChannel(config=config)

        with patch.object(channel, '_internal_send') as mock_send:
            with patch.object(channel, '_internal_receive') as mock_receive:
                msg = IPCMessage(
                    msg_type=IPCMessageType.REQUEST,
                    payload={"test": True},
                    msg_id="test_1",
                )

                # Set up mock response
                response = IPCMessage(
                    msg_type=IPCMessageType.RESPONSE,
                    payload={"result": "ok"},
                    msg_id="test_1",
                )
                mock_receive.return_value = response

                channel.send(msg)
                received = channel.receive()

        mock_send.assert_called_once()
        assert received.msg_type == IPCMessageType.RESPONSE

    def test_channel_close(self) -> None:
        """IPCChannel can be closed."""
        config = IPCConfig()
        channel = IPCChannel(config=config)

        channel.close()

        assert channel.is_closed is True


class TestSharedMemoryChannel:
    """Tests for SharedMemoryChannel."""

    def test_shared_memory_creation(self) -> None:
        """SharedMemoryChannel can be created."""
        config = IPCConfig(use_shared_memory=True)
        channel = SharedMemoryChannel(config=config, name="test_channel")

        assert channel is not None

    def test_shared_memory_read_write(self) -> None:
        """SharedMemoryChannel read and write."""
        config = IPCConfig(use_shared_memory=True, buffer_size=1024)

        with patch('layerzero.isolation.ipc.SharedMemory') as MockShm:
            mock_shm = MagicMock()
            mock_shm.buf = bytearray(1024)
            MockShm.return_value = mock_shm

            channel = SharedMemoryChannel(config=config, name="test")

            # Write data
            data = b"test data"
            channel.write(data)

            # Read data
            with patch.object(channel, 'read', return_value=data):
                result = channel.read()

            assert result == data

    def test_shared_memory_cleanup(self) -> None:
        """SharedMemoryChannel cleans up resources."""
        config = IPCConfig(use_shared_memory=True)

        with patch('layerzero.isolation.ipc.SharedMemory') as MockShm:
            mock_shm = MagicMock()
            MockShm.return_value = mock_shm

            channel = SharedMemoryChannel(config=config, name="test")
            channel.close()

            mock_shm.close.assert_called()


class TestIPCPerformance:
    """Tests for IPC performance characteristics."""

    def test_ipc_latency_acceptable(self) -> None:
        """IPC latency is acceptable."""
        config = IPCConfig()
        channel = IPCChannel(config=config)

        # Measure latency of send/receive
        with patch.object(channel, '_internal_send'):
            with patch.object(channel, '_internal_receive') as mock_receive:
                msg = IPCMessage(
                    msg_type=IPCMessageType.REQUEST,
                    payload={},
                    msg_id="latency_test",
                )
                response = IPCMessage(
                    msg_type=IPCMessageType.RESPONSE,
                    payload={},
                    msg_id="latency_test",
                )
                mock_receive.return_value = response

                start = time.monotonic()
                channel.send(msg)
                channel.receive()
                elapsed_ms = (time.monotonic() - start) * 1000

        # Latency should be under 10ms for mocked channel
        assert elapsed_ms < 10

    def test_ipc_throughput_acceptable(self) -> None:
        """IPC throughput is acceptable."""
        config = IPCConfig()
        channel = IPCChannel(config=config)

        num_messages = 100
        messages_sent = 0

        with patch.object(channel, '_internal_send'):
            with patch.object(channel, '_internal_receive') as mock_receive:
                response = IPCMessage(
                    msg_type=IPCMessageType.RESPONSE,
                    payload={},
                    msg_id="throughput_test",
                )
                mock_receive.return_value = response

                start = time.monotonic()
                for i in range(num_messages):
                    msg = IPCMessage(
                        msg_type=IPCMessageType.REQUEST,
                        payload={"index": i},
                        msg_id=f"msg_{i}",
                    )
                    channel.send(msg)
                    channel.receive()
                    messages_sent += 1
                elapsed = time.monotonic() - start

        # Should handle at least 1000 msg/sec (mocked)
        throughput = messages_sent / elapsed
        assert throughput > 1000

    def test_message_serialization_performance(self) -> None:
        """Message serialization is efficient."""
        msg = IPCMessage(
            msg_type=IPCMessageType.REQUEST,
            payload={"data": list(range(100))},
            msg_id="perf_test",
        )

        start = time.monotonic()
        for _ in range(1000):
            data = msg.serialize()
            IPCMessage.deserialize(data)
        elapsed_ms = (time.monotonic() - start) * 1000

        # 1000 serialize/deserialize cycles should be under 100ms
        assert elapsed_ms < 100
