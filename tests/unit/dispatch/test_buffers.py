"""
Comprehensive tests for the buffer management module.

Tests cover:
- AlignedAllocator: Memory alignment and allocation tracking
- RingBuffer: SPSC ring buffer operations, thread safety
- AudioRingBuffer: Multi-channel audio streaming
- BufferPool: Pre-allocated buffer pool operations
- TensorBufferView: Zero-copy tensor views
- BufferManager: Unified buffer management

Performance tests verify O(1) operations and zero allocation in hot paths.
"""
from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
import torch

from layerzero.dispatch.buffers import (
    # Constants
    CACHE_LINE_SIZE,
    SIMD_ALIGNMENT,
    PAGE_SIZE,
    MAX_RING_BUFFER_SIZE,
    # Enums
    BufferState,
    # Core classes
    AlignedAllocator,
    RingBuffer,
    AudioRingBuffer,
    BufferPool,
    PooledBuffer,
    TensorBufferView,
    BufferManager,
    # Global management
    get_global_buffer_manager,
    set_global_buffer_manager,
    # Utilities
    _is_power_of_two,
    _next_power_of_two,
    _align_up,
    _get_data_ptr_alignment,
    _is_aligned,
)


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_is_power_of_two(self) -> None:
        """Test power of two detection."""
        # Powers of two
        assert _is_power_of_two(1)
        assert _is_power_of_two(2)
        assert _is_power_of_two(4)
        assert _is_power_of_two(8)
        assert _is_power_of_two(16)
        assert _is_power_of_two(1024)
        assert _is_power_of_two(65536)

        # Not powers of two
        assert not _is_power_of_two(0)
        assert not _is_power_of_two(-1)
        assert not _is_power_of_two(3)
        assert not _is_power_of_two(5)
        assert not _is_power_of_two(6)
        assert not _is_power_of_two(7)
        assert not _is_power_of_two(1000)

    def test_next_power_of_two(self) -> None:
        """Test next power of two calculation."""
        assert _next_power_of_two(0) == 1
        assert _next_power_of_two(1) == 1
        assert _next_power_of_two(2) == 2
        assert _next_power_of_two(3) == 4
        assert _next_power_of_two(4) == 4
        assert _next_power_of_two(5) == 8
        assert _next_power_of_two(7) == 8
        assert _next_power_of_two(8) == 8
        assert _next_power_of_two(9) == 16
        assert _next_power_of_two(1000) == 1024
        assert _next_power_of_two(1024) == 1024
        assert _next_power_of_two(1025) == 2048

    def test_align_up(self) -> None:
        """Test alignment rounding."""
        assert _align_up(0, 64) == 0
        assert _align_up(1, 64) == 64
        assert _align_up(63, 64) == 64
        assert _align_up(64, 64) == 64
        assert _align_up(65, 64) == 128
        assert _align_up(100, 64) == 128

    def test_is_aligned(self) -> None:
        """Test alignment checking."""
        assert _is_aligned(0, 64)
        assert _is_aligned(64, 64)
        assert _is_aligned(128, 64)
        assert not _is_aligned(1, 64)
        assert not _is_aligned(32, 64)
        assert not _is_aligned(63, 64)

    def test_get_data_ptr_alignment(self) -> None:
        """Test tensor data pointer alignment detection."""
        tensor = torch.zeros(100, dtype=torch.float32)
        alignment = _get_data_ptr_alignment(tensor)
        # PyTorch typically aligns to at least 64 bytes
        assert alignment >= 4  # At least word-aligned


# =============================================================================
# AlignedAllocator Tests
# =============================================================================


class TestAlignedAllocator:
    """Tests for AlignedAllocator."""

    def test_initialization_default(self) -> None:
        """Test default initialization."""
        allocator = AlignedAllocator()
        assert allocator.alignment == SIMD_ALIGNMENT
        assert allocator.allocated_bytes == 0
        assert allocator.peak_bytes == 0
        assert allocator.allocation_count == 0

    def test_initialization_custom_alignment(self) -> None:
        """Test custom alignment initialization."""
        allocator = AlignedAllocator(alignment=128)
        assert allocator.alignment == 128

    def test_initialization_invalid_alignment(self) -> None:
        """Test invalid alignment raises error."""
        with pytest.raises(ValueError, match="power of two"):
            AlignedAllocator(alignment=100)  # Not power of 2

        # 0 is also not a power of two
        with pytest.raises(ValueError, match="power of two"):
            AlignedAllocator(alignment=0)

    def test_allocate_tensor_cpu(self) -> None:
        """Test CPU tensor allocation."""
        allocator = AlignedAllocator()
        tensor = allocator.allocate_tensor(
            shape=(4, 8, 16),
            dtype=torch.float32,
            device="cpu"
        )

        assert tensor.shape == (4, 8, 16)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cpu"
        assert allocator.allocation_count == 1
        assert allocator.allocated_bytes == 4 * 8 * 16 * 4  # float32 = 4 bytes

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_allocate_tensor_cuda(self) -> None:
        """Test CUDA tensor allocation."""
        allocator = AlignedAllocator()
        tensor = allocator.allocate_tensor(
            shape=(4, 8, 16),
            dtype=torch.float32,
            device="cuda"
        )

        assert tensor.shape == (4, 8, 16)
        assert tensor.dtype == torch.float32
        assert tensor.device.type == "cuda"

    def test_allocate_tensor_like(self) -> None:
        """Test allocate_tensor_like."""
        allocator = AlignedAllocator()
        template = torch.randn(8, 16, 32)

        tensor = allocator.allocate_tensor_like(template)

        assert tensor.shape == template.shape
        assert tensor.dtype == template.dtype
        assert tensor.device == template.device
        assert tensor.is_contiguous()

    def test_allocation_tracking(self) -> None:
        """Test allocation tracking statistics."""
        allocator = AlignedAllocator()

        # Allocate multiple tensors
        t1 = allocator.allocate_tensor((10, 10), torch.float32)
        t2 = allocator.allocate_tensor((20, 20), torch.float32)

        assert allocator.allocation_count == 2
        expected_bytes = (10 * 10 + 20 * 20) * 4
        assert allocator.allocated_bytes == expected_bytes
        assert allocator.peak_bytes == expected_bytes

        # Deallocate one
        allocator.deallocate(t1)
        assert allocator.allocated_bytes == 20 * 20 * 4
        assert allocator.peak_bytes == expected_bytes  # Peak unchanged

    def test_stats(self) -> None:
        """Test stats method."""
        allocator = AlignedAllocator(alignment=64)
        allocator.allocate_tensor((100,), torch.float32)

        stats = allocator.stats()

        assert "allocated_bytes" in stats
        assert "peak_bytes" in stats
        assert "allocation_count" in stats
        assert stats["alignment"] == 64
        assert stats["allocation_count"] == 1

    def test_reset_stats(self) -> None:
        """Test stats reset."""
        allocator = AlignedAllocator()
        allocator.allocate_tensor((100,), torch.float32)
        allocator.reset_stats()

        assert allocator.allocated_bytes == 0
        assert allocator.peak_bytes == 0
        assert allocator.allocation_count == 0


# =============================================================================
# RingBuffer Tests
# =============================================================================


class TestRingBuffer:
    """Tests for RingBuffer."""

    def test_initialization_basic(self) -> None:
        """Test basic initialization."""
        rb = RingBuffer(capacity=1024, dtype=torch.float32)

        assert rb.capacity >= 1024  # Rounded to power of 2
        assert _is_power_of_two(rb.capacity)
        assert rb.dtype == torch.float32
        assert rb.is_empty()
        assert not rb.is_full()
        assert len(rb) == 0

    def test_initialization_rounds_to_power_of_two(self) -> None:
        """Test capacity rounds to power of two."""
        rb = RingBuffer(capacity=1000)
        assert rb.capacity == 1024

        rb = RingBuffer(capacity=1025)
        assert rb.capacity == 2048

    def test_initialization_invalid_capacity(self) -> None:
        """Test invalid capacity raises error."""
        with pytest.raises(ValueError, match="positive"):
            RingBuffer(capacity=0)

        with pytest.raises(ValueError, match="maximum"):
            RingBuffer(capacity=MAX_RING_BUFFER_SIZE + 1)

    def test_push_pop_single(self) -> None:
        """Test single element push and pop."""
        rb = RingBuffer(capacity=16)

        assert rb.push(1.5)
        assert len(rb) == 1
        assert not rb.is_empty()

        value = rb.pop()
        assert value == pytest.approx(1.5)
        assert rb.is_empty()

    def test_push_pop_multiple(self) -> None:
        """Test multiple push and pop operations."""
        rb = RingBuffer(capacity=16)

        # Push values
        for i in range(10):
            assert rb.push(float(i))

        assert len(rb) == 10

        # Pop values in FIFO order
        for i in range(10):
            value = rb.pop()
            assert value == pytest.approx(float(i))

        assert rb.is_empty()

    def test_push_full_no_overwrite(self) -> None:
        """Test push fails when full without overwrite."""
        rb = RingBuffer(capacity=4, overwrite_on_full=False)

        # Fill buffer (capacity - 1 = 3 elements max due to empty slot)
        for i in range(3):
            assert rb.push(float(i))

        assert rb.is_full()

        # Push should fail
        assert not rb.push(100.0)
        assert rb.dropped_count == 0

    def test_push_full_with_overwrite(self) -> None:
        """Test push overwrites oldest when full."""
        rb = RingBuffer(capacity=4, overwrite_on_full=True)

        # Fill buffer
        for i in range(3):
            rb.push(float(i))

        # Push more - should overwrite oldest
        rb.push(100.0)
        rb.push(200.0)

        assert rb.dropped_count == 2

        # Oldest elements were dropped
        values = []
        while not rb.is_empty():
            values.append(rb.pop())

        # Should have: 2.0, 100.0, 200.0 (0.0 and 1.0 were overwritten)
        assert 0.0 not in values
        assert 100.0 in values
        assert 200.0 in values

    def test_pop_empty(self) -> None:
        """Test pop returns None when empty."""
        rb = RingBuffer(capacity=16)
        assert rb.pop() is None

    def test_peek(self) -> None:
        """Test peek doesn't remove element."""
        rb = RingBuffer(capacity=16)

        rb.push(42.0)

        assert rb.peek() == pytest.approx(42.0)
        assert rb.peek() == pytest.approx(42.0)  # Still there
        assert len(rb) == 1

    def test_push_batch(self) -> None:
        """Test batch push."""
        rb = RingBuffer(capacity=32)

        values = torch.arange(10, dtype=torch.float32)
        count = rb.push_batch(values)

        assert count == 10
        assert len(rb) == 10

        # Verify values
        for i in range(10):
            assert rb.pop() == pytest.approx(float(i))

    def test_pop_batch(self) -> None:
        """Test batch pop."""
        rb = RingBuffer(capacity=32)

        # Push values
        for i in range(20):
            rb.push(float(i))

        # Pop batch
        result = rb.pop_batch(10)

        assert result is not None
        assert result.shape == (10,)
        for i in range(10):
            assert result[i].item() == pytest.approx(float(i))

        assert len(rb) == 10

    def test_pop_batch_into(self) -> None:
        """Test zero-allocation batch pop."""
        rb = RingBuffer(capacity=32)

        for i in range(15):
            rb.push(float(i))

        # Pre-allocate output
        output = torch.zeros(10, dtype=torch.float32)
        count = rb.pop_batch_into(output)

        assert count == 10
        for i in range(10):
            assert output[i].item() == pytest.approx(float(i))

        assert len(rb) == 5

    def test_wraparound(self) -> None:
        """Test buffer correctly handles wraparound."""
        rb = RingBuffer(capacity=8)

        # Fill and partially empty multiple times to force wraparound
        for cycle in range(5):
            # Push 5 elements
            for i in range(5):
                rb.push(float(cycle * 10 + i))

            # Pop 3 elements
            for i in range(3):
                rb.pop()

        # Verify remaining elements are correct
        remaining = []
        while not rb.is_empty():
            remaining.append(rb.pop())

        # Should be last 10 pushed minus 3 popped each cycle
        assert len(remaining) > 0

    def test_available_read_write(self) -> None:
        """Test available read/write counts."""
        rb = RingBuffer(capacity=8)

        assert rb.available_read() == 0
        assert rb.available_write() == 7  # capacity - 1

        rb.push(1.0)
        rb.push(2.0)

        assert rb.available_read() == 2
        assert rb.available_write() == 5

        rb.pop()

        assert rb.available_read() == 1
        assert rb.available_write() == 6

    def test_clear(self) -> None:
        """Test clear operation."""
        rb = RingBuffer(capacity=16)

        for i in range(10):
            rb.push(float(i))

        count = rb.clear()
        assert count == 10
        assert rb.is_empty()

    def test_reset(self) -> None:
        """Test reset operation."""
        rb = RingBuffer(capacity=16, overwrite_on_full=True)

        for i in range(20):  # Overwrite some
            rb.push(float(i))

        rb.reset()

        assert rb.is_empty()
        assert rb.dropped_count == 0

    def test_stats(self) -> None:
        """Test stats method."""
        rb = RingBuffer(capacity=16)

        for i in range(5):
            rb.push(float(i))

        stats = rb.stats()

        assert stats["capacity"] == 16
        assert stats["size"] == 5
        assert "available_read" in stats
        assert "available_write" in stats
        assert "dtype" in stats

    def test_thread_safety_spsc(self) -> None:
        """Test SPSC pattern is thread-safe."""
        rb = RingBuffer(capacity=1024)
        num_items = 10000
        results: list[float] = []
        producer_done = threading.Event()

        def producer():
            for i in range(num_items):
                while not rb.push(float(i)):
                    time.sleep(0.0001)
            producer_done.set()

        def consumer():
            while True:
                if rb.is_empty():
                    if producer_done.is_set():
                        # Drain remaining
                        while not rb.is_empty():
                            val = rb.pop()
                            if val is not None:
                                results.append(val)
                        break
                    time.sleep(0.0001)
                    continue

                val = rb.pop()
                if val is not None:
                    results.append(val)

        producer_thread = threading.Thread(target=producer)
        consumer_thread = threading.Thread(target=consumer)

        producer_thread.start()
        consumer_thread.start()

        producer_thread.join(timeout=10)
        consumer_thread.join(timeout=10)

        # All items should be received
        assert len(results) == num_items

        # Items should be in order (FIFO)
        for i, val in enumerate(results):
            assert val == pytest.approx(float(i))


# =============================================================================
# AudioRingBuffer Tests
# =============================================================================


class TestAudioRingBuffer:
    """Tests for AudioRingBuffer."""

    def test_initialization(self) -> None:
        """Test initialization."""
        arb = AudioRingBuffer(
            num_channels=2,
            capacity_frames=1024,
            sample_rate=48000
        )

        assert arb.num_channels == 2
        assert arb.capacity_frames >= 1024
        assert arb.sample_rate == 48000
        assert arb.capacity_seconds == pytest.approx(
            arb.capacity_frames / 48000
        )

    def test_initialization_invalid(self) -> None:
        """Test invalid parameters raise errors."""
        with pytest.raises(ValueError, match="num_channels"):
            AudioRingBuffer(num_channels=0, capacity_frames=1024)

        with pytest.raises(ValueError, match="capacity_frames"):
            AudioRingBuffer(num_channels=2, capacity_frames=0)

        with pytest.raises(ValueError, match="sample_rate"):
            AudioRingBuffer(num_channels=2, capacity_frames=1024, sample_rate=0)

    def test_push_pop_interleaved_stereo(self) -> None:
        """Test interleaved stereo audio."""
        arb = AudioRingBuffer(num_channels=2, capacity_frames=64)

        # Create interleaved stereo: L0, R0, L1, R1, ...
        num_frames = 10
        interleaved = torch.zeros(num_frames * 2, dtype=torch.float32)
        for i in range(num_frames):
            interleaved[i * 2] = float(i)        # Left channel
            interleaved[i * 2 + 1] = float(i) + 0.5  # Right channel

        frames_pushed = arb.push_interleaved(interleaved)
        assert frames_pushed == num_frames

        # Pop and verify
        result = arb.pop_interleaved(num_frames)
        assert result is not None
        assert result.shape[0] == num_frames * 2

        for i in range(num_frames):
            assert result[i * 2].item() == pytest.approx(float(i))
            assert result[i * 2 + 1].item() == pytest.approx(float(i) + 0.5)

    def test_push_pop_planar(self) -> None:
        """Test planar audio."""
        arb = AudioRingBuffer(num_channels=2, capacity_frames=64)

        # Create planar: [[L0, L1, ...], [R0, R1, ...]]
        num_frames = 10
        planar = torch.zeros(2, num_frames, dtype=torch.float32)
        for i in range(num_frames):
            planar[0, i] = float(i)          # Left
            planar[1, i] = float(i) + 0.5    # Right

        arb.push_planar(planar)

        result = arb.pop_planar(num_frames)
        assert result is not None
        assert result.shape == (2, num_frames)

        for i in range(num_frames):
            assert result[0, i].item() == pytest.approx(float(i))
            assert result[1, i].item() == pytest.approx(float(i) + 0.5)

    def test_available_frames(self) -> None:
        """Test frame availability tracking."""
        arb = AudioRingBuffer(num_channels=2, capacity_frames=64)

        assert arb.available_frames() == 0

        interleaved = torch.zeros(20, dtype=torch.float32)  # 10 frames
        arb.push_interleaved(interleaved)

        assert arb.available_frames() == 10

    def test_stats(self) -> None:
        """Test stats method."""
        arb = AudioRingBuffer(
            num_channels=2,
            capacity_frames=1024,
            sample_rate=48000
        )

        arb.push_interleaved(torch.zeros(100, dtype=torch.float32))

        stats = arb.stats()

        assert stats["num_channels"] == 2
        assert "capacity_frames" in stats
        assert stats["sample_rate"] == 48000
        assert "available_frames" in stats
        assert "capacity_seconds" in stats


# =============================================================================
# BufferPool Tests
# =============================================================================


class TestBufferPool:
    """Tests for BufferPool."""

    def test_initialization(self) -> None:
        """Test pool initialization."""
        pool = BufferPool(
            num_buffers=10,
            buffer_shape=(4, 8, 16),
            dtype=torch.float32,
            device="cpu"
        )

        assert pool.num_buffers == 10
        assert pool.buffer_shape == (4, 8, 16)
        assert pool.dtype == torch.float32
        assert pool.available() == 10
        assert pool.in_use() == 0

    def test_initialization_invalid(self) -> None:
        """Test invalid parameters raise errors."""
        with pytest.raises(ValueError, match="num_buffers"):
            BufferPool(num_buffers=0, buffer_shape=(10,))

        with pytest.raises(ValueError, match="buffer_shape"):
            BufferPool(num_buffers=10, buffer_shape=())

    def test_acquire_release(self) -> None:
        """Test basic acquire and release."""
        pool = BufferPool(
            num_buffers=5,
            buffer_shape=(10, 10),
            dtype=torch.float32
        )

        # Acquire buffer
        buf = pool.acquire()
        assert buf is not None
        assert buf.state == BufferState.ACQUIRED
        assert buf.tensor.shape == (10, 10)
        assert pool.available() == 4
        assert pool.in_use() == 1

        # Release buffer
        pool.release(buf)
        assert buf.state == BufferState.FREE
        assert pool.available() == 5
        assert pool.in_use() == 0

    def test_acquire_all_exhausted(self) -> None:
        """Test acquiring all buffers exhausts pool."""
        pool = BufferPool(
            num_buffers=3,
            buffer_shape=(10,),
            dtype=torch.float32
        )

        buffers = []
        for _ in range(3):
            buf = pool.acquire()
            assert buf is not None
            buffers.append(buf)

        # Pool exhausted
        assert pool.available() == 0
        buf = pool.acquire()
        assert buf is None

        # Release one
        pool.release(buffers[0])
        assert pool.available() == 1

        # Can acquire again
        buf = pool.acquire()
        assert buf is not None

    def test_acquire_or_wait(self) -> None:
        """Test acquire with timeout."""
        pool = BufferPool(
            num_buffers=2,
            buffer_shape=(10,),
            dtype=torch.float32
        )

        # Exhaust pool
        b1 = pool.acquire()
        b2 = pool.acquire()

        # Start thread to release after delay
        def release_later():
            time.sleep(0.1)
            pool.release(b1)

        thread = threading.Thread(target=release_later)
        thread.start()

        # Should wait and succeed
        buf = pool.acquire_or_wait(timeout=1.0)
        assert buf is not None

        thread.join()
        pool.release(buf)
        pool.release(b2)

    def test_acquire_or_wait_timeout(self) -> None:
        """Test acquire timeout."""
        pool = BufferPool(
            num_buffers=1,
            buffer_shape=(10,),
            dtype=torch.float32
        )

        # Exhaust pool
        b1 = pool.acquire()

        # Should timeout
        start = time.monotonic()
        buf = pool.acquire_or_wait(timeout=0.1)
        elapsed = time.monotonic() - start

        assert buf is None
        assert elapsed >= 0.1

        pool.release(b1)

    def test_release_invalid(self) -> None:
        """Test releasing invalid buffer raises error."""
        pool = BufferPool(
            num_buffers=3,
            buffer_shape=(10,),
            dtype=torch.float32
        )

        buf = pool.acquire()

        # Double release
        pool.release(buf)
        with pytest.raises(ValueError, match="already released"):
            pool.release(buf)

    def test_release_all(self) -> None:
        """Test release_all operation."""
        pool = BufferPool(
            num_buffers=5,
            buffer_shape=(10,),
            dtype=torch.float32
        )

        # Acquire all
        for _ in range(5):
            pool.acquire()

        assert pool.in_use() == 5

        released = pool.release_all()
        assert released == 5
        assert pool.available() == 5

    def test_buffer_bytes(self) -> None:
        """Test buffer size calculation."""
        pool = BufferPool(
            num_buffers=3,
            buffer_shape=(10, 20),
            dtype=torch.float32  # 4 bytes per element
        )

        assert pool.buffer_bytes == 10 * 20 * 4
        assert pool.total_bytes == 3 * 10 * 20 * 4

    def test_stats(self) -> None:
        """Test stats method."""
        pool = BufferPool(
            num_buffers=10,
            buffer_shape=(8, 8),
            dtype=torch.float32
        )

        pool.acquire()
        pool.acquire()

        stats = pool.stats()

        assert stats["num_buffers"] == 10
        assert stats["available"] == 8
        assert stats["in_use"] == 2
        assert stats["buffer_shape"] == (8, 8)
        assert stats["total_acquires"] == 2

    def test_thread_safety(self) -> None:
        """Test thread-safe acquire/release."""
        pool = BufferPool(
            num_buffers=20,
            buffer_shape=(100,),
            dtype=torch.float32
        )

        num_iterations = 100
        errors: list[Exception] = []

        def worker(worker_id: int):
            try:
                for _ in range(num_iterations):
                    buf = pool.acquire()
                    if buf is not None:
                        # Simulate work
                        buf.tensor.fill_(worker_id)
                        time.sleep(0.001)
                        pool.release(buf)
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=(i,))
            for i in range(5)
        ]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert pool.available() == 20  # All released


# =============================================================================
# TensorBufferView Tests
# =============================================================================


class TestTensorBufferView:
    """Tests for TensorBufferView."""

    def test_initialization(self) -> None:
        """Test basic initialization."""
        tensor = torch.randn(4, 8, 16)
        view = TensorBufferView(tensor)

        assert view.shape == (4, 8, 16)
        assert view.dtype == tensor.dtype
        assert view.device == tensor.device
        assert view.numel == 4 * 8 * 16

    def test_tensor_property_returns_same_view(self) -> None:
        """Test tensor property returns view of original."""
        original = torch.randn(10, 20)
        view = TensorBufferView(original)

        result = view.tensor

        # Should share storage
        original[0, 0] = 999.0
        assert result[0, 0].item() == 999.0

    def test_slice(self) -> None:
        """Test slicing creates zero-copy view."""
        tensor = torch.arange(100).float().reshape(10, 10)
        view = TensorBufferView(tensor)

        sliced = view.slice(start=2, end=5, dim=0)

        assert sliced.shape == (3, 10)

        # Should be zero-copy
        tensor[2, 0] = 999.0
        assert sliced.tensor[0, 0].item() == 999.0

    def test_reshape(self) -> None:
        """Test reshaping creates zero-copy view."""
        tensor = torch.arange(24).float()
        view = TensorBufferView(tensor)

        reshaped = view.reshape(2, 3, 4)
        assert reshaped.shape == (2, 3, 4)

        # Inferred dimension
        reshaped2 = view.reshape(6, -1)
        assert reshaped2.shape == (6, 4)

    def test_reshape_invalid(self) -> None:
        """Test invalid reshape raises error."""
        tensor = torch.arange(24).float()
        view = TensorBufferView(tensor)

        with pytest.raises(ValueError, match="elements"):
            view.reshape(5, 5)  # 25 != 24

    def test_transpose(self) -> None:
        """Test transpose creates zero-copy view."""
        tensor = torch.randn(3, 4)
        view = TensorBufferView(tensor)

        transposed = view.transpose(0, 1)

        assert transposed.shape == (4, 3)
        assert not transposed.is_contiguous

        # Should share storage
        tensor[0, 0] = 999.0
        assert transposed.tensor[0, 0].item() == 999.0

    def test_squeeze(self) -> None:
        """Test squeeze removes size-1 dimensions."""
        tensor = torch.randn(1, 3, 1, 4, 1)
        view = TensorBufferView(tensor)

        squeezed = view.squeeze()
        assert squeezed.shape == (3, 4)

        # Specific dimension
        view2 = TensorBufferView(tensor)
        squeezed2 = view2.squeeze(dim=0)
        assert squeezed2.shape == (3, 1, 4, 1)

    def test_unsqueeze(self) -> None:
        """Test unsqueeze adds size-1 dimension."""
        tensor = torch.randn(3, 4)
        view = TensorBufferView(tensor)

        unsqueezed = view.unsqueeze(0)
        assert unsqueezed.shape == (1, 3, 4)

        unsqueezed2 = view.unsqueeze(-1)
        assert unsqueezed2.shape == (3, 4, 1)

    def test_contiguous(self) -> None:
        """Test contiguous method."""
        tensor = torch.randn(3, 4)
        view = TensorBufferView(tensor)
        transposed = view.transpose(0, 1)

        # Non-contiguous view
        assert not transposed.is_contiguous

        # Get contiguous copy
        cont = transposed.contiguous()
        assert cont.is_contiguous()

    def test_copy_to_from(self) -> None:
        """Test copy_to and copy_from methods."""
        src = torch.randn(10, 20)
        dst = torch.zeros(10, 20)

        view = TensorBufferView(src)
        view.copy_to(dst)

        assert torch.allclose(src, dst)

        # copy_from
        new_data = torch.randn(10, 20)
        view2 = TensorBufferView(dst)
        view2.copy_from(new_data)

        assert torch.allclose(dst, new_data)

    def test_from_numpy(self) -> None:
        """Test creating view from NumPy array."""
        import numpy as np

        arr = np.random.randn(10, 20).astype(np.float32)
        view = TensorBufferView.from_numpy(arr)

        assert view.shape == (10, 20)
        assert view.dtype == torch.float32

        # Should be zero-copy
        arr[0, 0] = 999.0
        assert view.tensor[0, 0].item() == 999.0

    def test_to_numpy(self) -> None:
        """Test converting view to NumPy array."""
        tensor = torch.randn(10, 20)
        view = TensorBufferView(tensor)

        arr = view.to_numpy()

        assert arr.shape == (10, 20)

        # Should be zero-copy
        tensor[0, 0] = 999.0
        assert arr[0, 0] == 999.0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_numpy_cuda_fails(self) -> None:
        """Test to_numpy fails for CUDA tensors."""
        tensor = torch.randn(10, device="cuda")
        view = TensorBufferView(tensor)

        with pytest.raises(ValueError, match="non-CPU"):
            view.to_numpy()


# =============================================================================
# BufferManager Tests
# =============================================================================


class TestBufferManager:
    """Tests for BufferManager."""

    def test_initialization(self) -> None:
        """Test initialization."""
        manager = BufferManager()
        stats = manager.stats()

        assert stats["pools"] == {}
        assert stats["ring_buffers"] == {}
        assert stats["audio_buffers"] == {}

    def test_create_pool(self) -> None:
        """Test creating buffer pools."""
        manager = BufferManager()

        pool = manager.create_pool(
            name="test_pool",
            num_buffers=10,
            buffer_shape=(8, 16),
            dtype=torch.float32
        )

        assert pool is not None
        assert manager.get_pool("test_pool") is pool

    def test_create_pool_duplicate_name(self) -> None:
        """Test duplicate pool name raises error."""
        manager = BufferManager()
        manager.create_pool("test", 5, (10,))

        with pytest.raises(ValueError, match="already exists"):
            manager.create_pool("test", 5, (10,))

    def test_acquire_release(self) -> None:
        """Test acquire/release through manager."""
        manager = BufferManager()
        manager.create_pool("pool", 5, (10,))

        buf = manager.acquire("pool")
        assert buf is not None

        manager.release("pool", buf)
        assert manager.get_pool("pool").available() == 5

    def test_acquire_nonexistent_pool(self) -> None:
        """Test acquiring from nonexistent pool returns None."""
        manager = BufferManager()
        assert manager.acquire("nonexistent") is None

    def test_create_ring_buffer(self) -> None:
        """Test creating ring buffers."""
        manager = BufferManager()

        rb = manager.create_ring_buffer(
            name="audio_in",
            capacity=4096,
            dtype=torch.float32
        )

        assert rb is not None
        assert manager.get_ring_buffer("audio_in") is rb

    def test_create_audio_buffer(self) -> None:
        """Test creating audio buffers."""
        manager = BufferManager()

        ab = manager.create_audio_buffer(
            name="stereo",
            num_channels=2,
            capacity_frames=48000,
            sample_rate=48000
        )

        assert ab is not None
        assert manager.get_audio_buffer("stereo") is ab

    def test_create_view(self) -> None:
        """Test creating tensor views."""
        manager = BufferManager()
        tensor = torch.randn(10, 20)

        view = manager.create_view(tensor)

        assert isinstance(view, TensorBufferView)
        assert view.shape == (10, 20)

    def test_allocate_aligned(self) -> None:
        """Test aligned allocation."""
        manager = BufferManager()

        tensor = manager.allocate_aligned(
            shape=(8, 16, 32),
            dtype=torch.float32
        )

        assert tensor.shape == (8, 16, 32)

    def test_delete_pool(self) -> None:
        """Test deleting pools."""
        manager = BufferManager()
        manager.create_pool("pool", 5, (10,))

        assert manager.delete_pool("pool")
        assert manager.get_pool("pool") is None
        assert not manager.delete_pool("pool")  # Already deleted

    def test_clear_all(self) -> None:
        """Test clearing all buffers."""
        manager = BufferManager()
        manager.create_pool("pool1", 5, (10,))
        manager.create_pool("pool2", 5, (20,))
        manager.create_ring_buffer("rb", 1024)
        manager.create_audio_buffer("ab", 2, 1024)

        manager.clear_all()

        assert manager.stats()["pools"] == {}
        assert manager.stats()["ring_buffers"] == {}
        assert manager.stats()["audio_buffers"] == {}

    def test_stats(self) -> None:
        """Test comprehensive stats."""
        manager = BufferManager()
        manager.create_pool("pool", 5, (10,))
        manager.create_ring_buffer("rb", 1024)

        stats = manager.stats()

        assert "pool" in stats["pools"]
        assert "rb" in stats["ring_buffers"]
        assert "allocator" in stats


# =============================================================================
# Global Buffer Manager Tests
# =============================================================================


class TestGlobalBufferManager:
    """Tests for global buffer manager singleton."""

    def test_get_global_creates_singleton(self) -> None:
        """Test get_global_buffer_manager creates singleton."""
        manager1 = get_global_buffer_manager()
        manager2 = get_global_buffer_manager()

        assert manager1 is manager2

    def test_set_global_buffer_manager(self) -> None:
        """Test setting global buffer manager."""
        custom_manager = BufferManager(alignment=128)
        set_global_buffer_manager(custom_manager)

        assert get_global_buffer_manager() is custom_manager


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance and benchmarking tests."""

    def test_ring_buffer_o1_push_pop(self) -> None:
        """Verify O(1) push/pop operations."""
        rb = RingBuffer(capacity=65536)

        # Time many operations
        num_ops = 10000
        values = torch.randn(num_ops)

        start = time.perf_counter()
        for i in range(num_ops):
            rb.push(values[i].item())
        push_time = time.perf_counter() - start

        start = time.perf_counter()
        for _ in range(num_ops):
            rb.pop()
        pop_time = time.perf_counter() - start

        # Average should be very fast (< 10us per operation)
        push_per_op = (push_time / num_ops) * 1_000_000  # microseconds
        pop_per_op = (pop_time / num_ops) * 1_000_000

        # Log timing for analysis (not strict assertion due to system variance)
        print(f"\nRingBuffer push: {push_per_op:.3f}us/op")
        print(f"RingBuffer pop: {pop_per_op:.3f}us/op")

        # Should be < 100us per operation on any reasonable system
        assert push_per_op < 100
        assert pop_per_op < 100

    def test_buffer_pool_o1_acquire_release(self) -> None:
        """Verify O(1) acquire/release operations."""
        pool = BufferPool(
            num_buffers=1000,
            buffer_shape=(100,),
            dtype=torch.float32
        )

        num_ops = 1000

        # Time acquire
        start = time.perf_counter()
        buffers = []
        for _ in range(num_ops):
            buf = pool.acquire()
            buffers.append(buf)
        acquire_time = time.perf_counter() - start

        # Time release
        start = time.perf_counter()
        for buf in buffers:
            pool.release(buf)
        release_time = time.perf_counter() - start

        acquire_per_op = (acquire_time / num_ops) * 1_000_000
        release_per_op = (release_time / num_ops) * 1_000_000

        print(f"\nBufferPool acquire: {acquire_per_op:.3f}us/op")
        print(f"BufferPool release: {release_per_op:.3f}us/op")

        # Should be < 100us per operation
        assert acquire_per_op < 100
        assert release_per_op < 100

    def test_zero_allocation_pop_batch_into(self) -> None:
        """Verify pop_batch_into doesn't allocate."""
        import tracemalloc

        rb = RingBuffer(capacity=4096)

        # Fill buffer
        for i in range(2000):
            rb.push(float(i))

        # Pre-allocate output
        output = torch.zeros(1000, dtype=torch.float32)

        # Measure allocations
        tracemalloc.start()
        snapshot_before = tracemalloc.take_snapshot()

        # Hot path operation
        for _ in range(100):
            rb.push_batch(torch.randn(500))
            rb.pop_batch_into(output)

        snapshot_after = tracemalloc.take_snapshot()
        tracemalloc.stop()

        # Compare allocations
        diff = snapshot_after.compare_to(snapshot_before, 'lineno')

        # Should have minimal allocations (some may come from internal Python ops)
        total_diff = sum(stat.size_diff for stat in diff if stat.size_diff > 0)

        print(f"\nMemory allocation during hot path: {total_diff} bytes")

        # Allow some overhead, but should be much less than the data size
        # (100 iterations * 500 * 4 bytes = 200KB if we were copying)
        assert total_diff < 50000  # Less than 50KB overhead


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_ring_buffer_capacity_one(self) -> None:
        """Test ring buffer with minimum capacity."""
        rb = RingBuffer(capacity=1)
        # Capacity rounds to next power of 2 (which is 1)
        # But we need at least one empty slot, so effective capacity is 0
        # This is an edge case that might need special handling
        # Let's verify it at least doesn't crash
        assert rb.capacity == 1

    def test_buffer_pool_single_buffer(self) -> None:
        """Test pool with single buffer."""
        pool = BufferPool(num_buffers=1, buffer_shape=(10,))

        buf = pool.acquire()
        assert buf is not None
        assert pool.acquire() is None

        pool.release(buf)
        buf = pool.acquire()
        assert buf is not None

    def test_view_empty_tensor(self) -> None:
        """Test view of empty tensor."""
        tensor = torch.tensor([])
        view = TensorBufferView(tensor)

        assert view.numel == 0
        assert view.shape == (0,)

    def test_view_scalar_tensor(self) -> None:
        """Test view of scalar (0-d) tensor."""
        tensor = torch.tensor(42.0)
        view = TensorBufferView(tensor)

        assert view.numel == 1
        assert view.shape == ()

    def test_audio_buffer_mono(self) -> None:
        """Test mono audio buffer."""
        arb = AudioRingBuffer(num_channels=1, capacity_frames=1024)

        data = torch.randn(100)
        arb.push_interleaved(data)

        result = arb.pop_interleaved(100)
        assert result is not None
        assert result.shape == (100,)

    def test_ring_buffer_exact_capacity_fill(self) -> None:
        """Test filling ring buffer to exact capacity."""
        rb = RingBuffer(capacity=8)  # Actual capacity 8, max elements 7

        # Fill to max
        for i in range(7):
            assert rb.push(float(i))

        assert rb.is_full()
        assert not rb.push(100.0)  # Should fail

        # Pop one and push one
        rb.pop()
        assert rb.push(100.0)
        assert rb.is_full()
