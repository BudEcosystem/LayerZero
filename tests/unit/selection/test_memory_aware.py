"""Tests for memory-aware kernel selection."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

import torch

from layerzero.selection.memory_aware import (
    MemoryConfig,
    MemoryEstimator,
    MemoryFilter,
    MemoryRequirement,
)
from layerzero.reasons import MEMORY_HEADROOM_EXCEEDED, Reason, ReasonCategory


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = MemoryConfig()

        assert config.headroom_bytes is None  # Dynamic by default
        assert config.headroom_fraction == 0.1
        assert config.min_headroom_mb == 256
        assert config.include_workspace is True
        assert config.include_temp_buffers is True

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = MemoryConfig(
            headroom_bytes=1024 * 1024 * 1024,  # 1GB
            headroom_fraction=0.2,
            min_headroom_mb=512,
            include_workspace=False,
            include_temp_buffers=False,
        )

        assert config.headroom_bytes == 1024 * 1024 * 1024
        assert config.headroom_fraction == 0.2
        assert config.min_headroom_mb == 512
        assert config.include_workspace is False
        assert config.include_temp_buffers is False

    def test_config_immutable(self) -> None:
        """Config is immutable."""
        config = MemoryConfig()

        with pytest.raises(AttributeError):
            config.headroom_bytes = 1024


class TestMemoryRequirement:
    """Tests for MemoryRequirement dataclass."""

    def test_creation(self) -> None:
        """MemoryRequirement stores values correctly."""
        req = MemoryRequirement(
            workspace_bytes=1024 * 1024,
            temp_buffer_bytes=512 * 1024,
            output_bytes=256 * 1024,
        )

        assert req.workspace_bytes == 1024 * 1024
        assert req.temp_buffer_bytes == 512 * 1024
        assert req.output_bytes == 256 * 1024

    def test_total_bytes(self) -> None:
        """total_bytes sums all components."""
        req = MemoryRequirement(
            workspace_bytes=100,
            temp_buffer_bytes=200,
            output_bytes=300,
        )

        assert req.total_bytes == 600

    def test_total_mb(self) -> None:
        """total_mb converts to megabytes."""
        req = MemoryRequirement(
            workspace_bytes=512 * 1024,
            temp_buffer_bytes=512 * 1024,
            output_bytes=0,
        )

        assert req.total_mb == 1.0

    def test_zero_requirement(self) -> None:
        """Zero requirement has zero total."""
        req = MemoryRequirement()

        assert req.total_bytes == 0
        assert req.total_mb == 0.0


class TestMemoryEstimator:
    """Tests for MemoryEstimator."""

    def test_workspace_bytes_estimation(self) -> None:
        """Workspace bytes estimated per kernel."""
        estimator = MemoryEstimator()

        kernel = MagicMock()
        kernel.operation = "attention"
        kernel.workspace_bytes = 1024 * 1024  # 1MB

        ctx = MagicMock()
        ctx.batch_size = 4
        ctx.seq_len = 2048
        ctx.num_heads = 32
        ctx.head_dim = 128

        workspace = estimator.estimate_workspace(kernel, ctx)

        assert workspace > 0

    def test_temp_allocation_estimation(self) -> None:
        """Temporary allocation estimated."""
        estimator = MemoryEstimator()

        kernel = MagicMock()
        kernel.operation = "attention"
        kernel.temp_buffer_ratio = 0.5

        ctx = MagicMock()
        ctx.batch_size = 4
        ctx.seq_len = 2048
        ctx.num_heads = 32
        ctx.head_dim = 128
        ctx.dtype = torch.float16

        temp = estimator.estimate_temp_buffers(kernel, ctx)

        assert temp >= 0

    def test_total_memory_requirement(self) -> None:
        """Total memory requirement calculated."""
        estimator = MemoryEstimator()

        kernel = MagicMock()
        kernel.operation = "attention"
        kernel.workspace_bytes = 1024 * 1024
        kernel.temp_buffer_ratio = 0.0

        ctx = MagicMock()
        ctx.batch_size = 4
        ctx.seq_len = 2048
        ctx.num_heads = 32
        ctx.head_dim = 128
        ctx.dtype = torch.float16

        req = estimator.estimate(kernel, ctx)

        assert isinstance(req, MemoryRequirement)
        assert req.total_bytes > 0

    def test_estimate_output_size(self) -> None:
        """Output size estimated correctly."""
        estimator = MemoryEstimator()

        kernel = MagicMock()
        kernel.operation = "attention"

        ctx = MagicMock()
        ctx.batch_size = 4
        ctx.seq_len = 2048
        ctx.num_heads = 32
        ctx.head_dim = 128
        ctx.dtype = torch.float16  # 2 bytes

        output = estimator.estimate_output(kernel, ctx)

        # Output: batch * seq * heads * head_dim * dtype_size
        expected = 4 * 2048 * 32 * 128 * 2
        assert output == expected


class TestMemoryFilter:
    """Tests for MemoryFilter."""

    def test_kernel_accepted_within_headroom(self) -> None:
        """Kernel accepted if within headroom."""
        config = MemoryConfig(headroom_bytes=1024 * 1024 * 1024)  # 1GB
        memory_filter = MemoryFilter(config=config)

        kernel = MagicMock()
        kernel.kernel_id = "test_kernel"
        kernel.operation = "attention"
        kernel.workspace_bytes = 1024 * 1024  # 1MB
        kernel.temp_buffer_ratio = 0.0

        ctx = MagicMock()
        ctx.batch_size = 1
        ctx.seq_len = 512
        ctx.num_heads = 8
        ctx.head_dim = 64
        ctx.dtype = torch.float16

        # Mock available memory
        with patch.object(memory_filter, '_get_available_memory', return_value=2 * 1024 * 1024 * 1024):  # 2GB
            passed, reason = memory_filter.check(kernel, ctx)

        assert passed is True
        assert reason is None

    def test_kernel_rejected_exceeds_headroom(self) -> None:
        """Kernel rejected if exceeds headroom."""
        config = MemoryConfig(headroom_bytes=10 * 1024)  # Only 10KB headroom
        memory_filter = MemoryFilter(config=config)

        kernel = MagicMock()
        kernel.kernel_id = "test_kernel"
        kernel.operation = "attention"
        kernel.workspace_bytes = 100 * 1024 * 1024  # 100MB workspace
        kernel.temp_buffer_ratio = 0.0

        ctx = MagicMock()
        ctx.batch_size = 16
        ctx.seq_len = 4096
        ctx.num_heads = 32
        ctx.head_dim = 128
        ctx.dtype = torch.float16

        # Mock very low available memory
        with patch.object(memory_filter, '_get_available_memory', return_value=10 * 1024):  # 10KB
            passed, reason = memory_filter.check(kernel, ctx)

        assert passed is False
        assert reason is not None
        assert reason.code == MEMORY_HEADROOM_EXCEEDED

    def test_reason_code_memory_exceeded(self) -> None:
        """MEMORY_HEADROOM_EXCEEDED reason code returned."""
        config = MemoryConfig(headroom_bytes=0)  # No headroom
        memory_filter = MemoryFilter(config=config)

        kernel = MagicMock()
        kernel.kernel_id = "test_kernel"
        kernel.operation = "attention"
        kernel.workspace_bytes = 1024 * 1024  # 1MB
        kernel.temp_buffer_ratio = 0.0

        ctx = MagicMock()
        ctx.batch_size = 1
        ctx.seq_len = 512
        ctx.num_heads = 8
        ctx.head_dim = 64
        ctx.dtype = torch.float16

        # Mock zero available memory
        with patch.object(memory_filter, '_get_available_memory', return_value=0):
            passed, reason = memory_filter.check(kernel, ctx)

        assert passed is False
        assert reason.code == MEMORY_HEADROOM_EXCEEDED
        assert reason.category == ReasonCategory.MEMORY


class TestDynamicHeadroom:
    """Tests for dynamic headroom calculation."""

    def test_headroom_from_config(self) -> None:
        """Headroom from explicit configuration."""
        config = MemoryConfig(headroom_bytes=512 * 1024 * 1024)  # 512MB
        memory_filter = MemoryFilter(config=config)

        headroom = memory_filter.get_effective_headroom(total_memory=4 * 1024 * 1024 * 1024)

        assert headroom == 512 * 1024 * 1024

    def test_headroom_from_fraction(self) -> None:
        """Headroom calculated as fraction of available."""
        config = MemoryConfig(
            headroom_bytes=None,
            headroom_fraction=0.1,  # 10%
        )
        memory_filter = MemoryFilter(config=config)

        total = 4 * 1024 * 1024 * 1024  # 4GB
        headroom = memory_filter.get_effective_headroom(total_memory=total)

        # 10% of 4GB = 400MB (but min is 256MB)
        assert headroom == int(total * 0.1)

    def test_headroom_respects_minimum(self) -> None:
        """Headroom respects minimum threshold."""
        config = MemoryConfig(
            headroom_bytes=None,
            headroom_fraction=0.01,  # 1%
            min_headroom_mb=256,
        )
        memory_filter = MemoryFilter(config=config)

        total = 1 * 1024 * 1024 * 1024  # 1GB
        # 1% of 1GB = 10MB, but minimum is 256MB
        headroom = memory_filter.get_effective_headroom(total_memory=total)

        assert headroom == 256 * 1024 * 1024

    def test_headroom_updated_runtime(self) -> None:
        """Headroom can be updated at runtime."""
        config = MemoryConfig(headroom_bytes=100 * 1024 * 1024)
        memory_filter = MemoryFilter(config=config)

        # Update headroom
        new_config = MemoryConfig(headroom_bytes=200 * 1024 * 1024)
        memory_filter.update_config(new_config)

        assert memory_filter.config.headroom_bytes == 200 * 1024 * 1024


class TestMemoryFilterIntegration:
    """Integration tests for memory filter."""

    def test_filter_multiple_kernels(self) -> None:
        """Filter multiple kernels by memory."""
        config = MemoryConfig(headroom_bytes=100 * 1024 * 1024)  # 100MB
        memory_filter = MemoryFilter(config=config)

        # Create kernels with different memory requirements
        small_kernel = MagicMock()
        small_kernel.kernel_id = "small"
        small_kernel.operation = "attention"
        small_kernel.workspace_bytes = 1 * 1024 * 1024  # 1MB
        small_kernel.temp_buffer_ratio = 0.0

        large_kernel = MagicMock()
        large_kernel.kernel_id = "large"
        large_kernel.operation = "attention"
        large_kernel.workspace_bytes = 200 * 1024 * 1024  # 200MB - too big
        large_kernel.temp_buffer_ratio = 0.0

        ctx = MagicMock()
        ctx.batch_size = 1
        ctx.seq_len = 512
        ctx.num_heads = 8
        ctx.head_dim = 64
        ctx.dtype = torch.float16

        with patch.object(memory_filter, '_get_available_memory', return_value=150 * 1024 * 1024):
            small_passed, _ = memory_filter.check(small_kernel, ctx)
            large_passed, _ = memory_filter.check(large_kernel, ctx)

        assert small_passed is True
        assert large_passed is False

    def test_filter_respects_dtype_size(self) -> None:
        """Memory estimation respects dtype size."""
        estimator = MemoryEstimator()

        kernel = MagicMock()
        kernel.operation = "attention"

        ctx_fp16 = MagicMock()
        ctx_fp16.batch_size = 1
        ctx_fp16.seq_len = 1024
        ctx_fp16.num_heads = 8
        ctx_fp16.head_dim = 64
        ctx_fp16.dtype = torch.float16  # 2 bytes

        ctx_fp32 = MagicMock()
        ctx_fp32.batch_size = 1
        ctx_fp32.seq_len = 1024
        ctx_fp32.num_heads = 8
        ctx_fp32.head_dim = 64
        ctx_fp32.dtype = torch.float32  # 4 bytes

        output_fp16 = estimator.estimate_output(kernel, ctx_fp16)
        output_fp32 = estimator.estimate_output(kernel, ctx_fp32)

        # FP32 should be 2x FP16
        assert output_fp32 == 2 * output_fp16
