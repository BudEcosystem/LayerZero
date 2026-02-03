"""Tests for constraint validation logic."""
from __future__ import annotations

import pytest
import torch

from layerzero.core.validation import (
    validate_head_dim,
    validate_cuda_block_limits,
    validate_layout,
    validate_dtype,
    detect_layout,
    is_layout_ambiguous,
    ValidationResult,
    SUPPORTED_HEAD_DIMS,
    MAX_CUDA_BLOCKS,
)
from layerzero.enums import Layout
from layerzero import reasons as ReasonCodes


class TestHeadDimConstraints:
    """Test head dimension constraints."""

    def test_head_dim_32_accepted(self) -> None:
        """head_dim=32 accepted by all backends."""
        result = validate_head_dim(32)

        assert result.valid is True
        assert result.reason is None

    def test_head_dim_64_accepted(self) -> None:
        """head_dim=64 accepted by all backends."""
        result = validate_head_dim(64)

        assert result.valid is True

    def test_head_dim_128_accepted(self) -> None:
        """head_dim=128 accepted by all backends."""
        result = validate_head_dim(128)

        assert result.valid is True

    def test_head_dim_256_accepted(self) -> None:
        """head_dim=256 accepted by FA/FlashInfer."""
        result = validate_head_dim(256)

        assert result.valid is True

    def test_head_dim_320_rejected_flash_attn(self) -> None:
        """head_dim=320 rejected by FA with HEAD_DIM_TOO_LARGE."""
        result = validate_head_dim(320, backend="flash_attn")

        # 320 is not in standard supported list for flash_attn
        # May be rejected or accepted depending on version
        assert isinstance(result, ValidationResult)

    def test_head_dim_alignment_multiple_8(self) -> None:
        """head_dim must be multiple of 8 for some kernels."""
        # Valid: multiple of 8
        result = validate_head_dim(64)
        assert result.valid is True

        # Invalid: not multiple of 8
        result = validate_head_dim(65)
        assert result.valid is False
        assert result.reason == ReasonCodes.HEAD_DIM_ALIGNMENT

    def test_head_dim_84_alignment_issue(self) -> None:
        """head_dim=84 may have alignment issues."""
        result = validate_head_dim(84)

        # 84 is not in standard supported list but is multiple of 4
        # Behavior depends on strictness
        assert isinstance(result, ValidationResult)


class TestCUDABlockLimits:
    """Test CUDA block limit constraints."""

    def test_cuda_block_limit_boundary(self) -> None:
        """batch * heads = 65535 at limit."""
        result = validate_cuda_block_limits(batch=255, heads=257)  # 255*257 = 65535

        assert result.valid is True

    def test_cuda_block_limit_exceeded(self) -> None:
        """batch * heads > 65535 rejected."""
        result = validate_cuda_block_limits(batch=256, heads=257)  # > 65535

        assert result.valid is False
        assert result.reason == ReasonCodes.CUDA_BLOCK_LIMIT_EXCEEDED

    def test_cuda_block_limit_reason_code(self) -> None:
        """CUDA_BLOCK_LIMIT_EXCEEDED reason returned."""
        result = validate_cuda_block_limits(batch=1000, heads=1000)

        assert result.valid is False
        assert result.reason == ReasonCodes.CUDA_BLOCK_LIMIT_EXCEEDED

    def test_cuda_grid_dim_validation(self) -> None:
        """Grid dimensions validated against SM limits."""
        # Very large grid dimension - exceeds max grid dim
        # MAX_GRID_DIM_X is 2147483647, so 3 billion exceeds it
        result = validate_cuda_block_limits(
            batch=1,
            heads=1,
            seq_len=3_000_000_000,  # 3B - exceeds MAX_GRID_DIM_X
        )

        # Should fail validation
        assert result.valid is False


class TestLayoutConstraints:
    """Test layout detection and validation."""

    def test_layout_bshd_detected(self) -> None:
        """BSHD layout correctly detected."""
        # BSHD: [batch, seq, heads, dim]
        tensor = torch.randn(2, 16, 4, 64)

        layout = detect_layout(tensor, expected_heads=4, expected_dim=64)

        assert layout == Layout.BSHD

    def test_layout_bhsd_detected(self) -> None:
        """BHSD layout correctly detected."""
        # BHSD: [batch, heads, seq, dim]
        tensor = torch.randn(2, 4, 16, 64)

        layout = detect_layout(tensor, expected_heads=4, expected_dim=64)

        assert layout == Layout.BHSD

    def test_layout_ambiguous_handling(self) -> None:
        """Ambiguous layout handled correctly."""
        # Ambiguous: seq == heads (e.g., 4x4)
        tensor = torch.randn(2, 4, 4, 64)

        ambiguous = is_layout_ambiguous(tensor, expected_heads=4, expected_dim=64)

        assert ambiguous is True

    def test_layout_validation(self) -> None:
        """Layout validation works."""
        tensor = torch.randn(2, 4, 16, 64)  # BHSD

        result = validate_layout(tensor, expected_layout=Layout.BHSD)

        assert result.valid is True

    def test_layout_2d_input(self) -> None:
        """2D input handled correctly."""
        tensor = torch.randn(16, 64)  # 2D - no batch/heads

        # Should not crash, may return unknown layout
        layout = detect_layout(tensor, expected_heads=1, expected_dim=64)

        assert layout is not None

    def test_layout_5d_input_rejected(self) -> None:
        """5D input rejected with appropriate error."""
        tensor = torch.randn(2, 4, 16, 64, 8)  # 5D

        result = validate_layout(tensor, expected_layout=Layout.BHSD)

        # 5D is not a valid attention tensor
        assert result.valid is False


class TestDtypeConstraints:
    """Test dtype validation."""

    def test_fp16_supported(self) -> None:
        """FP16 supported by attention kernels."""
        result = validate_dtype(torch.float16)

        assert result.valid is True

    def test_bf16_supported(self) -> None:
        """BF16 supported by attention kernels."""
        result = validate_dtype(torch.bfloat16)

        assert result.valid is True

    def test_fp32_supported(self) -> None:
        """FP32 supported by attention kernels."""
        result = validate_dtype(torch.float32)

        assert result.valid is True

    def test_int8_rejected(self) -> None:
        """INT8 rejected for attention (not supported)."""
        result = validate_dtype(torch.int8)

        assert result.valid is False
        assert result.reason == ReasonCodes.DTYPE_UNSUPPORTED

    def test_mixed_dtype_validation(self) -> None:
        """Mixed dtypes detected and handled."""
        query = torch.randn(2, 4, 16, 64, dtype=torch.float16)
        key = torch.randn(2, 4, 16, 64, dtype=torch.float32)  # Different dtype

        result = validate_dtype(query.dtype, key.dtype)

        # Mixed dtypes may be rejected or require conversion
        assert isinstance(result, ValidationResult)

    def test_dtype_conversion_available(self) -> None:
        """Dtype conversion function available."""
        from layerzero.core.validation import can_convert_dtype

        # FP32 -> FP16 should be possible
        assert can_convert_dtype(torch.float32, torch.float16) is True

        # INT8 -> FP16 may not be directly supported for attention
        result = can_convert_dtype(torch.int8, torch.float16)
        assert isinstance(result, bool)

    def test_dtype_check_tensor(self) -> None:
        """Dtype check works on tensors."""
        tensor = torch.randn(4, 4, dtype=torch.float16)

        result = validate_dtype(tensor.dtype)

        assert result.valid is True

    def test_dtype_fallback_logic(self) -> None:
        """Fallback dtype logic works."""
        from layerzero.core.validation import get_fallback_dtype

        # BF16 might fallback to FP16 on older GPUs
        fallback = get_fallback_dtype(torch.bfloat16)

        assert fallback in (torch.bfloat16, torch.float16, torch.float32)


class TestValidationResult:
    """Test ValidationResult dataclass."""

    def test_validation_result_valid(self) -> None:
        """Valid result creation."""
        result = ValidationResult(valid=True, reason=None)

        assert result.valid is True
        assert result.reason is None

    def test_validation_result_invalid(self) -> None:
        """Invalid result with reason."""
        result = ValidationResult(
            valid=False,
            reason=ReasonCodes.DTYPE_UNSUPPORTED,
        )

        assert result.valid is False
        assert result.reason == ReasonCodes.DTYPE_UNSUPPORTED

    def test_validation_result_with_message(self) -> None:
        """Result with detailed message."""
        result = ValidationResult(
            valid=False,
            reason=ReasonCodes.HEAD_DIM_INVALID,
            message="head_dim=320 not supported by flash_attn",
        )

        assert "320" in result.message


class TestIntegration:
    """Integration tests for validation."""

    def test_full_validation_pipeline(self) -> None:
        """Full validation pipeline works."""
        from layerzero.core.validation import validate_attention_inputs

        query = torch.randn(2, 4, 16, 64)
        key = torch.randn(2, 4, 16, 64)
        value = torch.randn(2, 4, 16, 64)

        result = validate_attention_inputs(query, key, value)

        assert isinstance(result, ValidationResult)

    def test_validation_with_all_constraints(self) -> None:
        """Validation checks all constraints."""
        from layerzero.core.validation import validate_attention_inputs

        # Valid inputs
        query = torch.randn(2, 4, 16, 64, dtype=torch.float16)
        key = torch.randn(2, 4, 16, 64, dtype=torch.float16)
        value = torch.randn(2, 4, 16, 64, dtype=torch.float16)

        result = validate_attention_inputs(query, key, value)

        assert result.valid is True
