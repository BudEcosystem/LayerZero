"""Tests for tensor parallel invariance."""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from layerzero.distributed.tp_invariance import (
    TPConfig,
    TPInvarianceFilter,
    TPContext,
    get_tp_context,
    is_tp_enabled,
    require_tp_invariant,
)
from layerzero.reasons import (
    TP_INVARIANCE_REQUIRED,
    TP_INVARIANT_KERNEL_REQUIRED,
    Reason,
    ReasonCategory,
)


class TestTPConfig:
    """Tests for TPConfig dataclass."""

    def test_default_values(self) -> None:
        """Default config values."""
        config = TPConfig()

        assert config.require_invariant is False
        assert config.max_tp_size is None
        assert config.allow_non_invariant_inference is True
        assert config.strict_training is True

    def test_custom_values(self) -> None:
        """Custom config values."""
        config = TPConfig(
            require_invariant=True,
            max_tp_size=8,
            allow_non_invariant_inference=False,
            strict_training=False,
        )

        assert config.require_invariant is True
        assert config.max_tp_size == 8
        assert config.allow_non_invariant_inference is False
        assert config.strict_training is False

    def test_config_immutable(self) -> None:
        """Config is immutable."""
        config = TPConfig()

        with pytest.raises(AttributeError):
            config.require_invariant = True


class TestTPContext:
    """Tests for TPContext."""

    def test_context_creation(self) -> None:
        """TPContext stores values correctly."""
        ctx = TPContext(
            tp_size=4,
            tp_rank=2,
            enabled=True,
        )

        assert ctx.tp_size == 4
        assert ctx.tp_rank == 2
        assert ctx.enabled is True

    def test_context_disabled(self) -> None:
        """TPContext when TP disabled."""
        ctx = TPContext(
            tp_size=1,
            tp_rank=0,
            enabled=False,
        )

        assert ctx.tp_size == 1
        assert ctx.tp_rank == 0
        assert ctx.enabled is False

    def test_get_tp_context_no_tp(self, mock_dist_unavailable) -> None:
        """get_tp_context when TP not enabled."""
        ctx = get_tp_context()

        assert ctx.enabled is False
        assert ctx.tp_size == 1
        assert ctx.tp_rank == 0

    def test_get_tp_context_with_tp(self) -> None:
        """get_tp_context with TP environment variables."""
        with patch.dict('os.environ', {'TP_SIZE': '4', 'TP_RANK': '2'}):
            with patch('torch.distributed.is_available', return_value=True):
                with patch('torch.distributed.is_initialized', return_value=True):
                    ctx = get_tp_context()

        assert ctx.enabled is True
        assert ctx.tp_size == 4
        assert ctx.tp_rank == 2

    def test_is_tp_enabled_false(self, mock_dist_unavailable) -> None:
        """is_tp_enabled returns False when not in TP mode."""
        assert is_tp_enabled() is False

    def test_is_tp_enabled_true(self) -> None:
        """is_tp_enabled returns True in TP mode."""
        with patch.dict('os.environ', {'TP_SIZE': '4'}):
            with patch('torch.distributed.is_available', return_value=True):
                with patch('torch.distributed.is_initialized', return_value=True):
                    assert is_tp_enabled() is True


class TestTPInvarianceFilter:
    """Tests for TPInvarianceFilter."""

    def test_tp_invariant_mode_enabled(self) -> None:
        """TP-invariant mode can be enabled."""
        config = TPConfig(require_invariant=True)
        filter_ = TPInvarianceFilter(config=config)

        assert filter_.config.require_invariant is True

    def test_tp_invariant_kernel_selected(self, mock_tp_kernel) -> None:
        """TP-invariant kernel passes filter in TP mode."""
        config = TPConfig(require_invariant=True)
        filter_ = TPInvarianceFilter(config=config)

        ctx = MagicMock()
        ctx.tp_size = 4
        ctx.is_training = True

        passed, reason = filter_.check(mock_tp_kernel, ctx)

        assert passed is True
        assert reason is None

    def test_non_invariant_kernel_rejected_training(self, mock_kernel) -> None:
        """Non-invariant kernel rejected in training with require_invariant."""
        config = TPConfig(require_invariant=True, strict_training=True)
        filter_ = TPInvarianceFilter(config=config)

        ctx = MagicMock()
        ctx.tp_size = 4
        ctx.is_training = True

        passed, reason = filter_.check(mock_kernel, ctx)

        assert passed is False
        assert reason is not None
        assert reason.code == TP_INVARIANT_KERNEL_REQUIRED

    def test_non_invariant_kernel_allowed_inference(self, mock_kernel) -> None:
        """Non-invariant kernel allowed in inference if configured."""
        config = TPConfig(
            require_invariant=True,
            allow_non_invariant_inference=True,
        )
        filter_ = TPInvarianceFilter(config=config)

        ctx = MagicMock()
        ctx.tp_size = 4
        ctx.is_training = False

        passed, reason = filter_.check(mock_kernel, ctx)

        assert passed is True
        assert reason is None

    def test_tp_size_in_context(self) -> None:
        """tp_size is used from context."""
        config = TPConfig(require_invariant=False)
        filter_ = TPInvarianceFilter(config=config)

        kernel = MagicMock()
        kernel.tp_invariant = False
        kernel.kernel_id = "test"

        ctx = MagicMock()
        ctx.tp_size = 8
        ctx.is_training = False

        # Filter should use tp_size from context
        passed, reason = filter_.check(kernel, ctx)

        assert passed is True  # No requirement for invariant

    def test_tp_rank_awareness(self) -> None:
        """Filter is aware of TP rank."""
        config = TPConfig(require_invariant=False)
        filter_ = TPInvarianceFilter(config=config)

        kernel = MagicMock()
        kernel.tp_invariant = False
        kernel.kernel_id = "test"

        ctx = MagicMock()
        ctx.tp_size = 4
        ctx.tp_rank = 2
        ctx.is_training = False

        passed, reason = filter_.check(kernel, ctx)

        assert passed is True

    def test_no_tp_no_filter(self, mock_kernel) -> None:
        """No filtering when TP is not enabled."""
        config = TPConfig(require_invariant=True)
        filter_ = TPInvarianceFilter(config=config)

        ctx = MagicMock()
        ctx.tp_size = 1  # No TP
        ctx.is_training = True

        passed, reason = filter_.check(mock_kernel, ctx)

        assert passed is True
        assert reason is None

    def test_require_tp_invariant_decorator(self) -> None:
        """require_tp_invariant decorator marks function."""
        @require_tp_invariant
        def my_kernel():
            pass

        assert getattr(my_kernel, '_tp_invariant', False) is True

    def test_max_tp_size_exceeded(self, mock_tp_kernel) -> None:
        """Kernel rejected if TP size exceeds maximum."""
        config = TPConfig(max_tp_size=4)
        filter_ = TPInvarianceFilter(config=config)

        ctx = MagicMock()
        ctx.tp_size = 8  # Exceeds max
        ctx.is_training = False

        passed, reason = filter_.check(mock_tp_kernel, ctx)

        assert passed is False
        assert reason is not None
        assert "TP size" in reason.message


class TestTPInvarianceIntegration:
    """Integration tests for TP invariance."""

    def test_filter_multiple_kernels(self) -> None:
        """Filter multiple kernels by TP invariance."""
        config = TPConfig(require_invariant=True, strict_training=True)
        filter_ = TPInvarianceFilter(config=config)

        # Create kernels with different invariance
        invariant_kernel = MagicMock()
        invariant_kernel.kernel_id = "invariant"
        invariant_kernel.tp_invariant = True

        non_invariant_kernel = MagicMock()
        non_invariant_kernel.kernel_id = "non_invariant"
        non_invariant_kernel.tp_invariant = False

        ctx = MagicMock()
        ctx.tp_size = 4
        ctx.is_training = True

        inv_passed, _ = filter_.check(invariant_kernel, ctx)
        non_inv_passed, _ = filter_.check(non_invariant_kernel, ctx)

        assert inv_passed is True
        assert non_inv_passed is False

    def test_tp_context_propagation(self) -> None:
        """TP context propagates correctly through filter."""
        config = TPConfig(require_invariant=True)
        filter_ = TPInvarianceFilter(config=config)

        kernel = MagicMock()
        kernel.kernel_id = "test"
        kernel.tp_invariant = True

        # Simulate different TP ranks
        for tp_rank in range(4):
            ctx = MagicMock()
            ctx.tp_size = 4
            ctx.tp_rank = tp_rank
            ctx.is_training = False

            passed, _ = filter_.check(kernel, ctx)
            assert passed is True

    def test_reason_category_distributed(self, mock_kernel) -> None:
        """Rejection reasons have DISTRIBUTED category."""
        config = TPConfig(require_invariant=True, strict_training=True)
        filter_ = TPInvarianceFilter(config=config)

        ctx = MagicMock()
        ctx.tp_size = 4
        ctx.is_training = True

        passed, reason = filter_.check(mock_kernel, ctx)

        assert passed is False
        assert reason.category == ReasonCategory.DISTRIBUTED
