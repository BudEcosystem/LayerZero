"""Tests for xFormers attention bias handling."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.xformers.bias import (
    validate_attn_bias,
    expand_attn_bias,
    check_bias_device,
    check_bias_broadcast,
)
from layerzero.reasons import (
    ATTN_BIAS_DEVICE_MISMATCH,
    ATTN_BIAS_BROADCAST_BATCH,
    ATTN_BIAS_BROADCAST_HEAD,
)


class TestBiasDeviceCheck:
    """Test attention bias device validation."""

    def test_same_device_accepted(self, device: torch.device) -> None:
        """Bias on same device as query is accepted."""
        batch, heads, seq_q, seq_k = 2, 8, 128, 128
        bias = torch.zeros(batch, heads, seq_q, seq_k, device=device, dtype=torch.float16)
        q = torch.randn(batch, seq_q, heads, 64, device=device, dtype=torch.float16)

        reasons = check_bias_device(bias, q)
        assert len(reasons) == 0

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_different_device_rejected(self) -> None:
        """Bias on different device is rejected."""
        batch, heads, seq_q, seq_k = 2, 8, 128, 128

        bias = torch.zeros(batch, heads, seq_q, seq_k, device="cpu", dtype=torch.float16)
        q = torch.randn(batch, seq_q, heads, 64, device="cuda", dtype=torch.float16)

        reasons = check_bias_device(bias, q)
        assert len(reasons) == 1
        assert ATTN_BIAS_DEVICE_MISMATCH in reasons[0].code


class TestBiasBroadcastCheck:
    """Test attention bias broadcast validation."""

    def test_full_dims_accepted(self, device: torch.device) -> None:
        """Fully expanded bias is accepted."""
        batch, heads, seq_q, seq_k = 2, 8, 128, 128
        bias = torch.zeros(batch, heads, seq_q, seq_k, device=device, dtype=torch.float16)

        reasons = check_bias_broadcast(bias, batch=batch, heads=heads)
        assert len(reasons) == 0

    def test_broadcast_batch_rejected(self, device: torch.device) -> None:
        """Broadcast batch dimension is rejected."""
        batch, heads, seq_q, seq_k = 2, 8, 128, 128
        # Batch dim is 1 (broadcast)
        bias = torch.zeros(1, heads, seq_q, seq_k, device=device, dtype=torch.float16)

        reasons = check_bias_broadcast(bias, batch=batch, heads=heads)
        assert len(reasons) == 1
        assert ATTN_BIAS_BROADCAST_BATCH in reasons[0].code

    def test_broadcast_head_rejected(self, device: torch.device) -> None:
        """Broadcast head dimension is rejected."""
        batch, heads, seq_q, seq_k = 2, 8, 128, 128
        # Head dim is 1 (broadcast)
        bias = torch.zeros(batch, 1, seq_q, seq_k, device=device, dtype=torch.float16)

        reasons = check_bias_broadcast(bias, batch=batch, heads=heads)
        assert len(reasons) == 1
        assert ATTN_BIAS_BROADCAST_HEAD in reasons[0].code

    def test_broadcast_both_rejected(self, device: torch.device) -> None:
        """Broadcast both batch and head dims is rejected."""
        batch, heads, seq_q, seq_k = 2, 8, 128, 128
        # Both dims are 1 (broadcast)
        bias = torch.zeros(1, 1, seq_q, seq_k, device=device, dtype=torch.float16)

        reasons = check_bias_broadcast(bias, batch=batch, heads=heads)
        assert len(reasons) == 2


class TestValidateAttnBias:
    """Test combined attention bias validation."""

    def test_valid_bias_passes(self, device: torch.device) -> None:
        """Valid bias passes all checks."""
        batch, heads, seq_q, seq_k, dim = 2, 8, 128, 128, 64
        bias = torch.zeros(batch, heads, seq_q, seq_k, device=device, dtype=torch.float16)
        q = torch.randn(batch, seq_q, heads, dim, device=device, dtype=torch.float16)

        reasons = validate_attn_bias(bias, q, batch=batch, heads=heads)
        assert len(reasons) == 0

    def test_none_bias_passes(self, device: torch.device) -> None:
        """None bias passes (no validation needed)."""
        batch, heads, seq_q, dim = 2, 8, 128, 64
        q = torch.randn(batch, seq_q, heads, dim, device=device, dtype=torch.float16)

        reasons = validate_attn_bias(None, q, batch=batch, heads=heads)
        assert len(reasons) == 0


class TestExpandAttnBias:
    """Test attention bias expansion."""

    def test_expand_batch_dim(self, device: torch.device) -> None:
        """Expand batch dimension from 1."""
        heads, seq_q, seq_k = 8, 128, 128
        bias = torch.zeros(1, heads, seq_q, seq_k, device=device, dtype=torch.float16)

        result = expand_attn_bias(bias, batch=4, heads=heads)

        assert result.shape == (4, heads, seq_q, seq_k)

    def test_expand_head_dim(self, device: torch.device) -> None:
        """Expand head dimension from 1."""
        batch, seq_q, seq_k = 2, 128, 128
        bias = torch.zeros(batch, 1, seq_q, seq_k, device=device, dtype=torch.float16)

        result = expand_attn_bias(bias, batch=batch, heads=8)

        assert result.shape == (batch, 8, seq_q, seq_k)

    def test_expand_both_dims(self, device: torch.device) -> None:
        """Expand both batch and head dimensions."""
        seq_q, seq_k = 128, 128
        bias = torch.zeros(1, 1, seq_q, seq_k, device=device, dtype=torch.float16)

        result = expand_attn_bias(bias, batch=4, heads=8)

        assert result.shape == (4, 8, seq_q, seq_k)

    def test_no_expansion_needed(self, device: torch.device) -> None:
        """No expansion when dims already match."""
        batch, heads, seq_q, seq_k = 4, 8, 128, 128
        bias = torch.zeros(batch, heads, seq_q, seq_k, device=device, dtype=torch.float16)

        result = expand_attn_bias(bias, batch=batch, heads=heads)

        # Should return same tensor (no copy)
        assert result.shape == (batch, heads, seq_q, seq_k)
        assert result.data_ptr() == bias.data_ptr()

    def test_expansion_preserves_values(self, device: torch.device) -> None:
        """Expansion preserves tensor values."""
        heads, seq_q, seq_k = 2, 4, 4
        # Create bias with specific values
        bias = torch.arange(32, device=device, dtype=torch.float16).reshape(1, heads, seq_q, seq_k)

        result = expand_attn_bias(bias, batch=3, heads=heads)

        # All batch elements should have same values as original
        torch.testing.assert_close(result[0], bias[0])
        torch.testing.assert_close(result[1], bias[0])
        torch.testing.assert_close(result[2], bias[0])

    def test_expansion_preserves_dtype(self, device: torch.device) -> None:
        """Expansion preserves dtype."""
        bias = torch.zeros(1, 1, 16, 16, device=device, dtype=torch.bfloat16)

        result = expand_attn_bias(bias, batch=2, heads=4)

        assert result.dtype == torch.bfloat16

    def test_expansion_preserves_device(self, device: torch.device) -> None:
        """Expansion preserves device."""
        bias = torch.zeros(1, 1, 16, 16, device=device, dtype=torch.float16)

        result = expand_attn_bias(bias, batch=2, heads=4)

        assert result.device == bias.device
