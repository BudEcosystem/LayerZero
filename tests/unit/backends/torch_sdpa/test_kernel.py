"""Tests for SDPA kernel wrapper function."""
from __future__ import annotations

import pytest
import torch

from layerzero.backends.torch_sdpa.kernel import (
    sdpa_forward,
    SDPAConfig,
)


class TestSDPAConfig:
    """Test SDPAConfig dataclass."""

    def test_config_creation(self) -> None:
        """Test SDPAConfig can be created."""
        config = SDPAConfig(
            is_causal=True,
            dropout_p=0.1,
            scale=None,
            enable_gqa=False,
        )
        assert config.is_causal is True
        assert config.dropout_p == 0.1

    def test_config_defaults(self) -> None:
        """Test SDPAConfig default values."""
        config = SDPAConfig()
        assert config.is_causal is False
        assert config.dropout_p == 0.0
        assert config.scale is None
        assert config.enable_gqa is False


class TestSDPAForward:
    """Test sdpa_forward function."""

    def test_forward_basic(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test basic forward pass."""
        output = sdpa_forward(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
        )

        assert output.shape == query_tensor.shape
        assert output.dtype == query_tensor.dtype

    def test_forward_with_config(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test forward with config object."""
        config = SDPAConfig(is_causal=True, scale=0.1)

        output = sdpa_forward(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            config=config,
        )

        assert output.shape == query_tensor.shape

    def test_forward_causal(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test forward with causal masking."""
        output = sdpa_forward(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            is_causal=True,
        )

        assert output.shape == query_tensor.shape

    def test_forward_with_mask(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        bool_mask: torch.Tensor,
    ) -> None:
        """Test forward with attention mask."""
        output = sdpa_forward(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            attn_mask=bool_mask,
        )

        assert output.shape == query_tensor.shape

    def test_forward_dropout_inference(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test forward with dropout in inference mode."""
        # Dropout should have no effect in eval mode
        output1 = sdpa_forward(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            dropout_p=0.5,
            training=False,
        )

        output2 = sdpa_forward(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            dropout_p=0.5,
            training=False,
        )

        # Should be identical without dropout
        torch.testing.assert_close(output1, output2)

    def test_forward_scale(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
    ) -> None:
        """Test forward with custom scale."""
        output_default = sdpa_forward(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
        )

        output_scaled = sdpa_forward(
            query=query_tensor,
            key=key_tensor,
            value=value_tensor,
            scale=0.5,
        )

        # Different scale should produce different output
        assert not torch.allclose(output_default, output_scaled, rtol=1e-3)


class TestSDPAForwardErrorHandling:
    """Test SDPA forward error handling."""

    def test_mask_plus_causal_error(
        self,
        query_tensor: torch.Tensor,
        key_tensor: torch.Tensor,
        value_tensor: torch.Tensor,
        bool_mask: torch.Tensor,
    ) -> None:
        """Test error when using mask + is_causal together."""
        with pytest.raises(ValueError, match="Cannot use both attn_mask and is_causal"):
            sdpa_forward(
                query=query_tensor,
                key=key_tensor,
                value=value_tensor,
                attn_mask=bool_mask,
                is_causal=True,
            )

    def test_shape_mismatch_error(
        self,
        device: torch.device,
    ) -> None:
        """Test error on shape mismatch."""
        q = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)
        k = torch.randn(2, 8, 32, 64, device=device, dtype=torch.float16)  # Different seq_len
        v = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float16)  # Mismatch with k

        with pytest.raises((RuntimeError, ValueError)):
            sdpa_forward(query=q, key=k, value=v)


class TestSDPAForwardBackends:
    """Test SDPA forward with different backends."""

    def test_forward_math_fallback(
        self,
        qkv_fp32: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    ) -> None:
        """Test forward uses math fallback for fp32."""
        q, k, v = qkv_fp32

        # fp32 forces math backend on many configs
        output = sdpa_forward(
            query=q,
            key=k,
            value=v,
            backend_hint="math",
        )

        assert output.shape == q.shape
        assert output.dtype == torch.float32
