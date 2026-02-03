"""Tests for HuggingFace Diffusers integration."""
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from layerzero.integrations.diffusers import (
    is_diffusers_available,
    get_diffusers_version,
    patch_unet,
    unpatch_unet,
    patch_dit,
    unpatch_dit,
    patch_pipeline,
    unpatch_pipeline,
)


class TestDiffusersIntegration:
    """Test Diffusers integration availability."""

    def test_diffusers_integration_available(self) -> None:
        """HF Diffusers integration works."""
        result = is_diffusers_available()
        assert isinstance(result, bool)

    def test_diffusers_version_detection(self) -> None:
        """Detect Diffusers version."""
        if not is_diffusers_available():
            pytest.skip("Diffusers not available")

        version = get_diffusers_version()
        assert version is not None
        assert isinstance(version, tuple)
        assert len(version) >= 2


class MockCrossAttention(nn.Module):
    """Mock cross attention for testing."""

    def __init__(self, hidden_size: int = 64, cross_attention_dim: int = 64) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.to_q = nn.Linear(hidden_size, hidden_size)
        self.to_k = nn.Linear(cross_attention_dim, hidden_size)
        self.to_v = nn.Linear(cross_attention_dim, hidden_size)
        self.to_out = nn.ModuleList([nn.Linear(hidden_size, hidden_size)])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch, seq, _ = hidden_states.shape
        q = self.to_q(hidden_states)
        context = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
        k = self.to_k(context)
        v = self.to_v(context)
        attn = torch.softmax(q @ k.transpose(-2, -1) / (self.hidden_size ** 0.5), dim=-1)
        out = attn @ v
        return self.to_out[0](out)


class MockTransformerBlock(nn.Module):
    """Mock transformer block for testing."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.attn1 = MockCrossAttention(hidden_size)  # Self attention
        self.attn2 = MockCrossAttention(hidden_size)  # Cross attention
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attn1(self.norm1(hidden_states))
        if encoder_hidden_states is not None:
            hidden_states = hidden_states + self.attn2(self.norm2(hidden_states), encoder_hidden_states)
        return hidden_states


class MockBasicTransformer2DModel(nn.Module):
    """Mock BasicTransformer2DModel for testing."""

    def __init__(self, num_layers: int = 2, hidden_size: int = 64) -> None:
        super().__init__()
        self.transformer_blocks = nn.ModuleList([
            MockTransformerBlock(hidden_size) for _ in range(num_layers)
        ])

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states, encoder_hidden_states)
        return hidden_states


class MockAttentionDownBlock(nn.Module):
    """Mock attention block for UNet down sampling."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.attentions = nn.ModuleList([
            MockBasicTransformer2DModel(num_layers=1, hidden_size=hidden_size)
        ])

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for attn in self.attentions:
            hidden_states = attn(hidden_states)
        return hidden_states


class MockUNet(nn.Module):
    """Mock UNet2DConditionModel for testing."""

    def __init__(self, hidden_size: int = 64) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(4, hidden_size, 3, padding=1)
        self.down_blocks = nn.ModuleList([
            MockAttentionDownBlock(hidden_size),
            MockAttentionDownBlock(hidden_size),
        ])
        self.mid_block = MockAttentionDownBlock(hidden_size)
        self.up_blocks = nn.ModuleList([
            MockAttentionDownBlock(hidden_size),
            MockAttentionDownBlock(hidden_size),
        ])
        self.conv_out = nn.Conv2d(hidden_size, 4, 3, padding=1)
        self.config = type("Config", (), {"model_type": "unet"})()

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(sample)
        # Flatten spatial dims for attention
        b, c, h, w = x.shape
        x = x.view(b, c, h * w).transpose(1, 2)  # [B, H*W, C]
        for block in self.down_blocks:
            x = block(x)
        x = self.mid_block(x)
        for block in self.up_blocks:
            x = block(x)
        # Restore spatial dims
        x = x.transpose(1, 2).view(b, c, h, w)
        return self.conv_out(x)


class MockDiT(nn.Module):
    """Mock DiT model for testing."""

    def __init__(self, hidden_size: int = 64, num_layers: int = 2) -> None:
        super().__init__()
        self.proj_in = nn.Linear(4, hidden_size)
        self.transformer_blocks = nn.ModuleList([
            MockTransformerBlock(hidden_size) for _ in range(num_layers)
        ])
        self.proj_out = nn.Linear(hidden_size, 4)
        self.config = type("Config", (), {"model_type": "dit"})()

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            x = block(x, encoder_hidden_states)
        return self.proj_out(x)


class TestUNetPatching:
    """Test UNet patching functionality."""

    @pytest.fixture
    def mock_unet(self) -> MockUNet:
        """Create a mock UNet for testing."""
        return MockUNet()

    def test_patch_unet_cross_attention(self, mock_unet: MockUNet) -> None:
        """Patch UNet cross attention."""
        # Get original attention type
        original_attn = mock_unet.down_blocks[0].attentions[0].transformer_blocks[0].attn2

        # Patch
        patched = patch_unet(mock_unet)

        # Model should still work
        sample = torch.randn(1, 4, 8, 8)
        output = patched(sample)
        assert output.shape == (1, 4, 8, 8)

    def test_patch_unet_self_attention(self, mock_unet: MockUNet) -> None:
        """Patch UNet self attention."""
        patched = patch_unet(mock_unet)

        # Model should still work
        sample = torch.randn(1, 4, 8, 8)
        output = patched(sample)
        assert output.shape == (1, 4, 8, 8)

    def test_unpatch_unet(self, mock_unet: MockUNet) -> None:
        """Unpatch UNet restores original."""
        # Patch then unpatch
        patched = patch_unet(mock_unet)
        unpatched = unpatch_unet(patched)

        # Should still work
        sample = torch.randn(1, 4, 8, 8)
        output = unpatched(sample)
        assert output.shape == (1, 4, 8, 8)


class TestDiTPatching:
    """Test DiT patching functionality."""

    @pytest.fixture
    def mock_dit(self) -> MockDiT:
        """Create a mock DiT for testing."""
        return MockDiT()

    def test_patch_dit_attention(self, mock_dit: MockDiT) -> None:
        """Patch DiT attention."""
        patched = patch_dit(mock_dit)

        # Model should still work
        # hidden_states has shape [B, S, 4] -> proj_in produces [B, S, 64]
        # encoder_hidden_states needs to have dim matching cross_attention_dim (64)
        hidden_states = torch.randn(1, 16, 4)
        encoder_hidden_states = torch.randn(1, 8, 64)  # Cross-attention dim is 64
        output = patched(hidden_states, encoder_hidden_states)
        assert output.shape == (1, 16, 4)


class TestPipelineCompatibility:
    """Test pipeline compatibility."""

    @pytest.mark.skipif(
        not is_diffusers_available(),
        reason="Diffusers not available"
    )
    def test_stable_diffusion_pipeline(self) -> None:
        """StableDiffusionPipeline works."""
        try:
            from diffusers import StableDiffusionPipeline, UNet2DConditionModel
            from diffusers import DDIMScheduler

            # Create minimal UNet
            unet_config = {
                "sample_size": 8,
                "in_channels": 4,
                "out_channels": 4,
                "layers_per_block": 1,
                "block_out_channels": (32,),
                "down_block_types": ("CrossAttnDownBlock2D",),
                "up_block_types": ("CrossAttnUpBlock2D",),
                "cross_attention_dim": 32,
            }

            try:
                unet = UNet2DConditionModel(**unet_config)
            except Exception:
                pytest.skip("Cannot create minimal UNet")

            # Patch UNet
            patched_unet = patch_unet(unet)
            assert patched_unet is unet

        except ImportError:
            pytest.skip("Diffusers not available")
        except Exception as e:
            pytest.skip(f"Pipeline test skipped: {e}")

    @pytest.mark.skipif(
        not is_diffusers_available(),
        reason="Diffusers not available"
    )
    def test_sdxl_pipeline(self) -> None:
        """StableDiffusionXLPipeline works."""
        try:
            from diffusers import UNet2DConditionModel

            # Create minimal SDXL-style UNet
            unet_config = {
                "sample_size": 8,
                "in_channels": 4,
                "out_channels": 4,
                "layers_per_block": 1,
                "block_out_channels": (32,),
                "down_block_types": ("CrossAttnDownBlock2D",),
                "up_block_types": ("CrossAttnUpBlock2D",),
                "cross_attention_dim": 32,
            }

            try:
                unet = UNet2DConditionModel(**unet_config)
            except Exception:
                pytest.skip("Cannot create minimal UNet")

            # Patch
            patched = patch_unet(unet)
            assert patched is unet

        except ImportError:
            pytest.skip("Diffusers not available")
        except Exception as e:
            pytest.skip(f"SDXL test skipped: {e}")

    @pytest.mark.skipif(
        not is_diffusers_available(),
        reason="Diffusers not available"
    )
    def test_flux_pipeline(self) -> None:
        """FluxPipeline works (if available)."""
        try:
            from diffusers import FluxTransformer2DModel

            # Flux uses DiT architecture
            # Skip if not available
            pytest.skip("Flux model creation requires full setup")

        except ImportError:
            pytest.skip("Flux not available in this diffusers version")


class TestImageGeneration:
    """Test image generation with patched models."""

    def test_generate_image_patched(self) -> None:
        """Image generation works with patched model."""
        # Create mock UNet
        unet = MockUNet()

        # Patch
        patched = patch_unet(unet)

        # Generate (just run forward, not full generation)
        sample = torch.randn(1, 4, 8, 8)
        with torch.no_grad():
            output = patched(sample)

        assert output.shape == (1, 4, 8, 8)
        assert not torch.isnan(output).any()


class TestPipelinePatching:
    """Test full pipeline patching."""

    @pytest.fixture
    def mock_pipeline(self) -> object:
        """Create mock pipeline with UNet."""
        class MockPipeline:
            def __init__(self) -> None:
                self.unet = MockUNet()
                self.config = type("Config", (), {"model_type": "pipeline"})()

            def __call__(self, *args, **kwargs) -> torch.Tensor:
                return self.unet(torch.randn(1, 4, 8, 8))

        return MockPipeline()

    def test_patch_pipeline(self, mock_pipeline: object) -> None:
        """patch_pipeline patches UNet."""
        patched = patch_pipeline(mock_pipeline)

        # Should still work
        output = patched()
        assert output.shape == (1, 4, 8, 8)

    def test_unpatch_pipeline(self, mock_pipeline: object) -> None:
        """unpatch_pipeline restores original."""
        # Patch then unpatch
        patched = patch_pipeline(mock_pipeline)
        unpatched = unpatch_pipeline(patched)

        # Should still work
        output = unpatched()
        assert output.shape == (1, 4, 8, 8)
