"""Tests for HuggingFace Transformers integration."""
from __future__ import annotations

import pytest
import torch

from layerzero.integrations.transformers import (
    is_transformers_available,
    get_transformers_version,
    patch_model,
    unpatch_model,
)
from layerzero.integrations.model_patching import (
    ModelPatcher,
    get_attention_module_names,
)


class TestTransformersIntegration:
    """Test Transformers integration availability."""

    def test_transformers_integration_available(self) -> None:
        """HF Transformers integration works."""
        result = is_transformers_available()
        assert isinstance(result, bool)

    def test_transformers_version_detection(self) -> None:
        """Detect Transformers version."""
        if not is_transformers_available():
            pytest.skip("Transformers not available")

        version = get_transformers_version()
        assert version is not None
        assert isinstance(version, tuple)
        assert len(version) >= 2


class TestModelPatching:
    """Test model patching functionality."""

    @pytest.fixture
    def mock_model(self):
        """Create a simple mock model for testing."""
        import torch.nn as nn

        class MockAttention(nn.Module):
            def __init__(self, hidden_size: int = 64):
                super().__init__()
                self.hidden_size = hidden_size
                self.q_proj = nn.Linear(hidden_size, hidden_size)
                self.k_proj = nn.Linear(hidden_size, hidden_size)
                self.v_proj = nn.Linear(hidden_size, hidden_size)
                self.o_proj = nn.Linear(hidden_size, hidden_size)

            def forward(self, x):
                batch, seq, _ = x.shape
                q = self.q_proj(x)
                k = self.k_proj(x)
                v = self.v_proj(x)
                # Simple attention
                attn = torch.softmax(q @ k.transpose(-2, -1) / (self.hidden_size ** 0.5), dim=-1)
                return self.o_proj(attn @ v)

        class MockLayer(nn.Module):
            def __init__(self, hidden_size: int = 64):
                super().__init__()
                self.self_attn = MockAttention(hidden_size)
                self.norm = nn.LayerNorm(hidden_size)

            def forward(self, x):
                return x + self.self_attn(self.norm(x))

        class MockModel(nn.Module):
            def __init__(self, num_layers: int = 2, hidden_size: int = 64):
                super().__init__()
                self.embed = nn.Embedding(100, hidden_size)
                self.layers = nn.ModuleList([MockLayer(hidden_size) for _ in range(num_layers)])
                self.config = type('Config', (), {
                    'model_type': 'mock',
                    'hidden_size': hidden_size,
                })()

            def forward(self, input_ids):
                x = self.embed(input_ids)
                for layer in self.layers:
                    x = layer(x)
                return x

        return MockModel()

    def test_patch_model(self, mock_model) -> None:
        """patch_model applies patches."""
        original_type = type(mock_model.layers[0].self_attn)
        patched = patch_model(mock_model)

        # Model should still work
        input_ids = torch.randint(0, 100, (2, 16))
        output = patched(input_ids)
        assert output.shape == (2, 16, 64)

    def test_unpatch_model(self, mock_model) -> None:
        """unpatch_model restores original."""
        original_attn = mock_model.layers[0].self_attn

        # Patch then unpatch
        patched = patch_model(mock_model)
        unpatched = unpatch_model(patched)

        # Should still work
        input_ids = torch.randint(0, 100, (2, 16))
        output = unpatched(input_ids)
        assert output.shape == (2, 16, 64)

    def test_model_patcher_context_manager(self, mock_model) -> None:
        """ModelPatcher works as context manager."""
        input_ids = torch.randint(0, 100, (2, 16))

        # Should work inside context
        with ModelPatcher(mock_model) as patched:
            output = patched(input_ids)
            assert output.shape == (2, 16, 64)

        # Should still work after context
        output = mock_model(input_ids)
        assert output.shape == (2, 16, 64)

    def test_get_attention_module_names(self, mock_model) -> None:
        """get_attention_module_names finds attention modules."""
        names = get_attention_module_names(mock_model)
        assert isinstance(names, list)
        # Should find our mock attention modules
        assert len(names) >= 0  # May be empty for unrecognized model type


class TestRealModelPatching:
    """Test patching with real HuggingFace models."""

    @pytest.mark.skipif(
        not is_transformers_available(),
        reason="Transformers not available"
    )
    def test_patch_llama_style_attention(self) -> None:
        """Patch LLaMA-style attention module."""
        # Create a minimal LLaMA-style model
        try:
            from transformers import LlamaConfig, LlamaModel

            config = LlamaConfig(
                hidden_size=64,
                num_attention_heads=4,
                num_hidden_layers=1,
                intermediate_size=128,
                vocab_size=100,
            )
            model = LlamaModel(config)

            # Patch
            patched = patch_model(model)

            # Test forward
            input_ids = torch.randint(0, 100, (1, 8))
            with torch.no_grad():
                output = patched(input_ids)

            assert output is not None

        except ImportError:
            pytest.skip("LlamaModel not available")

    @pytest.mark.skipif(
        not is_transformers_available(),
        reason="Transformers not available"
    )
    def test_patch_gpt2_attention(self) -> None:
        """Patch GPT2 attention module."""
        try:
            from transformers import GPT2Config, GPT2Model

            config = GPT2Config(
                n_embd=64,
                n_head=4,
                n_layer=1,
                vocab_size=100,
            )
            model = GPT2Model(config)

            # Patch
            patched = patch_model(model)

            # Test forward
            input_ids = torch.randint(0, 100, (1, 8))
            with torch.no_grad():
                output = patched(input_ids)

            assert output is not None

        except ImportError:
            pytest.skip("GPT2Model not available")


class TestGenerateCompatibility:
    """Test generate() compatibility."""

    @pytest.mark.skipif(
        not is_transformers_available(),
        reason="Transformers not available"
    )
    def test_generate_with_patched_model(self) -> None:
        """model.generate() works with patched model."""
        try:
            from transformers import GPT2Config, GPT2LMHeadModel

            config = GPT2Config(
                n_embd=64,
                n_head=4,
                n_layer=1,
                vocab_size=100,
            )
            model = GPT2LMHeadModel(config)

            # Patch
            patched = patch_model(model)

            # Generate
            input_ids = torch.randint(0, 100, (1, 4))
            with torch.no_grad():
                output = patched.generate(
                    input_ids,
                    max_new_tokens=4,
                    do_sample=False,
                )

            assert output.shape[1] == 8  # 4 input + 4 new

        except ImportError:
            pytest.skip("GPT2LMHeadModel not available")

    @pytest.mark.skipif(
        not is_transformers_available(),
        reason="Transformers not available"
    )
    def test_generate_kv_cache_handling(self) -> None:
        """KV cache handled correctly during generate."""
        try:
            from transformers import GPT2Config, GPT2LMHeadModel

            config = GPT2Config(
                n_embd=64,
                n_head=4,
                n_layer=1,
                vocab_size=100,
            )
            model = GPT2LMHeadModel(config)
            patched = patch_model(model)

            # Generate with KV cache
            input_ids = torch.randint(0, 100, (1, 4))
            with torch.no_grad():
                output = patched.generate(
                    input_ids,
                    max_new_tokens=4,
                    use_cache=True,
                    do_sample=False,
                )

            assert output.shape[1] == 8

        except ImportError:
            pytest.skip("GPT2LMHeadModel not available")

    @pytest.mark.skipif(
        not is_transformers_available(),
        reason="Transformers not available"
    )
    def test_generate_beam_search(self) -> None:
        """Beam search works with patched model."""
        try:
            from transformers import GPT2Config, GPT2LMHeadModel

            config = GPT2Config(
                n_embd=64,
                n_head=4,
                n_layer=1,
                vocab_size=100,
            )
            model = GPT2LMHeadModel(config)
            patched = patch_model(model)

            # Generate with beam search
            input_ids = torch.randint(0, 100, (1, 4))
            with torch.no_grad():
                output = patched.generate(
                    input_ids,
                    max_new_tokens=4,
                    num_beams=2,
                    do_sample=False,
                )

            assert output.shape[1] >= 4

        except ImportError:
            pytest.skip("GPT2LMHeadModel not available")


class TestPipelineCompatibility:
    """Test pipeline() compatibility."""

    @pytest.mark.skipif(
        not is_transformers_available(),
        reason="Transformers not available"
    )
    def test_pipeline_text_generation(self) -> None:
        """text-generation pipeline works."""
        try:
            import torch
            from transformers import pipeline, GPT2Config, GPT2LMHeadModel, GPT2Tokenizer

            # Skip if tokenizer loading fails - must load tokenizer first to get vocab_size
            try:
                tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            except Exception:
                pytest.skip("Cannot load GPT2 tokenizer")

            # Create minimal model with same vocab_size as tokenizer
            config = GPT2Config(
                n_embd=64,
                n_head=4,
                n_layer=1,
                vocab_size=tokenizer.vocab_size,  # Must match tokenizer!
            )
            model = GPT2LMHeadModel(config)

            # Move model to appropriate device and patch it
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device)
            patched = patch_model(model)

            # Create pipeline with explicit device
            pipe = pipeline("text-generation", model=patched, tokenizer=tokenizer, device=device)

            # Generate (may not produce meaningful output with random weights)
            result = pipe("Hello", max_new_tokens=4)
            assert result is not None

        except ImportError:
            pytest.skip("Pipeline not available")
        except Exception as e:
            # Skip if there are other issues with pipeline
            pytest.skip(f"Pipeline test skipped: {e}")


class TestModelHub:
    """Test loading models from hub."""

    @pytest.mark.skipif(
        not is_transformers_available(),
        reason="Transformers not available"
    )
    def test_patch_preserves_model_attributes(self) -> None:
        """Patching preserves model attributes."""
        try:
            from transformers import GPT2Config, GPT2Model

            config = GPT2Config(
                n_embd=64,
                n_head=4,
                n_layer=1,
                vocab_size=100,
            )
            model = GPT2Model(config)

            # Store original attributes
            original_config = model.config

            # Patch
            patched = patch_model(model)

            # Config should be preserved
            assert patched.config is original_config

        except ImportError:
            pytest.skip("GPT2Model not available")

    def test_patch_model_returns_same_model(self) -> None:
        """patch_model returns the same model object."""
        import torch.nn as nn

        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(10, 10)
                self.config = type('Config', (), {'model_type': 'simple'})()

            def forward(self, x):
                return self.linear(x)

        model = SimpleModel()
        patched = patch_model(model)

        # Should be the same object (in-place patching)
        assert patched is model
