"""Pytest fixtures for Liger kernel tests."""
from __future__ import annotations

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Get test device (prefer CUDA if available)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@pytest.fixture
def liger_available() -> bool:
    """Check if Liger is available."""
    try:
        import liger_kernel  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def triton_available() -> bool:
    """Check if Triton is available."""
    try:
        import triton  # noqa: F401
        return True
    except ImportError:
        return False


@pytest.fixture
def skip_if_no_liger(liger_available: bool) -> None:
    """Skip test if Liger is not available."""
    if not liger_available:
        pytest.skip("Liger not installed")


@pytest.fixture
def skip_if_no_cuda() -> None:
    """Skip test if CUDA is not available."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


@pytest.fixture
def sample_hidden_states(device: torch.device) -> torch.Tensor:
    """Create sample hidden states tensor.

    Shape: (batch=2, seq=128, hidden_dim=768)
    """
    batch, seq, hidden_dim = 2, 128, 768
    dtype = torch.float16

    return torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype)


@pytest.fixture
def sample_rms_norm_weight(device: torch.device) -> torch.Tensor:
    """Create sample RMSNorm weight tensor.

    Shape: (hidden_dim=768,)
    """
    hidden_dim = 768
    dtype = torch.float16

    return torch.ones(hidden_dim, device=device, dtype=dtype)


@pytest.fixture
def sample_layer_norm_weight_bias(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample LayerNorm weight and bias tensors.

    Shape: (hidden_dim=768,)
    """
    hidden_dim = 768
    dtype = torch.float16

    weight = torch.ones(hidden_dim, device=device, dtype=dtype)
    bias = torch.zeros(hidden_dim, device=device, dtype=dtype)

    return weight, bias


@pytest.fixture
def sample_qk_for_rope(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample Q and K tensors for RoPE.

    Shape: (batch=2, seq=128, heads=8, dim=64)
    """
    batch, seq, heads, dim = 2, 128, 8, 64
    dtype = torch.float16

    q = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)
    k = torch.randn(batch, seq, heads, dim, device=device, dtype=dtype)

    return q, k


@pytest.fixture
def sample_rope_cos_sin(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample cos and sin tensors for RoPE.

    Shape: (seq=128, dim=64)
    """
    seq, dim = 128, 64
    dtype = torch.float16

    # Create position-dependent cos/sin
    positions = torch.arange(seq, device=device, dtype=dtype)
    freqs = torch.arange(dim // 2, device=device, dtype=dtype)
    freqs = 1.0 / (10000 ** (2 * freqs / dim))

    angles = positions[:, None] * freqs[None, :]
    cos = torch.cos(angles).repeat(1, 2)
    sin = torch.sin(angles).repeat(1, 2)

    return cos, sin


@pytest.fixture
def sample_gate_up(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample gate and up tensors for SwiGLU.

    Shape: (batch=2, seq=128, hidden_dim=768)
    """
    batch, seq, hidden_dim = 2, 128, 768
    dtype = torch.float16

    gate = torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype)
    up = torch.randn(batch, seq, hidden_dim, device=device, dtype=dtype)

    return gate, up


@pytest.fixture
def sample_logits_labels(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    """Create sample logits and labels for CrossEntropy.

    Logits shape: (batch*seq=256, vocab_size=32000)
    Labels shape: (batch*seq=256,)
    """
    batch_seq, vocab_size = 256, 32000
    dtype = torch.float32  # CrossEntropy usually uses FP32

    logits = torch.randn(batch_seq, vocab_size, device=device, dtype=dtype)
    labels = torch.randint(0, vocab_size, (batch_seq,), device=device)

    return logits, labels
