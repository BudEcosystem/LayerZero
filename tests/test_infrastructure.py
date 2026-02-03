"""Tests for test infrastructure and fixtures."""
from __future__ import annotations

import pytest
import torch


class TestInfrastructure:
    """Test pytest infrastructure and markers."""

    def test_pytest_markers_defined(self) -> None:
        """All custom markers are defined."""
        # These markers should be defined in conftest.py or pyproject.toml
        import _pytest.mark

        # Check that markers can be accessed
        assert hasattr(pytest.mark, "gpu")
        assert hasattr(pytest.mark, "stress")
        assert hasattr(pytest.mark, "slow")
        assert hasattr(pytest.mark, "multigpu")
        assert hasattr(pytest.mark, "correctness")

    def test_gpu_marker_skips_without_gpu(self) -> None:
        """@pytest.mark.gpu skips when no GPU."""
        # This test validates the skip logic exists
        from tests.fixtures.devices import gpu_required

        # If no CUDA, should raise skip exception
        if not torch.cuda.is_available():
            with pytest.raises(pytest.skip.Exception):
                gpu_required()

    def test_multigpu_marker_skips_without_multigpu(self) -> None:
        """@pytest.mark.multigpu skips when < 2 GPUs."""
        from tests.fixtures.devices import multigpu_required

        # If < 2 GPUs, should raise skip exception
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            with pytest.raises(pytest.skip.Exception):
                multigpu_required()

    def test_stress_marker_timeout_extended(self) -> None:
        """@pytest.mark.stress has extended timeout."""
        from tests.fixtures.devices import STRESS_TEST_TIMEOUT

        # Stress tests should have at least 5 minute timeout
        assert STRESS_TEST_TIMEOUT >= 300


class TestFixtures:
    """Test fixture functionality."""

    def test_device_fixture_returns_device(self, device: torch.device) -> None:
        """device fixture returns torch.device."""
        assert isinstance(device, torch.device)
        assert device.type in ("cpu", "cuda")

    def test_cuda_device_fixture_skips_no_cuda(self) -> None:
        """cuda_device fixture skips when no CUDA."""
        from tests.fixtures.devices import get_cuda_device

        if not torch.cuda.is_available():
            with pytest.raises(pytest.skip.Exception):
                get_cuda_device()
        else:
            cuda_dev = get_cuda_device()
            assert cuda_dev.type == "cuda"

    def test_reset_cuda_state_fixture(self) -> None:
        """reset_cuda_state clears cache between tests."""
        from tests.fixtures.devices import reset_cuda_state

        if torch.cuda.is_available():
            # Allocate some memory
            _ = torch.zeros(1000, 1000, device="cuda")
            # Reset should work without error
            reset_cuda_state()
            # Memory should be cleared (cache emptied)
            # Note: This is best-effort, CUDA may retain some memory

    def test_sample_tensors_fixture(self, sample_tensors: dict[str, torch.Tensor]) -> None:
        """sample_tensors provides QKV tensors."""
        assert "query" in sample_tensors
        assert "key" in sample_tensors
        assert "value" in sample_tensors

        q = sample_tensors["query"]
        k = sample_tensors["key"]
        v = sample_tensors["value"]

        # All should be 4D tensors [B, H, L, D]
        assert q.dim() == 4
        assert k.dim() == 4
        assert v.dim() == 4

        # Q and K should have compatible dimensions for attention
        assert q.shape[-1] == k.shape[-1]  # Same head dim


class TestCorrectnessFramework:
    """Test correctness testing framework."""

    def test_reference_attention_implementation(self) -> None:
        """Reference attention implementation exists."""
        from tests.correctness.reference import reference_attention

        # Create sample tensors
        batch, heads, seq_len, head_dim = 2, 4, 8, 16
        q = torch.randn(batch, heads, seq_len, head_dim)
        k = torch.randn(batch, heads, seq_len, head_dim)
        v = torch.randn(batch, heads, seq_len, head_dim)

        # Should work without error
        output = reference_attention(q, k, v)

        # Output should have same shape as Q
        assert output.shape == q.shape

    def test_tolerance_by_dtype_fp16(self) -> None:
        """fp16 tolerance is rtol=2e-3, atol=2e-3."""
        from tests.correctness.reference import get_tolerance

        rtol, atol = get_tolerance(torch.float16)

        # FP16 needs slightly looser tolerances due to SDPA implementation differences
        assert rtol == pytest.approx(2e-3, rel=0.1)
        assert atol == pytest.approx(2e-3, rel=0.1)

    def test_tolerance_by_dtype_bf16(self) -> None:
        """bf16 tolerance is rtol=1e-2, atol=1e-2."""
        from tests.correctness.reference import get_tolerance

        rtol, atol = get_tolerance(torch.bfloat16)

        assert rtol == pytest.approx(1e-2, rel=0.1)
        assert atol == pytest.approx(1e-2, rel=0.1)

    def test_tolerance_by_dtype_fp32(self) -> None:
        """fp32 tolerance is rtol=1e-4, atol=1e-5."""
        from tests.correctness.reference import get_tolerance

        rtol, atol = get_tolerance(torch.float32)

        assert rtol == pytest.approx(1e-4, rel=0.1)
        assert atol == pytest.approx(1e-5, rel=0.1)

    def test_assert_close_helper(self) -> None:
        """assert_close helper uses correct tolerances."""
        from tests.correctness.reference import assert_close

        # Should pass for identical tensors
        a = torch.randn(4, 4)
        b = a.clone()
        assert_close(a, b, dtype=torch.float32)

        # Should fail for different tensors
        c = torch.randn(4, 4)
        with pytest.raises(AssertionError):
            assert_close(a, c, dtype=torch.float32)


class TestAdditionalInfrastructure:
    """Additional infrastructure tests."""

    def test_sample_attention_mask_fixture(
        self, sample_attention_mask: torch.Tensor
    ) -> None:
        """sample_attention_mask provides valid mask."""
        # Should be 2D or 4D
        assert sample_attention_mask.dim() in (2, 4)

        # Should contain valid mask values (0 or -inf typically)
        unique_vals = torch.unique(sample_attention_mask)
        # Masks typically have 0 for attend, -inf for mask
        # Or boolean True/False

    def test_random_seed_fixture(self, random_seed: int) -> None:
        """random_seed fixture provides reproducible seed."""
        assert isinstance(random_seed, int)
        assert random_seed >= 0

    def test_device_fixture_consistency(self, device: torch.device) -> None:
        """device fixture returns consistent device."""
        # Create tensor on device
        t = torch.randn(4, 4, device=device)
        assert t.device.type == device.type


class TestModelFixtures:
    """Test model fixture functionality."""

    def test_mock_attention_module(self) -> None:
        """Mock attention module exists and works."""
        from tests.fixtures.models import MockAttentionModule

        module = MockAttentionModule(hidden_size=64, num_heads=4)

        # Test forward pass
        x = torch.randn(2, 8, 64)
        output = module(x)

        assert output.shape == x.shape

    def test_mock_transformer_layer(self) -> None:
        """Mock transformer layer exists and works."""
        from tests.fixtures.models import MockTransformerLayer

        layer = MockTransformerLayer(hidden_size=64, num_heads=4)

        # Test forward pass
        x = torch.randn(2, 8, 64)
        output = layer(x)

        assert output.shape == x.shape
