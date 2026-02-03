"""
Test suite for LayerZero Device and GPU Generation Detection

Tests GPUGeneration enum, SM to generation mapping, tensor core detection.
Following TDD methodology - these tests define the expected behavior.
"""
import pytest


class TestGPUGeneration:
    """Test GPUGeneration enumeration."""

    def test_gpu_generation_turing_exists(self):
        """GPUGeneration.TURING exists."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.TURING.value == "turing"

    def test_gpu_generation_ampere_exists(self):
        """GPUGeneration.AMPERE exists."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.AMPERE.value == "ampere"

    def test_gpu_generation_ada_lovelace_exists(self):
        """GPUGeneration.ADA_LOVELACE exists."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.ADA_LOVELACE.value == "ada"

    def test_gpu_generation_hopper_exists(self):
        """GPUGeneration.HOPPER exists."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.HOPPER.value == "hopper"

    def test_gpu_generation_blackwell_exists(self):
        """GPUGeneration.BLACKWELL exists."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.BLACKWELL.value == "blackwell"

    def test_gpu_generation_unknown_exists(self):
        """GPUGeneration.UNKNOWN exists for fallback."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.UNKNOWN.value == "unknown"

    def test_gpu_generation_all_values(self):
        """GPUGeneration has all expected values."""
        from layerzero.device import GPUGeneration

        expected = ["unknown", "turing", "ampere", "ada", "hopper", "blackwell"]
        actual = [g.value for g in GPUGeneration]

        for exp in expected:
            assert exp in actual, f"Missing generation: {exp}"


class TestGPUGenerationOrdering:
    """Test GPUGeneration ordering (for comparison operators)."""

    def test_gpu_generation_ordering_turing_lt_ampere(self):
        """TURING < AMPERE."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.TURING < GPUGeneration.AMPERE

    def test_gpu_generation_ordering_ampere_lt_ada(self):
        """AMPERE < ADA_LOVELACE."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.AMPERE < GPUGeneration.ADA_LOVELACE

    def test_gpu_generation_ordering_ada_lt_hopper(self):
        """ADA_LOVELACE < HOPPER."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.ADA_LOVELACE < GPUGeneration.HOPPER

    def test_gpu_generation_ordering_hopper_lt_blackwell(self):
        """HOPPER < BLACKWELL."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.HOPPER < GPUGeneration.BLACKWELL

    def test_gpu_generation_ordering_unknown_lt_all(self):
        """UNKNOWN < all other generations."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.UNKNOWN < GPUGeneration.TURING
        assert GPUGeneration.UNKNOWN < GPUGeneration.BLACKWELL

    def test_gpu_generation_full_ordering(self):
        """Full ordering: UNKNOWN < TURING < AMPERE < ADA < HOPPER < BLACKWELL."""
        from layerzero.device import GPUGeneration

        ordered = [
            GPUGeneration.UNKNOWN,
            GPUGeneration.TURING,
            GPUGeneration.AMPERE,
            GPUGeneration.ADA_LOVELACE,
            GPUGeneration.HOPPER,
            GPUGeneration.BLACKWELL,
        ]

        for i in range(len(ordered) - 1):
            assert ordered[i] < ordered[i + 1]
            assert ordered[i + 1] > ordered[i]

    def test_gpu_generation_equality(self):
        """Same generation equals itself."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.AMPERE == GPUGeneration.AMPERE
        assert not (GPUGeneration.AMPERE < GPUGeneration.AMPERE)

    def test_gpu_generation_le_ge(self):
        """Less-than-or-equal and greater-than-or-equal work."""
        from layerzero.device import GPUGeneration
        assert GPUGeneration.TURING <= GPUGeneration.AMPERE
        assert GPUGeneration.TURING <= GPUGeneration.TURING
        assert GPUGeneration.AMPERE >= GPUGeneration.TURING
        assert GPUGeneration.AMPERE >= GPUGeneration.AMPERE


class TestSMToGenerationMapping:
    """Test SM version to GPU generation mapping."""

    def test_sm75_maps_to_turing(self):
        """SM 7.5 maps to TURING."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(7, 5) == GPUGeneration.TURING

    def test_sm80_maps_to_ampere(self):
        """SM 8.0 maps to AMPERE (A100)."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(8, 0) == GPUGeneration.AMPERE

    def test_sm86_maps_to_ampere(self):
        """SM 8.6 maps to AMPERE (RTX 30xx/A10)."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(8, 6) == GPUGeneration.AMPERE

    def test_sm87_maps_to_ampere(self):
        """SM 8.7 maps to AMPERE (Orin)."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(8, 7) == GPUGeneration.AMPERE

    def test_sm89_maps_to_ada(self):
        """SM 8.9 maps to ADA_LOVELACE (RTX 40xx)."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(8, 9) == GPUGeneration.ADA_LOVELACE

    def test_sm90_maps_to_hopper(self):
        """SM 9.0 maps to HOPPER (H100)."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(9, 0) == GPUGeneration.HOPPER

    def test_sm100_maps_to_blackwell(self):
        """SM 10.0 maps to BLACKWELL."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(10, 0) == GPUGeneration.BLACKWELL

    def test_sm120_maps_to_blackwell(self):
        """SM 12.0 maps to BLACKWELL (RTX 50xx)."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(12, 0) == GPUGeneration.BLACKWELL

    def test_unknown_sm_maps_to_unknown(self):
        """Unknown SM version maps to UNKNOWN."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(5, 0) == GPUGeneration.UNKNOWN
        assert sm_to_generation(6, 0) == GPUGeneration.UNKNOWN

    def test_future_sm_maps_to_blackwell_or_newer(self):
        """Future SM versions (>12) map to BLACKWELL or newer."""
        from layerzero.device import sm_to_generation, GPUGeneration
        # Any SM >= 10 should map to at least BLACKWELL
        result = sm_to_generation(15, 0)
        assert result >= GPUGeneration.BLACKWELL


class TestTensorCoreGeneration:
    """Test tensor core generation mapping."""

    def test_turing_tensor_core_gen2(self):
        """TURING has tensor_core_generation=2."""
        from layerzero.device import get_tensor_core_gen, GPUGeneration
        assert get_tensor_core_gen(GPUGeneration.TURING) == 2

    def test_ampere_tensor_core_gen3(self):
        """AMPERE has tensor_core_generation=3."""
        from layerzero.device import get_tensor_core_gen, GPUGeneration
        assert get_tensor_core_gen(GPUGeneration.AMPERE) == 3

    def test_ada_tensor_core_gen3(self):
        """ADA_LOVELACE has tensor_core_generation=3."""
        from layerzero.device import get_tensor_core_gen, GPUGeneration
        assert get_tensor_core_gen(GPUGeneration.ADA_LOVELACE) == 3

    def test_hopper_tensor_core_gen4(self):
        """HOPPER has tensor_core_generation=4."""
        from layerzero.device import get_tensor_core_gen, GPUGeneration
        assert get_tensor_core_gen(GPUGeneration.HOPPER) == 4

    def test_blackwell_tensor_core_gen5(self):
        """BLACKWELL has tensor_core_generation=5."""
        from layerzero.device import get_tensor_core_gen, GPUGeneration
        assert get_tensor_core_gen(GPUGeneration.BLACKWELL) == 5

    def test_unknown_tensor_core_gen0(self):
        """UNKNOWN has tensor_core_generation=0."""
        from layerzero.device import get_tensor_core_gen, GPUGeneration
        assert get_tensor_core_gen(GPUGeneration.UNKNOWN) == 0


class TestGPUGenerationSerialization:
    """Test GPU generation JSON serialization."""

    def test_gpu_generation_is_str_enum(self):
        """GPUGeneration is a string enum."""
        from layerzero.device import GPUGeneration
        assert isinstance(GPUGeneration.AMPERE, str)
        assert GPUGeneration.AMPERE == "ampere"

    def test_gpu_generation_json_roundtrip(self):
        """GPUGeneration can be serialized to/from JSON."""
        import json
        from layerzero.device import GPUGeneration

        gen = GPUGeneration.HOPPER
        json_str = json.dumps(gen.value)
        restored_value = json.loads(json_str)
        restored_gen = GPUGeneration(restored_value)

        assert restored_gen == gen

    def test_gpu_generation_from_string(self):
        """GPUGeneration can be constructed from string value."""
        from layerzero.device import GPUGeneration

        assert GPUGeneration("turing") == GPUGeneration.TURING
        assert GPUGeneration("ampere") == GPUGeneration.AMPERE
        assert GPUGeneration("hopper") == GPUGeneration.HOPPER


class TestSMToGenerationEdgeCases:
    """Test edge cases in SM to generation mapping."""

    def test_sm_negative_values(self):
        """Negative SM values return UNKNOWN."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(-1, 0) == GPUGeneration.UNKNOWN
        assert sm_to_generation(0, -1) == GPUGeneration.UNKNOWN

    def test_sm_very_high_minor_version(self):
        """High minor versions are handled correctly."""
        from layerzero.device import sm_to_generation, GPUGeneration
        # SM 8.99 should still map to Ada or Ampere family
        result = sm_to_generation(8, 99)
        assert result in [GPUGeneration.AMPERE, GPUGeneration.ADA_LOVELACE]

    def test_sm_zero_zero(self):
        """SM 0.0 returns UNKNOWN."""
        from layerzero.device import sm_to_generation, GPUGeneration
        assert sm_to_generation(0, 0) == GPUGeneration.UNKNOWN
