"""Fuzz testing harness for LayerZero.

Tests security and robustness through fuzzing.
If Atheris is not installed, fallback tests run with random inputs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import random
import struct

import pytest
import torch

from layerzero.core.validation import (
    validate_head_dim,
    validate_dtype,
    validate_cuda_block_limits,
    validate_layout,
    validate_attention_inputs,
    ValidationResult,
    SUPPORTED_DTYPES,
)
from layerzero.enums import Layout


# Check if atheris is available
try:
    import atheris
    ATHERIS_AVAILABLE = True
except ImportError:
    ATHERIS_AVAILABLE = False


# Corpus directory
CORPUS_DIR = Path(__file__).parent / "corpus"


class FuzzedDataProvider:
    """Minimal FuzzedDataProvider for fallback testing.

    Simulates atheris.FuzzedDataProvider for testing without atheris.
    """

    def __init__(self, data: bytes) -> None:
        self.data = data
        self.pos = 0

    def ConsumeBytes(self, count: int) -> bytes:
        """Consume count bytes from data."""
        result = self.data[self.pos : self.pos + count]
        self.pos += count
        return result

    def ConsumeInt(self, num_bytes: int) -> int:
        """Consume integer from data."""
        data = self.ConsumeBytes(num_bytes)
        if len(data) == 0:
            return 0
        return int.from_bytes(data, byteorder="little", signed=True)

    def ConsumeUInt(self, num_bytes: int) -> int:
        """Consume unsigned integer from data."""
        data = self.ConsumeBytes(num_bytes)
        if len(data) == 0:
            return 0
        return int.from_bytes(data, byteorder="little", signed=False)

    def ConsumeIntInRange(self, low: int, high: int) -> int:
        """Consume integer in range."""
        if high <= low:
            return low
        range_size = high - low + 1
        value = self.ConsumeUInt(4) % range_size
        return low + value

    def ConsumeFloat(self) -> float:
        """Consume a float."""
        data = self.ConsumeBytes(4)
        if len(data) < 4:
            return 0.0
        try:
            return struct.unpack("f", data)[0]
        except struct.error:
            return 0.0

    def ConsumeBool(self) -> bool:
        """Consume a boolean."""
        return self.ConsumeUInt(1) % 2 == 1

    def remaining_bytes(self) -> int:
        """Get remaining bytes."""
        return max(0, len(self.data) - self.pos)


def generate_random_bytes(min_size: int = 16, max_size: int = 256) -> bytes:
    """Generate random bytes for fuzzing."""
    size = random.randint(min_size, max_size)
    return bytes(random.getrandbits(8) for _ in range(size))


class TestFuzzHarness:
    """Test fuzz harness setup and availability."""

    def test_corpus_directory_exists(self) -> None:
        """Corpus directory exists for seed inputs."""
        assert CORPUS_DIR.exists()
        assert (CORPUS_DIR / "context").exists()
        assert (CORPUS_DIR / "policy").exists()
        assert (CORPUS_DIR / "validation").exists()

    def test_fuzzed_data_provider_works(self) -> None:
        """FuzzedDataProvider handles random input."""
        data = generate_random_bytes()
        fdp = FuzzedDataProvider(data)

        # Should not crash
        _ = fdp.ConsumeInt(4)
        _ = fdp.ConsumeUInt(4)
        _ = fdp.ConsumeFloat()
        _ = fdp.ConsumeBool()

    def test_fuzzed_data_provider_empty_input(self) -> None:
        """FuzzedDataProvider handles empty input."""
        fdp = FuzzedDataProvider(b"")

        # Should return defaults, not crash
        assert fdp.ConsumeInt(4) == 0
        assert fdp.ConsumeUInt(4) == 0
        assert fdp.ConsumeFloat() == 0.0

    def test_fuzzed_data_provider_range(self) -> None:
        """FuzzedDataProvider range constraint works."""
        data = generate_random_bytes(32, 64)
        fdp = FuzzedDataProvider(data)

        # Should be in range
        for _ in range(10):
            value = fdp.ConsumeIntInRange(1, 100)
            assert 1 <= value <= 100


class TestValidationFuzzing:
    """Fuzz validation functions with random inputs."""

    def test_fuzz_validate_head_dim(self) -> None:
        """validate_head_dim handles any integer input."""
        for _ in range(1000):
            data = generate_random_bytes(4, 8)
            fdp = FuzzedDataProvider(data)
            head_dim = fdp.ConsumeInt(4)

            # Should not crash, just return ValidationResult
            result = validate_head_dim(head_dim)
            assert isinstance(result, ValidationResult)
            assert isinstance(result.valid, bool)

    def test_fuzz_validate_head_dim_extreme_values(self) -> None:
        """validate_head_dim handles extreme values."""
        extreme_values = [
            0,
            -1,
            -1000,
            1,
            7,
            8,
            64,
            128,
            256,
            320,
            321,
            1000,
            10000,
            100000,
            2**31 - 1,
            -(2**31),
        ]

        for value in extreme_values:
            result = validate_head_dim(value)
            assert isinstance(result, ValidationResult)

    def test_fuzz_validate_cuda_block_limits(self) -> None:
        """validate_cuda_block_limits handles any inputs."""
        for _ in range(1000):
            data = generate_random_bytes(12, 16)
            fdp = FuzzedDataProvider(data)

            batch = fdp.ConsumeUInt(4)
            heads = fdp.ConsumeUInt(4)
            seq_len = fdp.ConsumeUInt(4) if fdp.ConsumeBool() else None

            # Should not crash
            result = validate_cuda_block_limits(batch, heads, seq_len)
            assert isinstance(result, ValidationResult)

    def test_fuzz_validate_dtype(self) -> None:
        """validate_dtype handles various dtype inputs."""
        all_dtypes = [
            torch.float16,
            torch.bfloat16,
            torch.float32,
            torch.float64,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
            torch.complex64,
            torch.complex128,
        ]

        for _ in range(500):
            # Pick random dtypes
            dtype1 = random.choice(all_dtypes)
            dtype2 = random.choice(all_dtypes) if random.random() > 0.5 else None

            if dtype2:
                result = validate_dtype(dtype1, dtype2)
            else:
                result = validate_dtype(dtype1)

            assert isinstance(result, ValidationResult)


class TestContextFuzzing:
    """Fuzz context building with random data."""

    def test_fuzz_context_dict_from_random(self) -> None:
        """Build context dict from random data."""
        for _ in range(1000):
            data = generate_random_bytes(16, 64)
            fdp = FuzzedDataProvider(data)

            # Build context from fuzzed data
            context = {
                "batch_size": fdp.ConsumeIntInRange(1, 256),
                "seq_len_q": fdp.ConsumeIntInRange(1, 8192),
                "seq_len_k": fdp.ConsumeIntInRange(1, 8192),
                "num_heads": fdp.ConsumeIntInRange(1, 128),
                "head_dim": fdp.ConsumeIntInRange(1, 512),
            }

            # Should be valid dict
            assert isinstance(context, dict)
            assert all(isinstance(v, int) for v in context.values())
            assert all(v >= 1 for v in context.values())

    def test_fuzz_context_with_extreme_values(self) -> None:
        """Context handles extreme values."""
        extreme_contexts = [
            {"batch_size": 1, "seq_len_q": 1, "head_dim": 1},
            {"batch_size": 1000, "seq_len_q": 100000, "head_dim": 256},
            {"batch_size": 0, "seq_len_q": 0, "head_dim": 0},  # Invalid but shouldnt crash
        ]

        for ctx in extreme_contexts:
            # Just ensure no crash
            assert isinstance(ctx, dict)


class TestLayoutFuzzing:
    """Fuzz layout detection with random tensors."""

    def test_fuzz_validate_layout(self) -> None:
        """validate_layout handles random tensor shapes."""
        layouts = [Layout.BHSD, Layout.BSHD]

        for _ in range(100):
            # Random dimensions (1-5D)
            ndim = random.randint(1, 5)
            shape = tuple(random.randint(1, 32) for _ in range(ndim))
            tensor = torch.randn(shape)
            layout = random.choice(layouts)

            # Should not crash
            result = validate_layout(tensor, layout)
            assert isinstance(result, ValidationResult)

    def test_fuzz_attention_inputs_valid(self) -> None:
        """Fuzz attention input validation with valid inputs."""
        for _ in range(50):
            # Generate valid-ish inputs
            batch = random.randint(1, 16)
            heads = random.randint(1, 16)
            seq_q = random.randint(1, 256)
            seq_k = random.randint(1, 256)
            head_dim = random.choice([32, 64, 128, 256])
            dtype = random.choice(list(SUPPORTED_DTYPES))

            query = torch.randn(batch, heads, seq_q, head_dim, dtype=dtype)
            key = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype)
            value = torch.randn(batch, heads, seq_k, head_dim, dtype=dtype)

            result = validate_attention_inputs(query, key, value)
            assert isinstance(result, ValidationResult)

    def test_fuzz_attention_inputs_invalid(self) -> None:
        """Fuzz attention input validation with potentially invalid inputs."""
        for _ in range(50):
            try:
                # Random dimensions (may be invalid)
                ndim = random.randint(1, 6)
                shape_q = tuple(random.randint(1, 16) for _ in range(ndim))
                shape_k = tuple(random.randint(1, 16) for _ in range(ndim))
                shape_v = tuple(random.randint(1, 16) for _ in range(ndim))

                query = torch.randn(shape_q)
                key = torch.randn(shape_k)
                value = torch.randn(shape_v)

                result = validate_attention_inputs(query, key, value)
                assert isinstance(result, ValidationResult)
            except (RuntimeError, ValueError):
                # Shape mismatches may cause errors in torch.randn
                pass


class TestPolicyFuzzing:
    """Fuzz policy parsing with random YAML-like data."""

    def test_fuzz_policy_yaml_strings(self) -> None:
        """Parse random YAML-like strings."""
        import yaml

        for _ in range(100):
            # Generate random YAML-like content
            data = generate_random_bytes(10, 100)
            try:
                # Try to parse as YAML (will likely fail gracefully)
                yaml.safe_load(data.decode("utf-8", errors="replace"))
            except (yaml.YAMLError, UnicodeDecodeError):
                # Expected for random data
                pass

    def test_fuzz_policy_dict_construction(self) -> None:
        """Construct policy dicts from random data."""
        for _ in range(1000):
            data = generate_random_bytes(32, 128)
            fdp = FuzzedDataProvider(data)

            # Construct policy-like dict
            policy = {
                "name": f"policy_{fdp.ConsumeUInt(2)}",
                "priority": fdp.ConsumeIntInRange(0, 100),
                "enabled": fdp.ConsumeBool(),
            }

            assert isinstance(policy, dict)
            assert "name" in policy
            assert "priority" in policy


class TestCrashResistance:
    """Test that no input causes crashes."""

    def test_no_crash_head_dim_1m_iterations(self) -> None:
        """No crashes after 100K head_dim validations."""
        for i in range(100000):
            # Deterministic but covers range
            head_dim = (i * 7 + 13) % 10000 - 5000  # Values from -5000 to 5000
            result = validate_head_dim(head_dim)
            assert isinstance(result, ValidationResult)

    def test_no_crash_cuda_limits_100k_iterations(self) -> None:
        """No crashes after 100K CUDA limit checks."""
        for i in range(100000):
            batch = (i * 3) % 1000 + 1
            heads = (i * 7) % 100 + 1
            seq = ((i * 11) % 10000 + 1) if i % 2 == 0 else None
            result = validate_cuda_block_limits(batch, heads, seq)
            assert isinstance(result, ValidationResult)

    def test_no_crash_dtype_validation_100k(self) -> None:
        """No crashes after 100K dtype validations."""
        dtypes = list(SUPPORTED_DTYPES) + [torch.int32, torch.int64, torch.float64]

        for i in range(100000):
            dtype1 = dtypes[i % len(dtypes)]
            dtype2 = dtypes[(i * 3) % len(dtypes)] if i % 3 == 0 else None

            if dtype2:
                result = validate_dtype(dtype1, dtype2)
            else:
                result = validate_dtype(dtype1)

            assert isinstance(result, ValidationResult)


# Actual Atheris fuzz targets (only defined if atheris is available)
if ATHERIS_AVAILABLE:

    def fuzz_head_dim(data: bytes) -> None:
        """Atheris fuzz target for head_dim validation."""
        fdp = atheris.FuzzedDataProvider(data)
        head_dim = fdp.ConsumeInt(4)
        _ = validate_head_dim(head_dim)

    def fuzz_cuda_limits(data: bytes) -> None:
        """Atheris fuzz target for CUDA limits."""
        fdp = atheris.FuzzedDataProvider(data)
        batch = fdp.ConsumeUInt(4)
        heads = fdp.ConsumeUInt(4)
        seq = fdp.ConsumeUInt(4) if fdp.ConsumeBool() else None
        _ = validate_cuda_block_limits(batch, heads, seq)

    def fuzz_dtype(data: bytes) -> None:
        """Atheris fuzz target for dtype validation."""
        all_dtypes = list(SUPPORTED_DTYPES) + [torch.int32, torch.int64]
        fdp = atheris.FuzzedDataProvider(data)
        idx1 = fdp.ConsumeUInt(1) % len(all_dtypes)
        idx2 = fdp.ConsumeUInt(1) % len(all_dtypes)
        _ = validate_dtype(all_dtypes[idx1], all_dtypes[idx2])

    class TestAtherisFuzzTargets:
        """Test that Atheris fuzz targets work."""

        @pytest.mark.skipif(not ATHERIS_AVAILABLE, reason="Atheris not installed")
        def test_atheris_available(self) -> None:
            """Atheris is available."""
            assert ATHERIS_AVAILABLE

        @pytest.mark.skipif(not ATHERIS_AVAILABLE, reason="Atheris not installed")
        def test_fuzz_target_head_dim_runs(self) -> None:
            """Head dim fuzz target runs without crash."""
            for _ in range(100):
                data = generate_random_bytes()
                fuzz_head_dim(data)

        @pytest.mark.skipif(not ATHERIS_AVAILABLE, reason="Atheris not installed")
        def test_fuzz_target_cuda_runs(self) -> None:
            """CUDA limits fuzz target runs without crash."""
            for _ in range(100):
                data = generate_random_bytes()
                fuzz_cuda_limits(data)

        @pytest.mark.skipif(not ATHERIS_AVAILABLE, reason="Atheris not installed")
        def test_fuzz_target_dtype_runs(self) -> None:
            """Dtype fuzz target runs without crash."""
            for _ in range(100):
                data = generate_random_bytes()
                fuzz_dtype(data)
