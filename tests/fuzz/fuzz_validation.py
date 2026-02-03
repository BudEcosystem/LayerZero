#!/usr/bin/env python3
"""Standalone fuzz target for validation functions.

Run with:
    python -m atheris tests/fuzz/fuzz_validation.py -max_len=256 -runs=100000
"""
from __future__ import annotations

import sys

# Try to import atheris
try:
    import atheris
except ImportError:
    print("Atheris not installed. Install with: pip install atheris")
    sys.exit(1)

# Setup instrumentation before importing targets
with atheris.instrument_imports():
    import torch
    from layerzero.core.validation import (
        validate_head_dim,
        validate_dtype,
        validate_cuda_block_limits,
        ValidationResult,
        SUPPORTED_DTYPES,
    )


ALL_DTYPES = list(SUPPORTED_DTYPES) + [
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.float64,
]


def fuzz_validation(data: bytes) -> None:
    """Fuzz target for all validation functions."""
    fdp = atheris.FuzzedDataProvider(data)

    # Fuzz head_dim validation
    head_dim = fdp.ConsumeInt(4)
    result = validate_head_dim(head_dim)
    assert isinstance(result, ValidationResult)

    # Fuzz CUDA limits
    batch = fdp.ConsumeUInt(4)
    heads = fdp.ConsumeUInt(4)
    seq_len = fdp.ConsumeUInt(4) if fdp.ConsumeBool() else None
    result = validate_cuda_block_limits(batch, heads, seq_len)
    assert isinstance(result, ValidationResult)

    # Fuzz dtype validation
    if fdp.remaining_bytes() >= 2:
        idx1 = fdp.ConsumeUInt(1) % len(ALL_DTYPES)
        idx2 = fdp.ConsumeUInt(1) % len(ALL_DTYPES)
        result = validate_dtype(ALL_DTYPES[idx1], ALL_DTYPES[idx2])
        assert isinstance(result, ValidationResult)


def main() -> None:
    """Run the fuzzer."""
    atheris.Setup(sys.argv, fuzz_validation)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
