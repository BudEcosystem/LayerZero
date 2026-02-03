#!/usr/bin/env python3
"""Standalone fuzz target for context building.

Run with:
    python -m atheris tests/fuzz/fuzz_context.py -max_len=256 -runs=100000
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
    from layerzero.core.validation import (
        validate_head_dim,
        validate_cuda_block_limits,
        ValidationResult,
    )


def fuzz_context(data: bytes) -> None:
    """Fuzz target for context building."""
    fdp = atheris.FuzzedDataProvider(data)

    # Build context-like dict from fuzzed data
    context = {
        "batch_size": fdp.ConsumeIntInRange(0, 10000),
        "seq_len_q": fdp.ConsumeIntInRange(0, 100000),
        "seq_len_k": fdp.ConsumeIntInRange(0, 100000),
        "num_heads": fdp.ConsumeIntInRange(0, 1000),
        "head_dim": fdp.ConsumeInt(4),
    }

    # Validate components
    head_result = validate_head_dim(context["head_dim"])
    assert isinstance(head_result, ValidationResult)

    # Validate CUDA limits if values are reasonable
    if context["batch_size"] > 0 and context["num_heads"] > 0:
        cuda_result = validate_cuda_block_limits(
            context["batch_size"],
            context["num_heads"],
            context["seq_len_q"] if context["seq_len_q"] > 0 else None,
        )
        assert isinstance(cuda_result, ValidationResult)


def main() -> None:
    """Run the fuzzer."""
    atheris.Setup(sys.argv, fuzz_context)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
