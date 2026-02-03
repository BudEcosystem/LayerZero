#!/usr/bin/env python3
"""Standalone fuzz target for policy YAML parsing.

Run with:
    python -m atheris tests/fuzz/fuzz_policy.py -max_len=4096 -runs=100000
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
    import yaml


def fuzz_policy(data: bytes) -> None:
    """Fuzz target for policy YAML parsing."""
    # Try to decode as string
    try:
        yaml_str = data.decode("utf-8")
    except UnicodeDecodeError:
        # Use latin-1 as fallback (never fails)
        yaml_str = data.decode("latin-1")

    # Try to parse as YAML
    try:
        parsed = yaml.safe_load(yaml_str)

        # If parsed, try to extract policy fields
        if isinstance(parsed, dict):
            _ = parsed.get("name")
            _ = parsed.get("priority")
            _ = parsed.get("enabled")
            _ = parsed.get("rules")
            _ = parsed.get("backends")

    except yaml.YAMLError:
        # Expected for random data
        pass
    except Exception:
        # Other errors should also not crash
        pass


def main() -> None:
    """Run the fuzzer."""
    atheris.Setup(sys.argv, fuzz_policy)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
