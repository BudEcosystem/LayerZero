"""
Test suite for Policy and PolicyLoader.

Tests policy parsing and rule compilation.
Following TDD methodology - tests define expected behavior.
"""
import os
import pytest
import tempfile
from pathlib import Path


class TestPolicyCreation:
    """Test Policy construction."""

    def test_policy_creation(self):
        """Policy can be created."""
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )
        assert policy.version == "1.0"

    def test_policy_is_empty(self):
        """Policy.is_empty returns True for empty policy."""
        from layerzero.policy.policy import Policy

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )
        assert policy.is_empty is True

    def test_policy_not_empty_with_locks(self):
        """Policy.is_empty returns False when has rules."""
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy = Policy(
            version="1.0",
            locks=(Rule(RuleType.LOCK, "attention.causal", (), "flash_attn.v3"),),
            allows=(),
            denies=(),
            boosts=(),
        )
        assert policy.is_empty is False


class TestPolicyHash:
    """Test Policy hash computation."""

    def test_policy_hash_is_string(self):
        """Policy hash is a string."""
        from layerzero.policy.policy import Policy

        policy = Policy(
            version="1.0",
            locks=(),
            allows=(),
            denies=(),
            boosts=(),
        )
        assert isinstance(policy.policy_hash, str)
        assert len(policy.policy_hash) > 0

    def test_policy_hash_deterministic(self):
        """Same policy produces same hash."""
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy1 = Policy(
            version="1.0",
            locks=(Rule(RuleType.LOCK, "op", (), "kernel"),),
            allows=(),
            denies=(),
            boosts=(),
        )
        policy2 = Policy(
            version="1.0",
            locks=(Rule(RuleType.LOCK, "op", (), "kernel"),),
            allows=(),
            denies=(),
            boosts=(),
        )
        assert policy1.policy_hash == policy2.policy_hash

    def test_policy_hash_differs_for_different_rules(self):
        """Different policies produce different hashes."""
        from layerzero.policy.policy import Policy
        from layerzero.policy.rule import Rule, RuleType

        policy1 = Policy(
            version="1.0",
            locks=(Rule(RuleType.LOCK, "op1", (), "kernel1"),),
            allows=(),
            denies=(),
            boosts=(),
        )
        policy2 = Policy(
            version="1.0",
            locks=(Rule(RuleType.LOCK, "op2", (), "kernel2"),),
            allows=(),
            denies=(),
            boosts=(),
        )
        assert policy1.policy_hash != policy2.policy_hash


class TestPolicyLoaderYAML:
    """Test PolicyLoader YAML parsing."""

    def test_loader_empty_yaml(self):
        """Loader handles empty YAML."""
        from layerzero.policy.loader import PolicyLoader

        loader = PolicyLoader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("version: '1.0'\n")
            f.flush()
            policy = loader.load(yaml_path=Path(f.name))

        os.unlink(f.name)
        assert policy.is_empty

    def test_loader_lock_rule(self):
        """Loader parses lock rules."""
        from layerzero.policy.loader import PolicyLoader

        yaml_content = """
version: '1.0'
locks:
  - operation: attention.causal
    kernel: flash_attn.v3.causal
"""
        loader = PolicyLoader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            policy = loader.load(yaml_path=Path(f.name))

        os.unlink(f.name)
        assert len(policy.locks) == 1
        assert policy.locks[0].target == "attention.causal"
        assert policy.locks[0].value == "flash_attn.v3.causal"

    def test_loader_deny_rule(self):
        """Loader parses deny rules."""
        from layerzero.policy.loader import PolicyLoader

        yaml_content = """
version: '1.0'
deny:
  - kernel: torch.sdpa
"""
        loader = PolicyLoader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            policy = loader.load(yaml_path=Path(f.name))

        os.unlink(f.name)
        assert len(policy.denies) == 1
        assert policy.denies[0].target == "torch.sdpa"

    def test_loader_allow_rule(self):
        """Loader parses allow rules."""
        from layerzero.policy.loader import PolicyLoader

        yaml_content = """
version: '1.0'
allow:
  - backend: flash_attn
  - backend: xformers
"""
        loader = PolicyLoader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            policy = loader.load(yaml_path=Path(f.name))

        os.unlink(f.name)
        assert len(policy.allows) == 2

    def test_loader_boost_rule(self):
        """Loader parses boost rules."""
        from layerzero.policy.loader import PolicyLoader

        yaml_content = """
version: '1.0'
boosts:
  - kernel: flash_attn.*
    priority_add: 20
"""
        loader = PolicyLoader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            policy = loader.load(yaml_path=Path(f.name))

        os.unlink(f.name)
        assert len(policy.boosts) == 1
        assert policy.boosts[0].value == 20

    def test_loader_conditional_rule(self):
        """Loader parses rules with conditions."""
        from layerzero.policy.loader import PolicyLoader
        from layerzero.policy.rule import ConditionOp

        yaml_content = """
version: '1.0'
locks:
  - operation: attention.causal
    kernel: flash_attn.v3.causal
    when:
      head_dim: ">=64"
      seq_len_q: ">=128"
"""
        loader = PolicyLoader()
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            f.flush()
            policy = loader.load(yaml_path=Path(f.name))

        os.unlink(f.name)
        assert len(policy.locks) == 1
        assert len(policy.locks[0].conditions) == 2


class TestPolicyLoaderEnv:
    """Test PolicyLoader environment variable parsing."""

    def test_loader_env_lock(self):
        """Loader parses lock from environment."""
        from layerzero.policy.loader import PolicyLoader

        os.environ["LAYERZERO_LOCK_attention_causal"] = "flash_attn.v3.causal"
        try:
            loader = PolicyLoader()
            policy = loader.load()
            assert len(policy.locks) == 1
            assert policy.locks[0].target == "attention.causal"
        finally:
            del os.environ["LAYERZERO_LOCK_attention_causal"]

    def test_loader_env_deny(self):
        """Loader parses deny from environment."""
        from layerzero.policy.loader import PolicyLoader

        os.environ["LAYERZERO_DENY"] = "torch.sdpa,triton.*"
        try:
            loader = PolicyLoader()
            policy = loader.load()
            assert len(policy.denies) == 2
        finally:
            del os.environ["LAYERZERO_DENY"]

    def test_loader_env_allow(self):
        """Loader parses allow from environment."""
        from layerzero.policy.loader import PolicyLoader

        os.environ["LAYERZERO_ALLOW"] = "flash_attn,xformers"
        try:
            loader = PolicyLoader()
            policy = loader.load()
            assert len(policy.allows) == 2
        finally:
            del os.environ["LAYERZERO_ALLOW"]

    def test_loader_env_boost(self):
        """Loader parses boost from environment."""
        from layerzero.policy.loader import PolicyLoader

        os.environ["LAYERZERO_BOOST_flash_attn"] = "+20"
        try:
            loader = PolicyLoader()
            policy = loader.load()
            assert len(policy.boosts) == 1
            assert policy.boosts[0].value == 20
        finally:
            del os.environ["LAYERZERO_BOOST_flash_attn"]


class TestPolicyLoaderPrecedence:
    """Test PolicyLoader source precedence."""

    def test_env_overrides_yaml(self):
        """Environment variables override YAML."""
        from layerzero.policy.loader import PolicyLoader

        yaml_content = """
version: '1.0'
locks:
  - operation: attention.causal
    kernel: xformers.causal
"""
        # Env should override
        os.environ["LAYERZERO_LOCK_attention_causal"] = "flash_attn.v3.causal"
        try:
            loader = PolicyLoader()
            with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
                f.write(yaml_content)
                f.flush()
                policy = loader.load(yaml_path=Path(f.name))

            os.unlink(f.name)
            # Env lock should take precedence
            assert any(
                r.value == "flash_attn.v3.causal"
                for r in policy.locks
            )
        finally:
            del os.environ["LAYERZERO_LOCK_attention_causal"]


class TestPolicyDefault:
    """Test default policy behavior."""

    def test_default_policy_empty(self):
        """Default policy with no sources is empty."""
        from layerzero.policy.loader import PolicyLoader

        loader = PolicyLoader()
        policy = loader.load()
        assert policy.is_empty
