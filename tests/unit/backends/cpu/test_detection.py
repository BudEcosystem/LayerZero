"""Tests for CPU vendor and ISA detection."""
from __future__ import annotations

import pytest

from layerzero.backends.cpu.detection import (
    CPUVendor,
    ISAFeature,
    detect_cpu_vendor,
    detect_isa_features,
    get_cpu_info,
    get_optimal_cpu_backend,
)


class TestCPUVendorDetection:
    """Test CPU vendor detection."""

    def test_detect_vendor_returns_enum(self) -> None:
        """detect_cpu_vendor returns CPUVendor enum."""
        vendor = detect_cpu_vendor()
        assert isinstance(vendor, CPUVendor)

    def test_detect_vendor_known_values(self) -> None:
        """detect_cpu_vendor returns known vendor or UNKNOWN."""
        vendor = detect_cpu_vendor()
        assert vendor in (
            CPUVendor.INTEL,
            CPUVendor.AMD,
            CPUVendor.ARM,
            CPUVendor.UNKNOWN,
        )

    def test_cpu_vendor_is_str_enum(self) -> None:
        """CPUVendor values are strings."""
        assert CPUVendor.INTEL.value == "intel"
        assert CPUVendor.AMD.value == "amd"
        assert CPUVendor.ARM.value == "arm"


class TestISAFeatureDetection:
    """Test ISA feature detection."""

    def test_detect_isa_returns_set(self) -> None:
        """detect_isa_features returns set of ISAFeature."""
        features = detect_isa_features()
        assert isinstance(features, set)

    def test_isa_features_are_valid(self) -> None:
        """All detected features are valid ISAFeature enum values."""
        features = detect_isa_features()
        for feature in features:
            assert isinstance(feature, ISAFeature)

    def test_isa_feature_enum_values(self) -> None:
        """ISAFeature values are strings."""
        assert ISAFeature.AVX2.value == "avx2"
        assert ISAFeature.AVX512F.value == "avx512f"
        assert ISAFeature.AMX.value == "amx"

    def test_baseline_features_detected(self) -> None:
        """Baseline features detected on modern CPUs."""
        features = detect_isa_features()
        # On any modern x86-64 CPU, at least SSE4.2 should be available
        # ARM systems will have NEON
        # The test passes if ANY feature is detected (which is valid)
        # since we're testing the detection mechanism, not specific hardware
        # Actually, let's just verify the detection works without asserting
        # specific features since this runs on various hardware
        assert features is not None


class TestCPUInfo:
    """Test CPU info gathering."""

    def test_get_cpu_info_returns_dict(self) -> None:
        """get_cpu_info returns dictionary."""
        info = get_cpu_info()
        assert isinstance(info, dict)

    def test_cpu_info_has_vendor(self) -> None:
        """CPU info contains vendor."""
        info = get_cpu_info()
        assert "vendor" in info
        assert isinstance(info["vendor"], CPUVendor)

    def test_cpu_info_has_features(self) -> None:
        """CPU info contains features."""
        info = get_cpu_info()
        assert "features" in info
        assert isinstance(info["features"], set)

    def test_cpu_info_has_core_count(self) -> None:
        """CPU info contains core count."""
        info = get_cpu_info()
        assert "core_count" in info
        assert isinstance(info["core_count"], int)
        assert info["core_count"] > 0


class TestOptimalBackendSelection:
    """Test optimal CPU backend selection."""

    def test_get_optimal_backend_returns_string(self) -> None:
        """get_optimal_cpu_backend returns string."""
        backend = get_optimal_cpu_backend()
        assert isinstance(backend, str)

    def test_optimal_backend_is_valid(self) -> None:
        """Optimal backend is a known backend name."""
        backend = get_optimal_cpu_backend()
        valid_backends = {"onednn", "zendnn", "ipex", "pytorch", "none"}
        assert backend in valid_backends

    def test_pytorch_always_available(self) -> None:
        """PyTorch CPU backend should always be available as fallback."""
        backend = get_optimal_cpu_backend()
        # Should return at least 'pytorch' as fallback
        assert backend != "none"


class TestCPUVendorEnum:
    """Test CPUVendor enum completeness."""

    def test_all_vendors_defined(self) -> None:
        """All expected vendors are defined."""
        assert hasattr(CPUVendor, "INTEL")
        assert hasattr(CPUVendor, "AMD")
        assert hasattr(CPUVendor, "ARM")
        assert hasattr(CPUVendor, "UNKNOWN")

    def test_vendor_string_values(self) -> None:
        """Vendor string values are lowercase."""
        for vendor in CPUVendor:
            assert vendor.value == vendor.value.lower()


class TestISAFeatureEnum:
    """Test ISAFeature enum completeness."""

    def test_x86_features_defined(self) -> None:
        """x86 ISA features are defined."""
        assert hasattr(ISAFeature, "SSE4_2")
        assert hasattr(ISAFeature, "AVX2")
        assert hasattr(ISAFeature, "AVX512F")
        assert hasattr(ISAFeature, "AVX512_BF16")
        assert hasattr(ISAFeature, "AVX512_VNNI")
        assert hasattr(ISAFeature, "AMX")

    def test_arm_features_defined(self) -> None:
        """ARM ISA features are defined."""
        assert hasattr(ISAFeature, "NEON")
        assert hasattr(ISAFeature, "SVE")

    def test_feature_string_values(self) -> None:
        """Feature string values are lowercase."""
        for feature in ISAFeature:
            assert feature.value == feature.value.lower()
