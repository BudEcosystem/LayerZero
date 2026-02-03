"""
Test suite for DeviceSpec dataclass.

Tests device capability detection and serialization.
Following TDD methodology - tests define expected behavior.
"""
import json
import pytest


class TestDeviceSpecCreation:
    """Test DeviceSpec construction."""

    def test_device_spec_required_fields(self):
        """DeviceSpec must have platform, device_index, device_name."""
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import Platform
        from layerzero.device import GPUGeneration

        spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="NVIDIA RTX 3080",
            sm_version=(8, 6),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=10 * 1024**3,
            available_memory_bytes=8 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.4",
            driver_version="550.54",
        )

        assert spec.platform == Platform.CUDA
        assert spec.device_index == 0
        assert spec.sm_version == (8, 6)
        assert spec.gpu_generation == GPUGeneration.AMPERE

    def test_device_spec_cpu_creation(self):
        """DeviceSpec can represent CPU with None GPU fields."""
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import Platform
        from layerzero.device import GPUGeneration

        spec = DeviceSpec(
            platform=Platform.CPU,
            device_index=0,
            device_name="Intel Xeon",
            sm_version=None,
            gpu_generation=GPUGeneration.UNKNOWN,
            tensor_core_gen=0,
            total_memory_bytes=64 * 1024**3,
            available_memory_bytes=32 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=0,
            cuda_version=None,
            driver_version=None,
        )

        assert spec.platform == Platform.CPU
        assert spec.sm_version is None
        assert spec.gpu_generation == GPUGeneration.UNKNOWN

    def test_device_spec_is_frozen(self):
        """DeviceSpec is immutable (frozen)."""
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import Platform
        from layerzero.device import GPUGeneration

        spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 0),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=10 * 1024**3,
            available_memory_bytes=8 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.4",
            driver_version="550.0",
        )

        with pytest.raises(AttributeError):
            spec.device_name = "Modified"  # type: ignore


class TestDeviceSpecDetection:
    """Test DeviceSpec.detect() method."""

    def test_detect_returns_device_spec(self):
        """DeviceSpec.detect() returns a DeviceSpec instance."""
        from layerzero.models.device_spec import DeviceSpec

        spec = DeviceSpec.detect()
        assert isinstance(spec, DeviceSpec)

    def test_detect_cpu_fallback(self):
        """DeviceSpec.detect() returns CPU spec when no GPU."""
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import Platform

        # Force CPU detection
        spec = DeviceSpec.detect(device="cpu")
        assert spec.platform == Platform.CPU

    def test_cpu_class_method(self):
        """DeviceSpec.cpu() returns CPU device spec."""
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import Platform

        spec = DeviceSpec.cpu()
        assert spec.platform == Platform.CPU
        assert spec.sm_version is None
        assert spec.tensor_core_gen == 0


class TestDeviceSpecSerialization:
    """Test DeviceSpec JSON serialization."""

    def test_device_spec_to_dict(self):
        """DeviceSpec can be serialized to dict."""
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import Platform
        from layerzero.device import GPUGeneration

        spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(8, 6),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=10 * 1024**3,
            available_memory_bytes=8 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.4",
            driver_version="550.0",
        )

        d = spec.to_dict()
        assert d["platform"] == "cuda"
        assert d["sm_version"] == [8, 6]
        assert d["gpu_generation"] == "ampere"

    def test_device_spec_json_roundtrip(self):
        """DeviceSpec serialize/deserialize preserves all fields."""
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import Platform
        from layerzero.device import GPUGeneration

        original = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="Test GPU",
            sm_version=(9, 0),
            gpu_generation=GPUGeneration.HOPPER,
            tensor_core_gen=4,
            total_memory_bytes=80 * 1024**3,
            available_memory_bytes=60 * 1024**3,
            supports_bf16=True,
            supports_fp8=True,
            supports_fp4=False,
            supports_tma=True,
            max_shared_memory_kb=228,
            cuda_version="12.4",
            driver_version="550.0",
        )

        json_str = json.dumps(original.to_dict())
        d = json.loads(json_str)
        restored = DeviceSpec.from_dict(d)

        assert restored.platform == original.platform
        assert restored.sm_version == original.sm_version
        assert restored.gpu_generation == original.gpu_generation
        assert restored.supports_tma == original.supports_tma


class TestDeviceSpecCapabilities:
    """Test DeviceSpec capability fields."""

    def test_ampere_capabilities(self):
        """Ampere has bf16 but not fp8/fp4/tma."""
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import Platform
        from layerzero.device import GPUGeneration

        spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="A100",
            sm_version=(8, 0),
            gpu_generation=GPUGeneration.AMPERE,
            tensor_core_gen=3,
            total_memory_bytes=40 * 1024**3,
            available_memory_bytes=35 * 1024**3,
            supports_bf16=True,
            supports_fp8=False,
            supports_fp4=False,
            supports_tma=False,
            max_shared_memory_kb=164,
            cuda_version="12.0",
            driver_version="525.0",
        )

        assert spec.supports_bf16 is True
        assert spec.supports_fp8 is False
        assert spec.supports_tma is False

    def test_hopper_capabilities(self):
        """Hopper has bf16, fp8, tma but not fp4."""
        from layerzero.models.device_spec import DeviceSpec
        from layerzero.enums import Platform
        from layerzero.device import GPUGeneration

        spec = DeviceSpec(
            platform=Platform.CUDA,
            device_index=0,
            device_name="H100",
            sm_version=(9, 0),
            gpu_generation=GPUGeneration.HOPPER,
            tensor_core_gen=4,
            total_memory_bytes=80 * 1024**3,
            available_memory_bytes=70 * 1024**3,
            supports_bf16=True,
            supports_fp8=True,
            supports_fp4=False,
            supports_tma=True,
            max_shared_memory_kb=228,
            cuda_version="12.4",
            driver_version="550.0",
        )

        assert spec.supports_bf16 is True
        assert spec.supports_fp8 is True
        assert spec.supports_tma is True
        assert spec.supports_fp4 is False
