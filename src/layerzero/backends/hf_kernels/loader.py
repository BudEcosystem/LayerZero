"""
LayerZero HuggingFace Kernel Hub Loader

Load pre-compiled kernels from HuggingFace Hub.
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from layerzero.backends.hf_kernels.version import is_hf_kernels_available

logger = logging.getLogger(__name__)


@dataclass
class LoadedKernel:
    """Represents a loaded kernel from HuggingFace Hub.

    Attributes:
        name: Kernel name
        version: Kernel version string
        path: Local path to the loaded kernel library
        sha256: SHA256 hash of the kernel file
        metadata: Optional metadata dictionary
    """

    name: str
    version: str
    path: Path
    sha256: str
    metadata: dict[str, Any] | None = None


class HFKernelLoader:
    """Load kernels from HuggingFace Hub.

    Provides functionality to download and load pre-compiled CUDA kernels
    from the HuggingFace Hub.

    Example:
        ```python
        loader = HFKernelLoader()
        kernel = loader.load("flash_attn", version="2.6.0")
        if kernel:
            print(f"Loaded {kernel.name} v{kernel.version}")
        ```
    """

    def __init__(
        self,
        cache_dir: str | Path | None = None,
    ) -> None:
        """Initialize the kernel loader.

        Args:
            cache_dir: Directory for caching downloaded kernels.
                      Defaults to ~/.cache/layerzero/hf_kernels/
        """
        if cache_dir is None:
            cache_dir = Path.home() / ".cache" / "layerzero" / "hf_kernels"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        kernel_name: str,
        version: str | None = None,
    ) -> LoadedKernel | None:
        """Load a kernel by name from HuggingFace Hub.

        Args:
            kernel_name: Name of the kernel to load
            version: Optional version string. If None, loads latest.

        Returns:
            LoadedKernel if successful, None otherwise.
        """
        if not is_hf_kernels_available():
            logger.warning("HuggingFace Hub not available")
            return None

        try:
            from huggingface_hub import hf_hub_download, HfApi

            # Try to find kernel in Hub
            api = HfApi()

            # Construct repo ID for kernel
            repo_id = f"kernels/{kernel_name}"

            try:
                # Get repo info to verify it exists
                api.repo_info(repo_id, repo_type="model")
            except Exception as e:
                logger.debug(f"Kernel {kernel_name} not found: {e}")
                return None

            # Determine filename based on platform
            import platform as plat
            if plat.system() == "Linux":
                filename = f"{kernel_name}.so"
            elif plat.system() == "Darwin":
                filename = f"{kernel_name}.dylib"
            else:
                filename = f"{kernel_name}.dll"

            # Download kernel
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=filename,
                    revision=version,
                    cache_dir=str(self.cache_dir),
                )
            except Exception as e:
                logger.debug(f"Failed to download kernel: {e}")
                return None

            # Compute SHA256
            import hashlib
            with open(local_path, "rb") as f:
                sha256 = hashlib.sha256(f.read()).hexdigest()

            return LoadedKernel(
                name=kernel_name,
                version=version or "latest",
                path=Path(local_path),
                sha256=sha256,
            )

        except ImportError:
            logger.warning("huggingface_hub not installed")
            return None
        except Exception as e:
            logger.error(f"Failed to load kernel {kernel_name}: {e}")
            return None

    def load_from_lockfile(
        self,
        lockfile_path: str,
    ) -> list[LoadedKernel]:
        """Load all kernels specified in a lockfile.

        Args:
            lockfile_path: Path to the kernel lockfile (JSON)

        Returns:
            List of LoadedKernel objects. Empty list if lockfile
            not found or parsing fails.
        """
        path = Path(lockfile_path)
        if not path.exists():
            logger.warning(f"Lockfile not found: {lockfile_path}")
            return []

        try:
            with open(path, "r") as f:
                data = json.load(f)

            kernels_data = data.get("kernels", [])
            loaded_kernels: list[LoadedKernel] = []

            for kernel_info in kernels_data:
                name = kernel_info.get("name")
                version = kernel_info.get("version")

                if name:
                    kernel = self.load(name, version)
                    if kernel:
                        loaded_kernels.append(kernel)

            return loaded_kernels

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse lockfile: {e}")
            return []
        except Exception as e:
            logger.error(f"Failed to load from lockfile: {e}")
            return []
