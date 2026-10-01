"""The CUDA libraries Whisper needs on a GPU (DESIGN.md §6.8, §12).

faster-whisper needs cuBLAS for CUDA 12 and cuDNN 9, which ctranslate2 looks up by name.

- Linux: the [gpu] extra installs them with pip, into the Python environment, where the dynamic linker
  does not look: faster-whisper's documentation has you set LD_LIBRARY_PATH. Instead, they are loaded
  here by path; ctranslate2's lookups by name then find the ones already loaded.
- Windows: ctranslate2 finds no cuBLAS unless the CUDA toolkit is on PATH, and the cuDNN it ships, which
  it loads when it is imported, breaks PyTorch's: a PyTorch model that then uses cuDNN aborts the whole
  process ("Could not load symbol cudnnGetLibConfig"). PyTorch's CUDA build has both in torch/lib.
  Loaded by path before ctranslate2 is imported, they serve both."""

import ctypes
import functools
import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Only what ctranslate2 loads: the cublas package also holds NVBLAS, which would take over BLAS calls.
LIBRARIES = {"nvidia.cublas": "libcublas*.so*", "nvidia.cudnn": "libcudnn*.so*"}
# What ctranslate2 needs on Windows, as PyTorch's CUDA build names them, each after those it needs.
TORCH_LIBRARIES = ("cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll")


def _pip_libraries() -> list[Path]:
    found: list[Path] = []
    for package, pattern in LIBRARIES.items():
        try:
            spec = importlib.util.find_spec(package)
        except ImportError:  # no nvidia packages at all
            continue
        # Recent NVIDIA wheels are namespace packages: no __file__, only search locations.
        for location in (spec.submodule_search_locations or []) if spec else []:
            found += sorted((Path(location) / "lib").glob(pattern))
    return found


def _torch_libraries() -> list[Path]:
    """TORCH_LIBRARIES in torch/lib, when PyTorch's CUDA build is installed."""
    try:
        spec = importlib.util.find_spec("torch")
    except ImportError:
        return []
    for location in (spec.submodule_search_locations or []) if spec else []:
        found = [Path(location) / "lib" / name for name in TORCH_LIBRARIES]
        if all(path.is_file() for path in found):
            return found
    return []


@functools.cache
def load_pip_libraries() -> int:
    """Load the CUDA libraries installed with pip: the [gpu] extra's on Linux, PyTorch's on Windows. On
    Windows this must come before ctranslate2 is imported. Returns how many libraries were loaded."""
    # Not early returns: checked on one platform, mypy would call the code for the others unreachable.
    if sys.platform.startswith("linux"):
        return _load(_pip_libraries())
    elif sys.platform == "win32":
        return _load(_torch_libraries())
    else:
        return 0


def _load(pending: list[Path]) -> int:
    """Load libraries by path, globally, so that ctranslate2 finds them by name. Returns how many loaded."""
    loaded = 0
    for _ in range(2):  # a library that needs one later in the list loads on the second pass
        failed = []
        for library in pending:
            try:
                ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
                loaded += 1
            except OSError:
                failed.append(library)
        pending = failed
    for library in pending:
        log.debug("could not load %s", library)
    if loaded:
        log.debug("loaded %d CUDA libraries installed with pip", loaded)
    return loaded
