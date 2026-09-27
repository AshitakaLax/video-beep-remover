"""The CUDA libraries of the [gpu] extra (DESIGN.md §12).

faster-whisper needs cuBLAS for CUDA 12 and cuDNN 9. The extra installs them with pip, into the
Python environment, where the dynamic linker does not look: faster-whisper's documentation has you
set LD_LIBRARY_PATH. Instead, the libraries are loaded here by path before the first GPU model;
ctranslate2's later lookups by name then find the ones already loaded."""

import ctypes
import functools
import importlib.util
import logging
import sys
from pathlib import Path

log = logging.getLogger(__name__)

# Only what ctranslate2 loads: the cublas package also holds NVBLAS, which would take over BLAS calls.
LIBRARIES = {"nvidia.cublas": "libcublas*.so*", "nvidia.cudnn": "libcudnn*.so*"}


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


@functools.cache
def load_pip_libraries() -> int:
    """Load pip's cuBLAS and cuDNN, if installed (Linux). Returns how many libraries were loaded."""
    if not sys.platform.startswith("linux"):
        return 0
    pending, loaded = _pip_libraries(), 0
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
