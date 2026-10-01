import sys
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from video_beep_remover.asr import cuda


@pytest.fixture
def wheels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """The layout of the nvidia-cublas-cu12 and nvidia-cudnn-cu12 wheels, and a record of loads."""
    for package, names in {
        "cublas": ["libcublas.so.12", "libcublasLt.so.12", "libnvblas.so.12"],
        "cudnn": ["libcudnn.so.9", "libcudnn_ops.so.9"],
    }.items():
        (tmp_path / package / "lib").mkdir(parents=True)
        for name in names:
            (tmp_path / package / "lib" / name).write_bytes(b"")
    specs = {
        "nvidia.cublas": SimpleNamespace(submodule_search_locations=[str(tmp_path / "cublas")]),
        "nvidia.cudnn": SimpleNamespace(submodule_search_locations=[str(tmp_path / "cudnn")]),
    }
    monkeypatch.setattr(cuda.importlib.util, "find_spec", lambda name: specs.get(name))
    loads: list[str] = []

    def load(path: str, mode: int = 0) -> Any:
        name = Path(path).name
        # libcublas needs libcublasLt, which comes after it in the list: it loads on the second pass
        if name == "libcublas.so.12" and "libcublasLt.so.12" not in loads:
            raise OSError("libcublasLt.so.12: cannot open shared object file")
        loads.append(name)
        return object()

    monkeypatch.setattr(cuda.ctypes, "CDLL", load)
    monkeypatch.setattr(sys, "platform", "linux")
    cuda.load_pip_libraries.cache_clear()
    yield loads
    cuda.load_pip_libraries.cache_clear()


def test_pip_installed_cuda_libraries_are_loaded_by_path(wheels: list[str]) -> None:
    assert cuda.load_pip_libraries() == 4
    assert wheels == ["libcublasLt.so.12", "libcudnn.so.9", "libcudnn_ops.so.9", "libcublas.so.12"]
    assert cuda.load_pip_libraries() == 4 and len(wheels) == 4  # once per process


def test_nothing_is_loaded_without_the_gpu_extra_or_pytorchs_cuda_build(
    wheels: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cuda.importlib.util, "find_spec", lambda name: None)
    assert cuda.load_pip_libraries() == 0
    cuda.load_pip_libraries.cache_clear()

    def missing(name: str) -> None:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(cuda.importlib.util, "find_spec", missing)
    assert cuda.load_pip_libraries() == 0
    cuda.load_pip_libraries.cache_clear()
    monkeypatch.setattr(sys, "platform", "win32")  # and no PyTorch
    assert cuda.load_pip_libraries() == 0
    cuda.load_pip_libraries.cache_clear()
    monkeypatch.setattr(sys, "platform", "darwin")
    assert cuda.load_pip_libraries() == 0
    assert wheels == []


def test_on_windows_pytorchs_cuda_libraries_are_loaded(
    wheels: list[str], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ctranslate2 finds no cuBLAS on Windows, and the cuDNN it loads on import breaks PyTorch's."""
    lib = tmp_path / "torch" / "lib"
    lib.mkdir(parents=True)
    for name in (*cuda.TORCH_LIBRARIES, "torch_cpu.dll"):
        (lib / name).write_bytes(b"")
    torch = SimpleNamespace(submodule_search_locations=[str(tmp_path / "torch")])
    monkeypatch.setattr(cuda.importlib.util, "find_spec", lambda name: torch if name == "torch" else None)
    monkeypatch.setattr(sys, "platform", "win32")
    assert cuda.load_pip_libraries() == 4
    assert wheels == ["cudart64_12.dll", "cublasLt64_12.dll", "cublas64_12.dll", "cudnn64_9.dll"]
    cuda.load_pip_libraries.cache_clear()
    (lib / "cudnn64_9.dll").unlink()  # PyTorch's CPU build has none of them
    assert cuda.load_pip_libraries() == 0
