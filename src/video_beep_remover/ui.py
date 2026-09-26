"""How the pipeline reports progress. The CLI supplies a Rich implementation; tests use NullUI."""

import contextlib
from collections.abc import Callable, Iterator
from contextlib import AbstractContextManager
from typing import Protocol

Progress = Callable[[float], None]


class UI(Protocol):
    def info(self, message: str) -> None: ...
    def warn(self, message: str) -> None: ...
    def progress(self, label: str, total: float) -> AbstractContextManager[Progress]: ...
    def status(self, label: str) -> AbstractContextManager[None]: ...


class NullUI:
    def info(self, message: str) -> None:
        pass

    def warn(self, message: str) -> None:
        pass

    @contextlib.contextmanager
    def progress(self, label: str, total: float) -> Iterator[Progress]:
        yield lambda done: None

    @contextlib.contextmanager
    def status(self, label: str) -> Iterator[None]:
        yield
