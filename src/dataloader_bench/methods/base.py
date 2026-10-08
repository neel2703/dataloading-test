"""Method registry.

Each loading strategy is one class implementing `Method`. `open()` and
`close()` are timed separately from `read_patch()` by the runner, and
`open()` is never allowed to happen implicitly inside `read_patch()` (the
runner calls `open()` exactly once up front, per worker).

A method is registered with `@register("name")` and discovered later by that
name from `BenchConfig.methods` — adding a new loader means adding one file
plus one decorator, nothing else needs to change.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable

import numpy as np

from ..config import BenchConfig

REGISTRY: dict[str, Callable[[BenchConfig], "Method"]] = {}


def register(name: str):
    def deco(factory: Callable[[BenchConfig], "Method"]):
        if name in REGISTRY:
            raise ValueError(f"method {name!r} already registered")
        REGISTRY[name] = factory
        return factory
    return deco


class Method(ABC):
    """One loading strategy. Family groups related methods for plotting."""

    name: str = "unnamed"
    family: str = "other"

    @abstractmethod
    def open(self) -> None:
        """Acquire whatever handle/connection this method needs. Timed."""

    @abstractmethod
    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        """Return a (ph, ph) float32 patch in [0, 1], normalized the same way
        across every method so pixel-identity checks are meaningful."""

    def close(self) -> None:
        """Release resources. No-op by default."""


def to_float01(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.ndim == 3:            # (c=1, y, x) -> (y, x)
        a = a[0]
    if a.dtype == np.uint8:
        return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16:
        return a.astype(np.float32) / 65535.0
    return a.astype(np.float32)


class MethodError(RuntimeError):
    """Raised by the runner to wrap a failing method without killing the run."""
