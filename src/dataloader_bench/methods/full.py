"""Whole-image-in-RAM baseline (what a naive cv2.imread-based Dataset does)."""

from __future__ import annotations

import numpy as np

from ..config import BenchConfig
from .base import Method, register, to_float01


class FullInRam(Method):
    name = "full"
    family = "baseline"

    def __init__(self, path):
        self.path = str(path)
        self._arr: np.ndarray | None = None

    def open(self) -> None:
        import cv2
        raw = cv2.imread(self.path, cv2.IMREAD_UNCHANGED)
        if raw is None:
            raise RuntimeError(f"cv2 failed to read {self.path}")
        self._arr = to_float01(np.squeeze(raw))

    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        return self._arr[y:y + ph, x:x + ph]


@register("full")
def _factory(cfg: BenchConfig) -> Method:
    return FullInRam(cfg.src)
