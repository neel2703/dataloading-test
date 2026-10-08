"""Pixel-identity checks: every method must read exactly the same bytes as
the full-image-in-RAM reference, modulo the shared `to_float01` scaling."""

from __future__ import annotations

import random

import numpy as np

from .config import BenchConfig
from .methods.base import Method
from .methods.miao_method import MiaoVolume


def build_coords(h: int, w: int, ph: int, n: int, seed: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(0, h - ph), rng.randint(0, w - ph)) for _ in range(n)]


def verify_coord_method(ref: Method, other: Method, coords: list[tuple[int, int]],
                         n_check: int, ph: int) -> None:
    other.open()
    try:
        for y, x in coords[:n_check]:
            a = ref.read_patch(y, x, ph)
            b = other.read_patch(y, x, ph)
            if not np.array_equal(a, b):
                raise AssertionError(
                    f"{other.name}: patch at (y={y}, x={x}) differs from the full-image reference"
                )
    finally:
        other.close()


def verify_miao(ref: Method, miao_method: MiaoVolume, n_check: int, ph: int) -> None:
    """miao is index-driven (sequential grid), not coordinate-driven -- see
    the caveat in methods/miao_method.py. We pull the centers *it* picked
    and compare against the reference at those same centers."""
    miao_method.open()
    try:
        n = min(n_check, len(miao_method))
        for i in range(n):
            patch = miao_method.read_patch_by_index(i)
            cy, cx = miao_method.grid_center(i)
            y, x = cy - ph // 2, cx - ph // 2
            ref_patch = ref.read_patch(y, x, ph)
            if not np.array_equal(ref_patch, patch):
                raise AssertionError(f"miao: grid item {i} (center y={cy}, x={cx}) differs from reference")
    finally:
        miao_method.close()


def verify_all(cfg: BenchConfig, ref: Method, others: dict[str, Method],
                coords: list[tuple[int, int]], n_check: int = 20) -> None:
    ref.open()
    try:
        for name, method in others.items():
            if isinstance(method, MiaoVolume):
                verify_miao(ref, method, n_check, cfg.patch_size)
            else:
                verify_coord_method(ref, method, coords, n_check, cfg.patch_size)
            print(f"[verify] {name}: OK")
    finally:
        ref.close()
    print("[verify] all methods pixel-identical to the full-image reference")
