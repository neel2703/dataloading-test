"""Rung 2 of the ladder: dask.array.from_zarr on the same scale-0 array."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import BenchConfig
from .base import Method, register, to_float01
from .zarr_store import scale0_array_path


class DaskZarr(Method):
    family = "zarr_ladder"

    def __init__(self, store: Path, image_name: str):
        self.store = store
        self.image_name = image_name
        self._arr = None

    def open(self) -> None:
        import dask.array as da
        self._arr = da.from_zarr(scale0_array_path(self.store, self.image_name))

    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        sub = self._arr[:, y:y + ph, x:x + ph].compute()
        return to_float01(sub)


@register("dask_zarr_compressed")
def _f1(cfg: BenchConfig) -> Method:
    m = DaskZarr(cfg.sdata_compressed, cfg.image_name)
    m.name = "dask_zarr_compressed"
    return m


@register("dask_zarr_uncompressed")
def _f2(cfg: BenchConfig) -> Method:
    m = DaskZarr(cfg.sdata_uncompressed, cfg.image_name)
    m.name = "dask_zarr_uncompressed"
    return m
