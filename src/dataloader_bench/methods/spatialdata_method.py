"""Rung 4 of the ladder: spatialdata.read_zarr(...).images[name] -- the full
SpatialData read path, with whatever overhead that API adds on top of dask."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import BenchConfig
from .base import Method, register, to_float01


class SpatialDataRead(Method):
    name = "spatialdata"
    family = "zarr_ladder"

    def __init__(self, store: Path, image_name: str):
        self.store = str(store)
        self.image_name = image_name
        self._da = None

    def open(self) -> None:
        from spatialdata import read_zarr
        self._da = read_zarr(self.store).images[self.image_name]

    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        sub = self._da[:, y:y + ph, x:x + ph].values
        return to_float01(sub)


@register("spatialdata_compressed")
def _f1(cfg: BenchConfig) -> Method:
    m = SpatialDataRead(cfg.sdata_compressed, cfg.image_name)
    m.name = "spatialdata_compressed"
    return m


@register("spatialdata_uncompressed")
def _f2(cfg: BenchConfig) -> Method:
    m = SpatialDataRead(cfg.sdata_uncompressed, cfg.image_name)
    m.name = "spatialdata_uncompressed"
    return m
