"""Rung 1 of the ladder: raw scale-0 zarr array via zarr-python (+ zarrs
codec pipeline if configured). No dask, no xarray, no SpatialData overhead."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import BenchConfig
from .base import Method, register, to_float01
from .zarr_store import scale0_array


class ZarrDirect(Method):
    family = "zarr_ladder"

    def __init__(self, store: Path, image_name: str):
        self.store = store
        self.image_name = image_name
        self._z = None

    def open(self) -> None:
        self._z = scale0_array(self.store, self.image_name)

    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        return to_float01(self._z[:, y:y + ph, x:x + ph])


@register("zarr_direct_compressed")
def _f1(cfg: BenchConfig) -> Method:
    m = ZarrDirect(cfg.sdata_compressed, cfg.image_name)
    m.name = "zarr_direct_compressed"
    return m


@register("zarr_direct_uncompressed")
def _f2(cfg: BenchConfig) -> Method:
    m = ZarrDirect(cfg.sdata_uncompressed, cfg.image_name)
    m.name = "zarr_direct_uncompressed"
    return m
