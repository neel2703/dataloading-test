"""Rung 3 of the ladder: xarray.DataArray wrapping the dask array."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import BenchConfig
from .base import Method, register, to_float01
from .zarr_store import scale0_array_path


class XrDask(Method):
    family = "zarr_ladder"

    def __init__(self, store: Path, image_name: str):
        self.store = store
        self.image_name = image_name
        self._da = None

    def open(self) -> None:
        import dask.array as da
        import xarray as xr
        arr = da.from_zarr(scale0_array_path(self.store, self.image_name))
        self._da = xr.DataArray(arr, dims=("c", "y", "x"))

    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        sub = self._da[:, y:y + ph, x:x + ph].values   # triggers dask compute
        return to_float01(sub)


@register("xr_dask_compressed")
def _f1(cfg: BenchConfig) -> Method:
    m = XrDask(cfg.sdata_compressed, cfg.image_name)
    m.name = "xr_dask_compressed"
    return m


@register("xr_dask_uncompressed")
def _f2(cfg: BenchConfig) -> Method:
    m = XrDask(cfg.sdata_uncompressed, cfg.image_name)
    m.name = "xr_dask_uncompressed"
    return m
