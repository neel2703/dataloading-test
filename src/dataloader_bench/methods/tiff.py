"""TIFF family: tifffile's aszarr view over tiled OME/plain TIFFs, plus a raw
memmap of the original (strip-based, uncompressed) file."""

from __future__ import annotations

import numpy as np
import tifffile
import zarr

from ..config import BenchConfig
from .base import Method, register, to_float01


class TiffAsZarr(Method):
    """Tiled TIFF read lazily through tifffile's zarr view -- only the tile(s)
    touched by a slice get decoded."""

    family = "tiff"

    def __init__(self, path, name: str):
        self.path = str(path)
        self.name = name
        self._z = None

    def open(self) -> None:
        self._z = zarr.open(tifffile.imread(self.path, aszarr=True), mode="r")

    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        return to_float01(self._z[y:y + ph, x:x + ph])


class TiffMemmap(Method):
    """numpy.memmap over the page -- no decode step, OS pages bytes in lazily.
    Only valid for uncompressed, contiguous data (the original source file)."""

    name = "tiff_memmap"
    family = "tiff"

    def __init__(self, path):
        self.path = str(path)
        self._arr = None

    def open(self) -> None:
        self._arr = tifffile.memmap(self.path, page=0)

    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        return to_float01(self._arr[y:y + ph, x:x + ph])


@register("tiff_ome_uncompressed")
def _f1(cfg: BenchConfig) -> Method:
    m = TiffAsZarr(cfg.ome_uncompressed, "tiff_ome_uncompressed")
    m.name = "tiff_ome_uncompressed"
    return m


@register("tiff_ome_compressed")
def _f2(cfg: BenchConfig) -> Method:
    m = TiffAsZarr(cfg.ome_compressed, "tiff_ome_compressed")
    m.name = "tiff_ome_compressed"
    return m


@register("tiff_plain_tiled")
def _f3(cfg: BenchConfig) -> Method:
    m = TiffAsZarr(cfg.tiled_plain, "tiff_plain_tiled")
    m.name = "tiff_plain_tiled"
    return m


@register("tiff_memmap")
def _f4(cfg: BenchConfig) -> Method:
    return TiffMemmap(cfg.src)
