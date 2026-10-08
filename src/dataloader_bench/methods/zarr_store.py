"""Shared helpers for the SpatialData-backed OME-Zarr store.

Used both by `prepare.py` (to build the on-disk store) and by every method
in the zarr/-dask/xarray/spatialdata ladder (to locate scale-0 without
re-implementing SpatialData's internal layout in four places).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import zarr


def scale0_group_path(store: Path, image_name: str) -> Path:
    return store / "images" / image_name


def scale0_array(store: Path, image_name: str):
    g = zarr.open_group(str(scale0_group_path(store, image_name)), mode="r")
    return g[sorted(g.array_keys())[0]]


def scale0_array_path(store: Path, image_name: str) -> str:
    g = zarr.open_group(str(scale0_group_path(store, image_name)), mode="r")
    return str(scale0_group_path(store, image_name) / sorted(g.array_keys())[0])


def uncompress_scale0(store: Path, image_name: str) -> None:
    """Rewrite scale-0 with no compression, keeping shape/dtype/attrs, so the
    *_uncompressed store isolates decompression cost from everything else."""
    grp = scale0_group_path(store, image_name)
    g = zarr.open_group(str(grp), mode="r")
    name = sorted(g.array_keys())[0]
    z = g[name]
    data, chunks, dtype, attrs = z[:], z.chunks, z.dtype, dict(z.attrs)
    shutil.rmtree(grp / name)
    zn = zarr.create_array(store=str(grp / name), shape=data.shape,
                            chunks=chunks, dtype=dtype, compressors=None)
    zn[:] = data
    zn.attrs.update(attrs)


def build_spatialdata_store(store: Path, sdata_input: Path, image_name: str,
                             chunk_size: int, compressed: bool) -> None:
    import spatialdata_io
    from spatialdata import SpatialData

    if store.exists():
        shutil.rmtree(store)
    img = spatialdata_io.image(input=sdata_input, data_axes=("c", "y", "x"),
                                coordinate_system="global", chunks=chunk_size,
                                scale_factors=None)
    SpatialData(images={image_name: img}).write(store)
    if not compressed:
        uncompress_scale0(store, image_name)
