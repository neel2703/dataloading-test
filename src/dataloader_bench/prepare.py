"""Download the source image (if needed) and build every derived on-disk
variant the methods read from."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import numpy as np
import tifffile

from .config import BenchConfig
from .methods.zarr_store import build_spatialdata_store


def _make_synthetic(shape: tuple[int, int], seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 65535, size=shape, dtype=np.uint16)


def download_src(cfg: BenchConfig) -> None:
    if cfg.src.exists():
        return
    if cfg.synthetic:
        print(f"[prepare] synthetic mode: writing a fake {cfg.synthetic_shape} source image")
        page0 = _make_synthetic(cfg.synthetic_shape, cfg.seed)
        tifffile.imwrite(str(cfg.src), page0[np.newaxis])  # fake single-page "original"
        return

    assert cfg.src_url and "PASTE" not in cfg.src_url, (
        "set BenchConfig.src_url (or pass --synthetic for a smoke test without the real data)"
    )
    part = cfg.src.with_suffix(cfg.src.suffix + ".part")
    print(f"[download] {cfg.src_url}")
    urllib.request.urlretrieve(cfg.src_url, part)
    part.rename(cfg.src)


def _read_page0(cfg: BenchConfig) -> np.ndarray:
    if cfg.synthetic:
        return tifffile.imread(str(cfg.src), key=0)
    return tifffile.imread(str(cfg.src), key=0)


def prepare(cfg: BenchConfig) -> tuple[int, int]:
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    download_src(cfg)
    page0 = _read_page0(cfg)
    print(f"[prepare] page0 {page0.shape} {page0.dtype}")

    tifffile.imwrite(str(cfg.ome_uncompressed), page0, tile=(cfg.chunk_size, cfg.chunk_size),
                      compression=None, photometric="minisblack",
                      ome=True, metadata={"axes": "YX"})
    tifffile.imwrite(str(cfg.ome_compressed), page0, tile=(cfg.chunk_size, cfg.chunk_size),
                      compression="zlib", photometric="minisblack",
                      ome=True, metadata={"axes": "YX"})
    tifffile.imwrite(str(cfg.tiled_plain), page0, tile=(cfg.chunk_size, cfg.chunk_size),
                      compression=None, photometric="minisblack")
    tifffile.imwrite(str(cfg.sdata_input), page0[np.newaxis], tile=(cfg.chunk_size, cfg.chunk_size),
                      compression=None, photometric="minisblack",
                      ome=True, metadata={"axes": "CYX"})

    build_spatialdata_store(cfg.sdata_compressed, cfg.sdata_input, cfg.image_name,
                             cfg.chunk_size, compressed=True)
    build_spatialdata_store(cfg.sdata_uncompressed, cfg.sdata_input, cfg.image_name,
                             cfg.chunk_size, compressed=False)

    return page0.shape[:2]
