"""Plain tensorstore read of the same scale-0 array, with no miao involved.

This exists so miao's overhead (OME-NGFF metadata parsing, axes handling,
config validation) is visible as the *difference* between this method and
`miao_method.py`, rather than being hidden inside "tensorstore is slow".

The tensorstore spec mirrors exactly what miao's own `miao.store.open_store`
builds (verified against the miao source, AI-HHMI/miao@master,
src/miao/store.py): {"driver": "zarr" | "zarr3", "kvstore": {"driver":
"file", "path": ...}}. zarr-version is detected from whether scale-0 has a
zarr.json (zarr3) or .zarray (zarr2).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import BenchConfig
from .base import Method, register, to_float01
from .zarr_store import scale0_group_path


def _zarr_version_dir(store: Path, image_name: str) -> tuple[str, Path]:
    grp = scale0_group_path(store, image_name)
    import zarr
    g = zarr.open_group(str(grp), mode="r")
    name = sorted(g.array_keys())[0]
    arr_path = grp / name
    if (arr_path / "zarr.json").exists():
        return "zarr3", arr_path
    return "zarr2", arr_path


class TensorstoreDirect(Method):
    family = "miao_ladder"

    def __init__(self, store: Path, image_name: str):
        self.store = store
        self.image_name = image_name
        self._ts_array = None

    def open(self) -> None:
        import tensorstore as ts
        version, arr_path = _zarr_version_dir(self.store, self.image_name)
        driver = "zarr3" if version == "zarr3" else "zarr"
        spec = {"driver": driver, "kvstore": {"driver": "file", "path": str(arr_path)}}
        self._ts_array = ts.open(spec).result()

    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        sub = self._ts_array[:, y:y + ph, x:x + ph].read().result()
        return to_float01(np.asarray(sub))


@register("tensorstore_direct")
def _f1(cfg: BenchConfig) -> Method:
    m = TensorstoreDirect(cfg.sdata_compressed, cfg.image_name)
    m.name = "tensorstore_direct"
    return m
