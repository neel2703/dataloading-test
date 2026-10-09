"""miao (AI-HHMI/miao, pip: miao-io) -- a tensorstore-backed PyTorch Dataset
for OME-NGFF Zarr.

Built from miao's real API (read from AI-HHMI/miao@master source, not
guessed): `VolumeDataset(MiaoConfig)` + `VolumeConfig`. See
`tensorstore_method.py` for why this method exists alongside a plain
tensorstore reader -- the gap between the two is miao's own overhead
(OME-NGFF metadata parsing, axes handling, batched tensorstore reads).

IMPORTANT -- pixel-identity check caveat:
`VolumeDataset` does not accept external/explicit patch coordinates. In
"random" sampling mode it draws `np.random.randint` centers internally; in
"sequential" mode it tiles each volume into a fixed grid (`dataset._grid`,
a list of `(vol_idx, center, grid_index)` tuples -- this is miao's private
attribute, read from source, and may change in a future miao release).
Because of this, `verify.py` cannot reuse the same shared `coords` list it
uses for every other method. Instead, for this method specifically, it reads
a handful of grid-sampled patches and compares each one against the full
reference image at the center `VolumeDataset` itself picked (recovered from
`dataset._grid`) -- so the check still proves pixel correctness, just via
miao's own choice of coordinates. This whole file has not been executed yet
(per your request not to run anything); treat the exact field wiring below
as "best effort from the README/source" and expect to adjust `image_key`
or `output_axes` once you run it against the real store.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import BenchConfig
from .base import Method, register
from .zarr_store import scale0_array


class MiaoVolume(Method):
    name = "miao"
    family = "miao_ladder"

    def __init__(self, store: Path, image_name: str, patch_size: int,
                 sampling: str = "sequential", samples_per_epoch: int | None = None,
                 seed: int = 0):
        self.store = store
        self.image_name = image_name
        self.patch_size = patch_size
        self.sampling = sampling
        self.samples_per_epoch = samples_per_epoch
        self.seed = seed
        self._dataset = None
        self._scale = 1.0   # divisor matching base.to_float01's uint8/uint16 scaling
        self._shape_yx: tuple[int, int] | None = None

    def open(self) -> None:
        from miao import VolumeDataset
        from miao.config import MiaoConfig, VolumeConfig

        arr = scale0_array(self.store, self.image_name)
        self._shape_yx = (int(arr.shape[-2]), int(arr.shape[-1]))
        native_dtype = arr.dtype
        if native_dtype == np.uint8:
            self._scale = 255.0
        elif native_dtype == np.uint16:
            self._scale = 65535.0
        else:
            self._scale = 1.0

        volume = VolumeConfig(
            name="bench",
            path=str(self.store),
            image_key=f"images/{self.image_name}",
            zarr_version="zarr3",
            normalize=False,        # keep raw values; we rescale ourselves below,
            patch_normalize=False,  # the same way base.to_float01 does for every other method
        )
        config = MiaoConfig(
            volumes=[volume],
            # MiaoConfig requires a scale-level axis 'l' in output_axes even for
            # single-scale data (pydantic validator in miao.config, not documented
            # in the README) -- learned from an actual validation error, not guessed.
            output_axes="lcyx",
            patch_size=[self.patch_size, self.patch_size],
            # MiaoConfig requires exactly one of resolutions/resolution_sampling.
            # Single-scale 2D data with no real physical units here, so this is a
            # placeholder voxel size (one scale, one entry per spatial axis: y, x)
            # -- learned from a second validation error, not guessed up front.
            resolutions=[[1.0, 1.0]],
            sampling=self.sampling,  # "sequential" = deterministic grid -> coordinates we can verify
            # image_dtype="float32" makes miao cast raw ints to float32 WITHOUT
            # rescaling (normalize=False) -- base.to_float01 would then see an
            # already-float array and skip its own /255 or /65535 step, so we
            # can't reuse it here; _scale (computed above from the real on-disk
            # dtype) replicates exactly what to_float01 does for every other method.
            image_dtype="float32",
            # samples_per_epoch is only consulted by miao in sampling="random"
            # mode (VolumeDataset.__len__); it is ignored in "sequential" mode,
            # so it is only passed when it actually matters.
            **({"samples_per_epoch": self.samples_per_epoch}
               if self.sampling == "random" and self.samples_per_epoch is not None else {}),
        )
        self._dataset = VolumeDataset(config)
        if self.sampling == "random":
            # miao draws random centers from the *module-level* np.random inside
            # VolumeDataset.__getitem__ (miao-io 0.4.2, src/miao/dataset.py), so
            # reseeding here -- immediately before the reads that follow open() --
            # makes the patch sequence reproducible across the recording pass, the
            # verification pass and every timed run. Only done in "random" mode, so
            # the default ("sequential") path is unchanged.
            np.random.seed(self.seed)

    def __len__(self) -> int:
        return len(self._dataset)

    def read_patch_by_index(self, i: int) -> np.ndarray:
        batch = self._dataset[i]
        img = np.asarray(batch["img"])      # (l, c, ph, ph) per output_axes="lcyx", raw values as float32
        while img.ndim > 2:                 # squeeze the singleton level/channel dims
            img = img[0]
        return (img / self._scale).astype(np.float32)

    def read_patch_and_coord_by_index(self, i: int) -> tuple[np.ndarray, tuple[int, int]]:
        """Same read as read_patch_by_index(), plus the TOP-LEFT (y, x) miao used.

        miao reports patch positions as CENTERS, via the public
        `sample["meta"]["coordinate"]` (output_axes spatial order, so "yx" for
        output_axes="lcyx"). It is populated in both sampling modes, which is why
        this uses it instead of the private `_grid` that `grid_center()` reads.

        The center -> top-left conversion below replicates miao's own origin
        computation in `VolumeDataset.__getitem__` (miao-io 0.4.2,
        src/miao/dataset.py):

            origin = clip(floor((center + img_offset) / rel_factors) - eff_shape // 2,
                          0, level_shape - eff_shape)

        For this benchmark's single-level, single-resolution store img_offset == 0
        and rel_factors == 1, so it reduces to the clip below -- edge-clamped grid
        positions included, exactly as miao produces them. This mirrors a miao
        internal rather than calling it; verify.py proves it is right by comparing
        miao's own returned pixels against the full-image reference at these
        coordinates.
        """
        batch = self._dataset[i]
        img = np.asarray(batch["img"])
        while img.ndim > 2:
            img = img[0]
        cy, cx = (int(v) for v in batch["meta"]["coordinate"])
        ph, (h, w) = self.patch_size, self._shape_yx
        y = int(np.clip(cy - ph // 2, 0, h - ph))
        x = int(np.clip(cx - ph // 2, 0, w - ph))
        return (img / self._scale).astype(np.float32), (y, x)

    def record_coords(self, n: int) -> list[tuple[int, int]]:
        """The first `min(n, len(self))` patch positions, in miao's own read
        order, as top-left (y, x). open() must have been called first."""
        return [self.read_patch_and_coord_by_index(i)[1]
                for i in range(min(n, len(self)))]

    def grid_center(self, i: int):
        """(y, x) center miao itself picked for item i -- see module docstring."""
        _, center, _ = self._dataset._grid[i]
        return center

    def read_patch(self, y: int, x: int, ph: int) -> np.ndarray:
        raise NotImplementedError(
            "MiaoVolume is index-driven (sequential grid), not coordinate-driven. "
            "Use read_patch_by_index() / grid_center() -- see verify.py."
        )


@register("miao")
def _factory(cfg: BenchConfig) -> Method:
    return MiaoVolume(cfg.sdata_compressed, cfg.image_name, cfg.patch_size,
                      sampling=cfg.miao_sampling, samples_per_epoch=cfg.n_patches,
                      seed=cfg.seed)
