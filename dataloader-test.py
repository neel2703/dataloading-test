import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import random
import shutil
import time
from pathlib import Path

import numpy as np
import tifffile
import torch
import zarr
from torch.utils.data import Dataset, DataLoader

import dask
dask.config.set(scheduler="synchronous")

# ---------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
SRC  = HERE / "Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif"

OME_UNCOMPRESSED = SRC.with_name("page0_uncompressed.ome.tif")
OME_COMPRESSED   = SRC.with_name("page0_compressed.ome.tif")
TILED_PLAIN      = SRC.with_name("page0_tiled_plain.tif")
SDATA_INPUT        = SRC.with_name("page0_cyx.ome.tif")            # (c,y,x) input for spatialdata_io
SDATA_COMPRESSED   = SRC.with_name("page0_sdata_compressed.zarr")
SDATA_UNCOMPRESSED = SRC.with_name("page0_sdata_uncompressed.zarr")

IMG_NAME    = "image"
PATCH_SIZE  = 256
CHUNK       = 256
N_PATCHES   = 500
BATCH_SIZE  = 32
NUM_WORKERS = 0
SEED        = 0


def to_float01(a):
    a = np.asarray(a)
    if a.ndim == 3:            # (c=1, y, x) -> (y, x)
        a = a[0]
    if a.dtype == np.uint8:  return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16: return a.astype(np.float32) / 65535.0
    return a.astype(np.float32)


# ---------------------------------------------------------------- spatialdata store
def _scale0(store):                          # full-res zarr array: images/<name>/0
    g = zarr.open_group(str(store / "images" / IMG_NAME), mode="r")
    return g[sorted(g.array_keys())[0]]


def _uncompress_scale0(store):               # rewrite scale-0 with no compression, keep attrs
    grp = store / "images" / IMG_NAME
    g = zarr.open_group(str(grp), mode="r")
    name = sorted(g.array_keys())[0]
    z = g[name]
    data, chunks, dtype, attrs = z[:], z.chunks, z.dtype, dict(z.attrs)
    shutil.rmtree(grp / name)
    zn = zarr.create_array(store=str(grp / name), shape=data.shape,
                           chunks=chunks, dtype=dtype, compressors=None)
    zn[:] = data
    zn.attrs.update(attrs)


def build_spatialdata_store(store, compressed):
    import spatialdata_io
    from spatialdata import SpatialData
    if store.exists():
        shutil.rmtree(store)
    img = spatialdata_io.image(input=SDATA_INPUT, data_axes=("c", "y", "x"),
                               coordinate_system="global", chunks=CHUNK,
                               scale_factors=None)
    SpatialData(images={IMG_NAME: img}).write(store)
    if not compressed:
        _uncompress_scale0(store)


# ---------------------------------------------------------------- dataset
class PatchDataset(Dataset):
    """kinds: full / zarr / memmap (tests 1-5),
              zstore_direct / zstore_xarray (spatialdata store).
    Handle opened lazily so each DataLoader worker gets its own."""

    def __init__(self, kind, path, coords, patch_size=PATCH_SIZE):
        self.kind = kind
        self.path = str(path)
        self.coords = coords
        self.ph = patch_size
        self._handle = None

    def _open(self):
        if self.kind == "full":
            import cv2
            self._handle = to_float01(np.squeeze(cv2.imread(self.path, cv2.IMREAD_UNCHANGED)))
        elif self.kind == "memmap":
            self._handle = tifffile.memmap(self.path, page=0)
        elif self.kind == "zarr":                      # tiff tiles via zarr interface
            self._handle = zarr.open(tifffile.imread(self.path, aszarr=True), mode="r")
        elif self.kind == "zstore_direct":             # raw scale-0 zarr array
            self._handle = _scale0(Path(self.path))
        elif self.kind == "zstore_xarray":             # spatialdata xarray interface
            from spatialdata import read_zarr
            self._handle = read_zarr(self.path).images[IMG_NAME]
        else:
            raise ValueError(self.kind)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self._handle is None:
            self._open()
        y, x = self.coords[i]
        ph = self.ph
        if self.kind in ("zstore_direct", "zstore_xarray"):
            sub = self._handle[:, y:y + ph, x:x + ph]      # (c, ph, ph)
            if self.kind == "zstore_xarray":
                sub = sub.values                           # trigger dask compute
        else:
            sub = self._handle[y:y + ph, x:x + ph]
        patch = np.asarray(sub) if self.kind == "full" else to_float01(sub)
        return torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)


# ---------------------------------------------------------------- prep
def prepare():
    assert SRC.exists(), f"source image not found: {SRC}"
    page0 = tifffile.imread(str(SRC), key=0)
    print(f"[prepare] page0 {page0.shape} {page0.dtype}")

    tifffile.imwrite(str(OME_UNCOMPRESSED), page0, tile=(CHUNK, CHUNK),
                     compression=None, photometric="minisblack",
                     ome=True, metadata={"axes": "YX"})
    tifffile.imwrite(str(OME_COMPRESSED), page0, tile=(CHUNK, CHUNK),
                     compression="zlib", photometric="minisblack",
                     ome=True, metadata={"axes": "YX"})
    tifffile.imwrite(str(TILED_PLAIN), page0, tile=(CHUNK, CHUNK),
                     compression=None, photometric="minisblack")

    # (c,y,x) copy for spatialdata_io.image, then build the two stores
    tifffile.imwrite(str(SDATA_INPUT), page0[np.newaxis], tile=(CHUNK, CHUNK),
                     compression=None, photometric="minisblack",
                     ome=True, metadata={"axes": "CYX"})
    build_spatialdata_store(SDATA_COMPRESSED, compressed=True)
    build_spatialdata_store(SDATA_UNCOMPRESSED, compressed=False)
    return page0.shape[:2]


def build_coords(h, w, ph, n, seed):
    rng = random.Random(seed)
    return [(rng.randint(0, h - ph), rng.randint(0, w - ph)) for _ in range(n)]


# ---------------------------------------------------------------- benchmark
def benchmark(kind, path, coords):
    ds = PatchDataset(kind, path, coords)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS,
                    shuffle=False, pin_memory=False)
    t0 = time.perf_counter()
    n = 0
    for batch in dl:
        n += batch.shape[0]
    dt = time.perf_counter() - t0
    return dt, n / dt


def verify(coords):
    ref = PatchDataset("full", SRC, coords)
    checks = [
        ("2", "zarr", OME_UNCOMPRESSED), ("3", "zarr", OME_COMPRESSED),
        ("4", "zarr", TILED_PLAIN),      ("5", "memmap", SRC),
        ("6", "zstore_direct", SDATA_COMPRESSED),   ("7", "zstore_xarray", SDATA_COMPRESSED),
        ("8", "zstore_direct", SDATA_UNCOMPRESSED), ("9", "zstore_xarray", SDATA_UNCOMPRESSED),
    ]
    for label, kind, path in checks:
        other = PatchDataset(kind, path, coords)
        ok = all(torch.equal(ref[i], other[i]) for i in range(20))
        assert ok, f"test {label} produced different pixels"
    print("[verify] all methods pixel-identical to test 1")


def main():
    h, w = prepare()
    coords = build_coords(h, w, PATCH_SIZE, N_PATCHES, SEED)
    print(f"\n{N_PATCHES} patches, batch={BATCH_SIZE}, workers={NUM_WORKERS}\n")
    verify(coords)

    tests = [
        ("1. full    (original logic)",     "full",          SRC),
        ("2. tiff    (ome uncompressed)",   "zarr",          OME_UNCOMPRESSED),
        ("3. tiff    (ome compressed)",     "zarr",          OME_COMPRESSED),
        ("4. tiff    (plain tiled)",        "zarr",          TILED_PLAIN),
        ("5. memmap  (original)",           "memmap",        SRC),
        ("6. sdata   (compr, direct)",      "zstore_direct", SDATA_COMPRESSED),
        ("7. sdata   (compr, xarray)",      "zstore_xarray", SDATA_COMPRESSED),
        ("8. sdata   (raw,   direct)",      "zstore_direct", SDATA_UNCOMPRESSED),
        ("9. sdata   (raw,   xarray)",      "zstore_xarray", SDATA_UNCOMPRESSED),
    ]

    print(f"\n{'method':34s} {'time (s)':>10s} {'patches/s':>12s}")
    print("-" * 58)
    for name, kind, path in tests:
        dt, rate = benchmark(kind, path, coords)
        print(f"{name:34s} {dt:10.3f} {rate:12.0f}")


if __name__ == "__main__":
    main()