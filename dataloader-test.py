import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # CPU only, before torch import

import random
import time
from pathlib import Path

import cv2
import numpy as np
import tifffile
import torch
import zarr
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------- config
HERE = Path(__file__).resolve().parent
SRC  = HERE / "Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif"

OME_UNCOMPRESSED = SRC.with_name("page0_uncompressed.ome.tif")
OME_COMPRESSED   = SRC.with_name("page0_compressed.ome.tif")
TILED_PLAIN      = SRC.with_name("page0_tiled_plain.tif")

PATCH_SIZE  = 256
N_PATCHES   = 500
BATCH_SIZE  = 32
NUM_WORKERS = 0      
SEED        = 42


def to_float01(a: np.ndarray) -> np.ndarray:
    if a.dtype == np.uint8:  return a.astype(np.float32) / 255.0
    if a.dtype == np.uint16: return a.astype(np.float32) / 65535.0
    return a.astype(np.float32)


# ---------------------------------------------------------------- dataset
class PatchDataset(Dataset):
    """One dataset, three loading styles selected by `kind`:
       'full'   -> whole normalized image cached in RAM, sliced.
       'zarr'   -> tiled tiff, only touched tiles decoded per patch.
       'memmap' -> original file mapped; OS pages in only touched bytes.
    The handle is opened LAZILY (first __getitem__), so with num_workers>0
    each worker process opens its own — nothing is shared across the fork."""

    def __init__(self, kind, path, coords, patch_size=PATCH_SIZE):
        self.kind = kind
        self.path = str(path)
        self.coords = coords
        self.ph = patch_size
        self._handle = None

    def _open(self):
        if self.kind == "full":
            img = np.squeeze(cv2.imread(self.path, cv2.IMREAD_UNCHANGED))
            self._handle = to_float01(img)                 # full float32 image resident
        elif self.kind == "memmap":
            self._handle = tifffile.memmap(self.path, page=0)
        elif self.kind == "zarr":
            store = tifffile.imread(self.path, aszarr=True)
            self._handle = zarr.open(store, mode="r")
        else:
            raise ValueError(self.kind)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        if self._handle is None:
            self._open()
        y, x = self.coords[i]
        patch = np.asarray(self._handle[y:y + self.ph, x:x + self.ph])
        if self.kind != "full":          # 'full' handle is already normalized
            patch = to_float01(patch)
        return torch.from_numpy(np.ascontiguousarray(patch)).unsqueeze(0)  # (1,H,W)


# ---------------------------------------------------------------- prep
def prepare_files():
    assert SRC.exists(), f"source image not found: {SRC}"
    print(f"[prepare] reading page 0 of {SRC.name}")
    page0 = tifffile.imread(str(SRC), key=0)
    print(f"[prepare] page0 {page0.shape} {page0.dtype}")

    tifffile.imwrite(str(OME_UNCOMPRESSED), page0, tile=(PATCH_SIZE, PATCH_SIZE),
                     compression=None, photometric="minisblack",
                     ome=True, metadata={"axes": "YX"})
    tifffile.imwrite(str(OME_COMPRESSED), page0, tile=(PATCH_SIZE, PATCH_SIZE),
                     compression="zlib", photometric="minisblack",
                     ome=True, metadata={"axes": "YX"})
    tifffile.imwrite(str(TILED_PLAIN), page0, tile=(PATCH_SIZE, PATCH_SIZE),
                     compression=None, photometric="minisblack")
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
    n_patches = 0
    for batch in dl:
        n_patches += batch.shape[0]
    dt = time.perf_counter() - t0
    return dt, n_patches / dt


def verify(coords):
    """All methods must yield pixel-identical patches (check first 20)."""
    ref = PatchDataset("full", SRC, coords)
    checks = [("2", "zarr", OME_UNCOMPRESSED), ("3", "zarr", OME_COMPRESSED),
              ("4", "zarr", TILED_PLAIN),       ("5", "memmap", SRC)]
    for label, kind, path in checks:
        other = PatchDataset(kind, path, coords)
        ok = all(torch.equal(ref[i], other[i]) for i in range(20))
        print(f"[verify] test {label} matches test 1: {ok}")
        assert ok, f"test {label} produced different pixels"


def main():
    h, w = prepare_files()
    coords = build_coords(h, w, PATCH_SIZE, N_PATCHES, SEED)
    print(f"\n{N_PATCHES} patches, batch={BATCH_SIZE}, workers={NUM_WORKERS}\n")
    verify(coords)

    tests = [
        ("1. full   (original logic)",  "full",   SRC),
        ("2. zarr   (ome uncompressed)", "zarr",  OME_UNCOMPRESSED),
        ("3. zarr   (ome compressed)",   "zarr",  OME_COMPRESSED),
        ("4. zarr   (plain tiled)",      "zarr",  TILED_PLAIN),
        ("5. memmap (original)",         "memmap", SRC),
    ]

    print(f"\n{'method':30s} {'time (s)':>10s} {'patches/s':>12s}")
    print("-" * 54)
    for name, kind, path in tests:
        dt, rate = benchmark(kind, path, coords)
        print(f"{name:30s} {dt:10.3f} {rate:12.0f}")


if __name__ == "__main__":
    main()