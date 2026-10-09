# dataloader-bench

How fast can random 256x256 patches be read from a large microscopy image
into a PyTorch `DataLoader`? This benchmarks OME-TIFF (`tifffile`, raw
memmap) against Zarr / OME-Zarr (`zarr-python` + `zarrs`, `dask`, `xarray`,
`SpatialData`, `tensorstore`, `miao`), on CPU only.

## The read-amplification effect

The source image is stored in on-disk chunks/tiles of `CHUNK` pixels
(default 512x512). Each requested patch is `PATCH_SIZE` pixels (default
256x256), placed at a random offset. A random 256px patch almost always
straddles a chunk boundary, so decoding it typically touches **more than
one** 512px chunk even though it only needs a quarter of each. This
over-fetch -- "read amplification" -- is a property of the chunk/patch size
ratio, not of any particular library, and it's one reason smaller-chunk /
better-aligned layouts can win even when they add per-chunk overhead.

## The method ladder

The point of comparing these specific methods, in this order, is to locate
*where* in the stack time goes -- each step adds one layer on top of the
previous one's cost:

| # | Method | What it isolates |
|---|---|---|
| 1 | `full` (cv2, whole image in RAM) | Upper bound / naive baseline |
| 2-5 | TIFF family (OME compressed/uncompressed, plain tiled, memmap) | Tile-based lazy decode vs. raw OS-level mmap; cost of compression; whether OME-XML metadata itself adds overhead |
| 6-7 | `zarr_direct` (compressed / uncompressed) | Raw scale-0 zarr array via zarr-python + the `zarrs` Rust codec pipeline -- floor of the Zarr-based path |
| 8-9 | `dask_zarr` | + `dask.array.from_zarr` overhead on top of (6-7) |
| 10-11 | `xr_dask` | + `xarray.DataArray` wrapping overhead on top of (8-9) |
| 12-13 | `spatialdata` (`read_zarr`) | + full SpatialData read-path overhead on top of (10-11) |
| 14 | `tensorstore_direct` | Plain tensorstore (C++) read of the same scale-0 array (compressed store) -- compare with 6 to separate the zarr *format* from the zarr-python *library*; also the base layer under miao |
| 15 | `miao` | + miao's own overhead on top of (14), both reading the same coordinates |

Compressed vs. uncompressed scale-0 arrays (6/7, 8/9, 10/11, 12/13) isolate
decompression cost specifically; everything else about those pairs is
identical.

## How to read the results

- **Open/connect time** and **read time** are reported and plotted
  separately. Opening a whole-image array into RAM (method 1) is *not*
  counted as part of its "read" time, and no method's handle-opening cost
  leaks into another method's numbers.
- Every number reflects a **warm OS page cache** unless you drop caches
  between runs yourself (`sync; echo 3 > /proc/sys/vm/drop_caches` on Linux;
  there's no exact equivalent on Windows -- reboot or use a tool like
  RAMMap). Cold-cache numbers for a network share can look very different.
- Each method runs `--warmup-runs` (default 1, discarded) then
  `--timed-runs` (default 5) times; the CSV/plots report the **median** and
  the **min-max spread** across those timed runs, not a single sample.
- `--num-workers 0` (the default) measures "open once, read N times" in a
  single process. With `--num-workers > 0`, each worker process must open
  its own handle; that per-worker open cost is measured and reported
  separately from read throughput, which is the only part worth comparing
  across methods at that point.
- A method that fails prints its error and is skipped; every other method
  still runs and gets plotted.

## Compressed vs. uncompressed, local vs. network storage

Compression trades CPU time (decode cost, paid on every read) for disk
space. Whether that trade is worth it depends on where the data lives: on a
fast local SSD, decode cost tends to dominate and uncompressed can win; on a
network share, the bytes saved by compression can matter more than the CPU
cost of decoding them, because network bandwidth / latency is usually the
bottleneck instead. `environment.json` (written on every run, see below)
records a best-effort guess at whether `--data-dir` is a local disk or a
network share, so results from different environments aren't silently
compared as if they were equivalent.

## Reproducing

```bash
pip install -r requirements.txt
python scripts/run_benchmark.py --data-dir ./data
```

Useful flags:

```bash
# use your own image (page 0 is read; any TIFF works)
python scripts/run_benchmark.py --data-dir ./data --src /path/to/your_image.tif

# CI / quick smoke test, no download, no real dataset
python scripts/run_benchmark.py --data-dir ./ci-data --synthetic --n-patches 20

# only run a subset of methods
python scripts/run_benchmark.py --data-dir ./data --methods full tiff_memmap miao

# read miao's own patch positions instead of the shared random ones (see "miao coordinates")
python scripts/run_benchmark.py --data-dir ./data --miao-coords

# ... using miao's random sampling mode instead of its default sequential grid
python scripts/run_benchmark.py --data-dir ./data --miao-coords --miao-sampling random
```

Before running against the real dataset, set the Dropbox direct-download
link in `src/dataloader_bench/config.py` (`BenchConfig.src_url`, `dl=1`) --
or just drop the source TIFF into `--data-dir` yourself and it'll be used
as-is.

`results.csv`, `results_throughput.png` and `results_open.png` are written to
`viz/` (override with `--viz-dir`). The derived files, the source image and
`environment.json` (python/numpy/zarr/zarrs/tifffile/dask/xarray/spatialdata/
tensorstore/torch versions, OS, CPU, and the local-disk-vs-network-share
guess) stay in `--data-dir`.

## Adding a new method

Add one file under `src/dataloader_bench/methods/`, implement `open()` +
`read_patch(y, x, patch_size)`, decorate the factory function with
`@register("your_name")`, import the module from `methods/__init__.py`, and
add `"your_name"` to `BenchConfig.methods`. That's the whole contract --
nothing else in the runner, verifier, or plotting code needs to change.

## miao coordinates

`miao.VolumeDataset` picks its own patch positions and doesn't accept
externally-chosen coordinates, so `--miao-coords` goes the other way: it
records the positions miao actually reads -- from the public
`sample["meta"]["coordinate"]`, converted from miao's centers to the
top-left `(y, x)` every other method takes -- and then runs every method
over that exact list, in that exact order. The pixel-identity check covers
every method, miao included, at those same coordinates.

`--miao-sampling` selects which miao mode is recorded: `sequential` (its
chunk-aligned grid, the default) or `random`. Random mode is made
reproducible by reseeding `np.random` before each run, so it requires
`--num-workers 0`.

## Results

One `--miao-coords` run against a Xenium Breast Cancer IF Image (page 0,
9777x14239 uint8): the first 500 positions of miao's sequential grid, batch
32, 1 warmup + 5 timed runs, single process, warm OS page cache -- see
caveats above. Full precision and per-run spread in `viz/results.csv`;
environment details in `environment.json`.

Hardware: AMD64 family 23 model 96 (AuthenticAMD), 12 logical cores,
Windows 11 (10.0.26200), local disk (not a network share), 173 GB free.
Python 3.11.17, torch 2.14.1+cpu, numpy 2.4.6, zarr 3.1.6, zarrs 0.2.3,
tifffile 2026.3.3, dask 2026.1.1, xarray 2026.9.0, spatialdata 0.7.3,
miao-io 0.4.2.

| Method | Family | open (s, median) | read (s, median) | patches/s (median) |
|---|---|---|---|---|
| full | baseline | 1.2712 | 0.0602 | 8,312 |
| tiff_ome_uncompressed | tiff | 0.0031 | 0.6655 | 751 |
| tiff_ome_compressed | tiff | 0.0026 | 0.8130 | 615 |
| tiff_plain_tiled | tiff | 0.0024 | 0.5988 | 835 |
| tiff_memmap | tiff | 0.0014 | 0.0994 | 5,028 |
| zarr_direct_compressed | zarr_ladder | 0.0058 | 0.9758 | 512 |
| zarr_direct_uncompressed | zarr_ladder | 0.0068 | 1.2702 | 394 |
| dask_zarr_compressed | zarr_ladder | 0.0186 | 2.8398 | 176 |
| dask_zarr_uncompressed | zarr_ladder | 0.0170 | 2.5914 | 193 |
| xr_dask_compressed | zarr_ladder | 0.0171 | 2.9374 | 170 |
| xr_dask_uncompressed | zarr_ladder | 0.0177 | 2.7474 | 182 |
| spatialdata_compressed | zarr_ladder | 0.0228 | 2.8556 | 175 |
| spatialdata_uncompressed | zarr_ladder | 0.0225 | 3.0618 | 163 |
| tensorstore_direct | miao_ladder | 0.0056 | 0.4865 | 1,028 |
| miao | miao_ladder | 0.0138 | 0.6048 | 827 |

![throughput](viz/results_throughput.png)
![open-time](viz/results_open.png)

