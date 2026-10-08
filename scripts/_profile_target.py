"""Standalone entry point for profiling a single method with py-spy/memray.

Not meant to be run directly for normal benchmarking -- use run_benchmark.py
for that. This exists only so py-spy/memray profile exactly this open()+read
loop, with none of the CLI/plotting/verification code around it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dataloader_bench.config import BenchConfig  # noqa: E402
from dataloader_bench.methods import get as get_method  # noqa: E402
from dataloader_bench.verify import build_coords  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--method", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--n-patches", type=int, default=500)
    args = p.parse_args()

    cfg = BenchConfig(data_dir=Path(args.data_dir), patch_size=args.patch_size,
                       n_patches=args.n_patches)
    method = get_method(args.method, cfg)
    method.open()

    import tifffile
    h, w = tifffile.imread(str(cfg.src), key=0).shape[:2]
    coords = build_coords(h, w, cfg.patch_size, cfg.n_patches, cfg.seed)
    for y, x in coords:
        method.read_patch(y, x, cfg.patch_size)
    method.close()


if __name__ == "__main__":
    main()
