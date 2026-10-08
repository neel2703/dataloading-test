#!/usr/bin/env python
"""Thin CLI entry point. Wires together prepare -> verify -> run -> save.

Examples:
  python scripts/run_benchmark.py --data-dir ./data
  python scripts/run_benchmark.py --data-dir ./data --synthetic           # CI / smoke test
  python scripts/run_benchmark.py --data-dir ./data --methods full tiff_memmap miao
  python scripts/run_benchmark.py --data-dir ./data --profile miao        # py-spy + memray only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import dask  # noqa: E402
dask.config.set(scheduler="synchronous")

import zarr  # noqa: E402
try:
    zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
except Exception as e:
    print(f"[warn] could not enable zarrs codec pipeline, falling back to default: {e}")

from dataloader_bench.config import BenchConfig  # noqa: E402
from dataloader_bench import envinfo, prepare as prepare_mod, plotting  # noqa: E402
from dataloader_bench.runner import run_all  # noqa: E402
from dataloader_bench.verify import build_coords, verify_all  # noqa: E402
from dataloader_bench.methods import get as get_method  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", required=True, type=Path,
                    help="working directory: derived files, results, and the downloaded "
                         "source image (unless --src is given) all live here")
    p.add_argument("--src", type=Path, default=None,
                    help="path to your own source TIFF (page 0 is used). Skips download_src() "
                         "entirely -- the Dropbox URL in config.py is never touched.")
    p.add_argument("--synthetic", action="store_true",
                    help="skip the real download, generate a small fake image (for CI)")
    p.add_argument("--patch-size", type=int, default=256)
    p.add_argument("--chunk-size", type=int, default=512)
    p.add_argument("--n-patches", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--warmup-runs", type=int, default=1)
    p.add_argument("--timed-runs", type=int, default=5)
    p.add_argument("--methods", nargs="*", default=None,
                    help="subset of methods to run (default: all registered)")
    p.add_argument("--no-verify", action="store_true")
    p.add_argument("--profile", metavar="METHOD", default=None,
                    help="run py-spy/memray on a single method instead of the full sweep")
    return p.parse_args()


def build_config(args: argparse.Namespace) -> BenchConfig:
    cfg = BenchConfig(
        data_dir=args.data_dir, src_path=args.src, synthetic=args.synthetic,
        patch_size=args.patch_size, chunk_size=args.chunk_size,
        n_patches=args.n_patches, batch_size=args.batch_size,
        num_workers=args.num_workers, seed=args.seed,
        warmup_runs=args.warmup_runs, timed_runs=args.timed_runs,
    )
    if args.methods:
        cfg.methods = args.methods
    return cfg


def main() -> None:
    args = parse_args()
    cfg = build_config(args)
    cfg.data_dir.mkdir(parents=True, exist_ok=True)

    env = envinfo.save(cfg.data_dir, cfg.env_json)
    print(f"[env] python={env['python'].split()[0]} os={env['os']} cpu_count={env['cpu_count']}")
    print(f"[env] network share: {env['data_dir_is_network_share']}")

    h, w = prepare_mod.prepare(cfg)
    coords = build_coords(h, w, cfg.patch_size, cfg.n_patches, cfg.seed)

    if args.profile:
        from dataloader_bench.profile import run_profile
        run_profile(args.profile, cfg, cfg.data_dir / "profiles")
        return

    if not args.no_verify:
        ref = get_method("full", cfg)
        others = {}
        for name in cfg.methods:
            if name == "full":
                continue
            others[name] = get_method(name, cfg)
        verify_all(cfg, ref, others, coords)

    results = run_all(cfg, coords)
    plotting.save_csv(cfg, results)
    plotting.save_plots(cfg, results)


if __name__ == "__main__":
    main()
