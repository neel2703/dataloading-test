"""CSV + grouped bar charts (read throughput, open time), error bars from
the repeated runs, methods grouped by family (TIFF / Zarr ladder / miao)."""

from __future__ import annotations

import csv

from .config import BenchConfig
from .runner import MethodResult

FAMILY_ORDER = ["baseline", "tiff", "zarr_ladder", "miao_ladder", "other"]
FAMILY_COLOR = {
    "baseline": "tab:gray",
    "tiff": "tab:blue",
    "zarr_ladder": "tab:green",
    "miao_ladder": "tab:purple",
    "other": "tab:orange",
}


def save_csv(cfg: BenchConfig, results: list[MethodResult]) -> None:
    rows = [r.summary() for r in results]
    with open(cfg.results_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        w.writeheader()
        w.writerows(rows)
    print(f"[csv] saved {cfg.results_csv.name}")


def save_plots(cfg: BenchConfig, results: list[MethodResult]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed, skipping charts")
        return

    ok_results = [r for r in results if r.ok]
    if not ok_results:
        print("[plot] no successful methods to plot")
        return

    ordered = sorted(ok_results, key=lambda r: (FAMILY_ORDER.index(r.family)
                                                 if r.family in FAMILY_ORDER else len(FAMILY_ORDER), r.name))
    names = [r.name for r in ordered]
    colors = [FAMILY_COLOR.get(r.family, "tab:orange") for r in ordered]

    def plot_one(values_key, spread_key, xlabel, out_path):
        summaries = [r.summary() for r in ordered]
        vals = [s[values_key] for s in summaries]
        errs = [s[spread_key] for s in summaries]
        fig, ax = plt.subplots(figsize=(9, 0.4 * len(names) + 1.5))
        ax.barh(names, vals, xerr=errs, color=colors)
        ax.set_xlabel(xlabel)
        ax.invert_yaxis()
        fig.tight_layout()
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[plot] saved {out_path.name}")

    plot_one("patches_per_s_median", "patches_per_s_spread",
              "patches/s (read only, median +/- range)", cfg.results_png)
    plot_one("open_s_median", "open_s_spread",
              "open / connect time (s, median +/- range)", cfg.results_open_png)

    failed = [r.name for r in results if not r.ok]
    if failed:
        print(f"[plot] skipped failed methods: {failed}")
