"""Runs every configured method through a DataLoader, timing open() and
read() separately, repeating with warmup, and isolating failures so one
broken method doesn't kill the rest of the run."""

from __future__ import annotations

import multiprocessing
import statistics
import time
import traceback
from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader, Dataset

from .config import BenchConfig
from .methods import get as get_method
from .methods.miao_method import MiaoVolume


class _CoordPatchDataset(Dataset):
    """Wraps a coordinate-driven Method. open() is called once up front when
    num_workers==0 (handle shared via the already-open object); when
    num_workers>0 each worker process opens its own handle lazily on first
    access and records how long that took into `open_times` (an mp.Manager
    list), since the handle can't be pickled from the main process."""

    def __init__(self, method_name: str, cfg: BenchConfig, coords, open_times=None):
        self.method_name = method_name
        self.cfg = cfg
        self.coords = coords
        self.open_times = open_times
        self._method = None

    def _ensure_open(self):
        if self._method is None:
            self._method = get_method(self.method_name, self.cfg)
            t0 = time.perf_counter()
            self._method.open()
            if self.open_times is not None:
                self.open_times.append(time.perf_counter() - t0)

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, i):
        self._ensure_open()
        y, x = self.coords[i]
        patch = self._method.read_patch(y, x, self.cfg.patch_size)
        return torch.from_numpy(patch.copy()).unsqueeze(0)


class _MiaoIndexDataset(Dataset):
    """miao is index-driven (sequential grid) -- see methods/miao_method.py."""

    def __init__(self, cfg: BenchConfig, n_items: int, open_times=None):
        self.cfg = cfg
        self.n_items = n_items
        self.open_times = open_times
        self._method: MiaoVolume | None = None

    def _ensure_open(self):
        if self._method is None:
            self._method = get_method("miao", self.cfg)
            t0 = time.perf_counter()
            self._method.open()
            if self.open_times is not None:
                self.open_times.append(time.perf_counter() - t0)

    def __len__(self):
        self._ensure_open()
        return min(self.n_items, len(self._method))

    def __getitem__(self, i):
        self._ensure_open()
        patch = self._method.read_patch_by_index(i)  # already (ph, ph) float32, via to_float01
        return torch.from_numpy(patch.copy()).unsqueeze(0)


@dataclass
class MethodResult:
    name: str
    family: str
    open_s: list[float] = field(default_factory=list)
    read_s: list[float] = field(default_factory=list)
    patches_per_s: list[float] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None

    def summary(self) -> dict:
        def med_spread(xs):
            if not xs:
                return (float("nan"), float("nan"))
            return (statistics.median(xs), (max(xs) - min(xs)) if len(xs) > 1 else 0.0)
        o_med, o_spread = med_spread(self.open_s)
        r_med, r_spread = med_spread(self.read_s)
        p_med, p_spread = med_spread(self.patches_per_s)
        return dict(method=self.name, family=self.family,
                    open_s_median=round(o_med, 4), open_s_spread=round(o_spread, 4),
                    read_s_median=round(r_med, 4), read_s_spread=round(r_spread, 4),
                    patches_per_s_median=round(p_med, 1), patches_per_s_spread=round(p_spread, 1),
                    error=self.error)


def _one_run(method_name: str, cfg: BenchConfig, coords) -> tuple[float, float, float]:
    """Single warmup/timed run. Returns (open_s, read_s, patches_per_s).
    open_s is NaN when num_workers>0 (reported separately, per-worker)."""
    workers = cfg.num_workers
    manager = multiprocessing.Manager() if workers > 0 else None
    open_times = manager.list() if manager else None

    if method_name == "miao":
        ds = _MiaoIndexDataset(cfg, cfg.n_patches, open_times=open_times)
    else:
        ds = _CoordPatchDataset(method_name, cfg, coords, open_times=open_times)

    t_open = float("nan")
    if workers == 0:
        # open() once, up front, in the main process; the DataLoader below
        # reuses this same already-open handle for every read.
        t0 = time.perf_counter()
        ds._ensure_open()
        t_open = time.perf_counter() - t0

    dl = DataLoader(ds, batch_size=cfg.batch_size, num_workers=workers,
                     shuffle=False, pin_memory=False)

    t0 = time.perf_counter()
    n = 0
    for batch in dl:
        n += batch.shape[0]
    t_read = time.perf_counter() - t0

    if workers > 0 and open_times is not None and len(open_times) > 0:
        t_open = statistics.median(open_times)

    return t_open, t_read, n / t_read if t_read > 0 else float("nan")


def run_method(method_name: str, family: str, cfg: BenchConfig, coords) -> MethodResult:
    result = MethodResult(name=method_name, family=family)
    try:
        total_runs = cfg.warmup_runs + cfg.timed_runs
        for run_i in range(total_runs):
            t_open, t_read, rate = _one_run(method_name, cfg, coords)
            if run_i >= cfg.warmup_runs:
                result.open_s.append(t_open)
                result.read_s.append(t_read)
                result.patches_per_s.append(rate)
    except Exception:
        result.error = traceback.format_exc(limit=4)
    return result


def _family_of(name: str, cfg: BenchConfig) -> str:
    """Ask the method's own class for its family, so this never drifts out of
    sync with the `family` attribute declared in methods/*.py."""
    try:
        return get_method(name, cfg).family
    except Exception:
        return "other"


def run_all(cfg: BenchConfig, coords) -> list[MethodResult]:
    print(f"\n{cfg.n_patches} patches, batch={cfg.batch_size}, workers={cfg.num_workers}, "
          f"warmup={cfg.warmup_runs}, timed={cfg.timed_runs}\n")
    print("NOTE: unless you drop OS page caches between runs, these numbers reflect a warm cache.\n")

    results = []
    for name in cfg.methods:
        family = _family_of(name, cfg)
        print(f"[run] {name} ...")
        res = run_method(name, family, cfg, coords)
        if res.ok:
            s = res.summary()
            print(f"  open={s['open_s_median']:.3f}s (+/-{s['open_s_spread']:.3f})  "
                  f"read={s['read_s_median']:.3f}s (+/-{s['read_s_spread']:.3f})  "
                  f"{s['patches_per_s_median']:.0f} patches/s")
        else:
            print(f"  FAILED: {res.error.splitlines()[-1] if res.error else 'unknown error'}")
        results.append(res)
    return results
