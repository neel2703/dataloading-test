"""Optional profiling mode: run ONE method under py-spy (flamegraph SVG) and
memray (memory profile), instead of the full benchmark sweep.

Modeled on the approach the spatialdata maintainers document in
https://github.com/scverse/spatialdata/tree/main/.claude/skills -- out-of-
process py-spy attached to a child running just the target method, plus a
separate memray run (`python -m memray run`) since the two don't compose
cleanly in one process.

py-spy and memray are NOT in requirements.txt by default (commented out) --
install them yourself (`pip install py-spy memray`) before using this mode.
memray has no Windows wheel; the py-spy half still works there.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

from .config import BenchConfig


def _check(tool: str) -> bool:
    return shutil.which(tool) is not None


def _runner_script_path() -> Path:
    # scripts/_profile_target.py -- a tiny standalone entry point py-spy/memray
    # attach to, so we're profiling exactly the open()+read loop and nothing
    # of this CLI's own startup cost.
    return Path(__file__).resolve().parent.parent.parent / "scripts" / "_profile_target.py"


def run_profile(method_name: str, cfg: BenchConfig, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    target = _runner_script_path()
    if not target.exists():
        raise FileNotFoundError(f"expected profiling entry point at {target}")

    args_common = [sys.executable, str(target), "--method", method_name,
                    "--data-dir", str(cfg.data_dir), "--patch-size", str(cfg.patch_size),
                    "--n-patches", str(cfg.n_patches)]

    if _check("py-spy"):
        svg_path = out_dir / f"{method_name}_flamegraph.svg"
        cmd = ["py-spy", "record", "-o", str(svg_path), "--"] + args_common
        print(f"[profile] py-spy: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
    else:
        print("[profile] py-spy not found on PATH -- `pip install py-spy` to enable flamegraphs")

    if _check("memray") or _module_available("memray"):
        bin_path = out_dir / f"{method_name}.memray.bin"
        cmd = [sys.executable, "-m", "memray", "run", "-o", str(bin_path)] + args_common[1:]
        print(f"[profile] memray: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)
        print(f"[profile] generate an HTML report with: python -m memray flamegraph {bin_path}")
    else:
        print("[profile] memray not available (no Windows wheel; Linux/macOS only) -- skipping")


def _module_available(name: str) -> bool:
    try:
        __import__(name)
        return True
    except ImportError:
        return False
