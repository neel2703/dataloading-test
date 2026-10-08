"""Collect environment info so results.csv is reproducible months later."""

from __future__ import annotations

import json
import os
import platform
import shutil
import sys
from pathlib import Path
from typing import Any


def _version(module_name: str) -> str | None:
    try:
        mod = __import__(module_name)
    except ImportError:
        return None
    return getattr(mod, "__version__", "unknown")


def _cpu_info() -> str:
    try:
        return platform.processor() or platform.uname().processor or "unknown"
    except Exception:
        return "unknown"


def _is_network_path(path: Path) -> bool | None:
    """Best-effort local-disk vs. network-share detection. Returns None if unknown."""
    path = path.resolve()
    if os.name == "nt":
        drive = path.drive
        if not drive:
            return True  # UNC path (\\server\share\...) with no drive letter
        try:
            import ctypes
            DRIVE_REMOTE = 4
            kind = ctypes.windll.kernel32.GetDriveTypeW(f"{drive}\\")
            return kind == DRIVE_REMOTE
        except Exception:
            return None
    else:
        # Linux/macOS: look for the mount's filesystem type among common network ones.
        try:
            import subprocess
            out = subprocess.run(["df", "-T", str(path)], capture_output=True, text=True, timeout=5)
            fstype = out.stdout.splitlines()[-1].split()[1].lower()
            return fstype in {"nfs", "nfs4", "cifs", "smbfs", "smb", "afpfs"}
        except Exception:
            return None


def collect(data_dir: Path) -> dict[str, Any]:
    packages = [
        "numpy", "zarr", "zarrs", "tifffile", "dask", "xarray",
        "spatialdata", "spatialdata_io", "tensorstore", "torch", "miao",
    ]
    info: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "os": platform.system(),
        "cpu": _cpu_info(),
        "cpu_count": os.cpu_count(),
        "packages": {name: _version(name) for name in packages},
        "data_dir": str(data_dir),
        "data_dir_is_network_share": _is_network_path(data_dir),
        "disk_free_gb": round(shutil.disk_usage(data_dir).free / 1e9, 2) if data_dir.exists() else None,
    }
    return info


def save(data_dir: Path, out_path: Path) -> dict[str, Any]:
    info = collect(data_dir)
    out_path.write_text(json.dumps(info, indent=2))
    return info
