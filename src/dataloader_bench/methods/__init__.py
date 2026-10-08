"""Importing this package registers every method (each submodule calls
`@register(...)` at import time). `get(name, cfg)` is the only thing callers
need."""

from __future__ import annotations

from ..config import BenchConfig
from .base import REGISTRY, Method, MethodError  # noqa: F401
from . import (  # noqa: F401
    full,
    tiff,
    zarr_direct,
    dask_zarr,
    xr_dask,
    spatialdata_method,
    tensorstore_method,
    miao_method,
)


def get(name: str, cfg: BenchConfig) -> Method:
    try:
        factory = REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown method {name!r}; registered: {sorted(REGISTRY)}")
    return factory(cfg)
