"""Single place for every tunable knob in the benchmark."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class BenchConfig:
    # --- paths -----------------------------------------------------------
    data_dir: Path
    src_path: Path | None = None   # point this at your own TIFF to skip download_src() entirely
    src_name: str = "Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif"
    # Direct-download Dropbox link (dl=1). Paste the real link before running
    # against the real dataset; leave as-is and pass --synthetic for CI / a
    # quick smoke test, which never touches this URL.
    src_url: str = (
        "https://www.dropbox.com/scl/fi/7ro751l65srv1il4zwpgi/"
        "Xenium_FFPE_Human_Breast_Cancer_Rep1_if_image.tif"
        "?rlkey=j6atckxarz2z0zvctnmvf40qn&e=1&dl=1"
    )

    # --- benchmark shape ---------------------------------------------------
    patch_size: int = 256
    chunk_size: int = 512          # on-disk tile/chunk size for every derived file
    n_patches: int = 500
    batch_size: int = 32
    num_workers: int = 0           # 0 = open() cost is measured once, in-process
    seed: int = 0

    # --- repeats ------------------------------------------------------------
    warmup_runs: int = 1
    timed_runs: int = 5

    # --- misc ---------------------------------------------------------------
    image_name: str = "image"      # name the spatialdata image is stored under
    synthetic: bool = False        # True -> skip download, generate a small fake image
    synthetic_shape: tuple[int, int] = (2048, 2048)

    methods: list[str] = field(default_factory=lambda: [
        "full",
        "tiff_ome_uncompressed",
        "tiff_ome_compressed",
        "tiff_plain_tiled",
        "tiff_memmap",
        "zarr_direct_compressed",
        "zarr_direct_uncompressed",
        "dask_zarr_compressed",
        "dask_zarr_uncompressed",
        "xr_dask_compressed",
        "xr_dask_uncompressed",
        "spatialdata_compressed",
        "spatialdata_uncompressed",
        "tensorstore_direct",
        "miao",
    ])

    # --- derived paths --------------------------------------------------
    @property
    def src(self) -> Path:
        if self.src_path is not None:
            return Path(self.src_path)
        return self.data_dir / self.src_name

    @property
    def ome_uncompressed(self) -> Path:
        return self.data_dir / "page0_uncompressed.ome.tif"

    @property
    def ome_compressed(self) -> Path:
        return self.data_dir / "page0_compressed.ome.tif"

    @property
    def tiled_plain(self) -> Path:
        return self.data_dir / "page0_tiled_plain.tif"

    @property
    def sdata_input(self) -> Path:
        return self.data_dir / "page0_cyx.ome.tif"

    @property
    def sdata_compressed(self) -> Path:
        return self.data_dir / "page0_sdata_compressed.zarr"

    @property
    def sdata_uncompressed(self) -> Path:
        return self.data_dir / "page0_sdata_uncompressed.zarr"

    @property
    def results_csv(self) -> Path:
        return self.data_dir / "results.csv"

    @property
    def results_png(self) -> Path:
        return self.data_dir / "results_throughput.png"

    @property
    def results_open_png(self) -> Path:
        return self.data_dir / "results_open.png"

    @property
    def env_json(self) -> Path:
        return self.data_dir / "environment.json"
