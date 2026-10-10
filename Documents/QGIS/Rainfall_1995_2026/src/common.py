
"""
Rwanda CHIRPS Rainfall Analysis (1995–2026)
------------------------------------------
Shared configuration, GIS, logging, dates, and raster utilities.

All paths are resolved relative to the project directory.
"""

from __future__ import annotations

import json
import logging
import sys

from datetime import date
from pathlib import Path
from typing import Iterator

import geopandas as gpd
import numpy as np
import rasterio

from rasterio.features import geometry_mask
from rasterio.warp import transform_geom


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CONFIG_PATH = PROJECT_ROOT / "config" / "settings.json"


def project_path(relative_path: str | Path) -> Path:
    """
    Resolve a relative path inside the project root.

    Example:
        project_path("data/boundaries/RWA_adm0.shp")
    """
    relative_path = Path(relative_path)

    if relative_path.is_absolute():
        raise ValueError(
            "Expected a project-relative path, "
            f"received: {relative_path}"
        )

    resolved = (PROJECT_ROOT / relative_path).resolve()

    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(
            f"Path must remain within project: {relative_path}"
        )

    return resolved


# ==========================================================
# 2. LOAD CONFIGURATION
# ==========================================================

def load_config(
    config_path: str | Path = CONFIG_PATH
) -> dict:
    """
    Load JSON configuration and check essential settings.
    """
    path = Path(config_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with path.open("r", encoding="utf-8-sig") as file:
        config = json.load(file)

    required_sections = [
        "project",
        "paths",
        "chirps",
        "spatial_analysis",
        "climatology",
        "seasons",
    ]

    missing = [
        key for key in required_sections
        if key not in config
    ]

    if missing:
        raise ValueError(
            "Missing configuration sections: "
            + ", ".join(missing)
        )

    start_year = int(config["project"]["start_year"])
    end_year = int(config["project"]["end_year"])

    if start_year > end_year:
        raise ValueError(
            "Project start_year cannot exceed end_year."
        )

    if not 1 <= int(
        config["project"]["last_requested_month"]
    ) <= 12:
        raise ValueError(
            "last_requested_month must be between 1 and 12."
        )

    for season, months in config["seasons"].items():
        if not months or any(
            not isinstance(m, int) or not 1 <= m <= 12
            for m in months
        ):
            raise ValueError(
                f"Invalid season definition: {season}"
            )

    return config


# ==========================================================
# 3. DIRECTORY MANAGEMENT
# ==========================================================

def ensure_directories(config: dict) -> None:
    """
    Ensure that all configured output directories exist.
    """
    directory_keys = [
        "raw_chirps",
        "processed",
        "climate_indices",
        "figures",
        "tables",
        "maps",
        "logs",
    ]

    for key in directory_keys:
        relative_path = config["paths"][key]

        directory = project_path(relative_path)
        directory.mkdir(
            parents=True,
            exist_ok=True
        )


# ==========================================================
# 4. LOGGING
# ==========================================================

def setup_logger(
    name: str = "rwanda_chirps",
    logfile: str = "rainfall_analysis.log",
) -> logging.Logger:
    """
    Configure logging to both terminal and file.
    """
    log_dir = project_path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    # Prevent duplicate handlers when called repeatedly
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_dir / logfile,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console)
    logger.addHandler(file_handler)

    return logger


# ==========================================================
# 5. RWANDA BOUNDARY
# ==========================================================

def load_rwanda_boundary(
    config: dict,
    target_crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Load Rwanda's national boundary, check its geometry,
    dissolve any multiple features, and reproject.
    """
    boundary_path = project_path(
        config["paths"]["boundary"]
    )

    if not boundary_path.exists():
        raise FileNotFoundError(
            f"Rwanda boundary not found: {boundary_path}"
        )

    gdf = gpd.read_file(boundary_path)

    if gdf.empty:
        raise ValueError(
            "Rwanda boundary dataset contains no features."
        )

    if gdf.crs is None:
        raise ValueError(
            "Rwanda boundary has no defined CRS. "
            "Assign the correct original CRS in QGIS."
        )

    gdf = gdf[gdf.geometry.notna()].copy()
    gdf = gdf[~gdf.geometry.is_empty].copy()

    if gdf.empty:
        raise ValueError(
            "No usable geometry found in Rwanda boundary."
        )

    # Repair invalid polygon geometry where possible
    gdf["geometry"] = gdf.geometry.make_valid()

    # Union all features into one national boundary
    national_geometry = gdf.geometry.union_all()

    if national_geometry.is_empty:
        raise ValueError(
            "The resulting Rwanda geometry is empty."
        )

    if national_geometry.geom_type not in (
        "Polygon",
        "MultiPolygon",
    ):
        raise ValueError(
            "Expected polygonal Rwanda boundary, got "
            f"{national_geometry.geom_type}"
        )

    national = gpd.GeoDataFrame(
        {"country": ["Rwanda"]},
        geometry=[national_geometry],
        crs=gdf.crs,
    )

    return national.to_crs(target_crs)


# ==========================================================
# 6. YEAR / MONTH ITERATOR
# ==========================================================

def iter_year_month(
    start_year: int,
    end_year: int,
    last_month: int = 12,
) -> Iterator[tuple[int, int]]:
    """
    Generate year/month pairs.

    The last_month restriction applies only to end_year.
    """
    if start_year > end_year:
        raise ValueError("Invalid year range.")

    if not 1 <= last_month <= 12:
        raise ValueError("last_month must be 1–12.")

    for year in range(start_year, end_year + 1):
        ending_month = (
            last_month if year == end_year else 12
        )

        for month in range(1, ending_month + 1):
            yield year, month


def is_future_month(year: int, month: int) -> bool:
    """
    Check whether a month is in the future.

    This does not guarantee CHIRPS data are published
    for a past or current month.
    """
    today = date.today()

    return (year, month) > (
        today.year, today.month
    )


# ==========================================================
# 7. CHIRPS FILENAME AND URL
# ==========================================================

def chirps_filename(
    year: int,
    month: int,
    config: dict,
) -> str:
    """
    Construct the CHIRPS monthly GeoTIFF filename.
    """
    pattern = config["chirps"]["filename_pattern"]

    return pattern.format(
        year=year,
        month=month,
    )


def chirps_url(
    year: int,
    month: int,
    config: dict,
) -> str:
    """
    Construct the download URL.
    """
    base = config["chirps"]["base_url"].rstrip("/")

    return (
        f"{base}/"
        f"{chirps_filename(year, month, config)}"
    )


# ==========================================================
# 8. AREA-WEIGHTED RASTER STATISTICS
# ==========================================================

def raster_area_weighted_mean(
    raster_path: str | Path,
    boundary: gpd.GeoDataFrame,
    all_touched: bool = False,
) -> float:
    """
    Calculate Rwanda's area-weighted mean precipitation
    from a raster clipped or masked by its boundary.

    Uses geodesic pixel-area weighting for geographic
    rasters with a regular north-up grid.

    Returns np.nan if no valid pixels are available.
    """
    from pyproj import Geod

    raster_path = Path(raster_path)

    with rasterio.open(raster_path) as src:

        if src.crs is None:
            raise ValueError(
                f"Raster has no CRS: {raster_path}"
            )

        if not src.crs.is_geographic:
            raise ValueError(
                "Area-weighted function requires "
                "a geographic raster CRS."
            )

        if src.transform.b != 0 or src.transform.d != 0:
            raise ValueError(
                "Rotated rasters are not supported."
            )

        data = src.read(1, masked=True)

        raster_boundary = boundary.to_crs(src.crs)

        mask = geometry_mask(
            raster_boundary.geometry,
            out_shape=(src.height, src.width),
            transform=src.transform,
            invert=True,
            all_touched=all_touched,
        )

        valid = (
            mask
            & ~np.ma.getmaskarray(data)
            & np.isfinite(data.data)
        )

        if not np.any(valid):
            return float("nan")

        rows, cols = np.where(valid)

        transform = src.transform
        geod = Geod(ellps="WGS84")

        unique_rows = np.unique(rows)
        row_area = {}

        for row in unique_rows:
            y_top = transform.f + row * transform.e
            y_bottom = y_top + transform.e

            x_left = transform.c
            x_right = x_left + transform.a

            area, _ = geod.polygon_area_perimeter(
                [x_left, x_right, x_right, x_left],
                [y_top, y_top, y_bottom, y_bottom],
            )

            row_area[int(row)] = abs(area)

        weights = np.array(
            [row_area[int(row)] for row in rows],
            dtype=np.float64,
        )

        values = data.data[rows, cols].astype(
            np.float64
        )

        return float(
            np.average(values, weights=weights)
        )


# ==========================================================
# 9. CONFIGURATION SUMMARY
# ==========================================================

def print_project_summary(config: dict) -> None:
    """
    Display current project configuration.
    """
    project = config["project"]

    print("=" * 62)
    print("RWANDA CHIRPS RAINFALL ANALYSIS")
    print("=" * 62)
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Country      : {project['country']}")
    print(
        f"Study period : {project['start_year']}"
        f"–{project['end_year']}"
    )
    print(
        f"CHIRPS       : "
        f"v{config['chirps']['version']}"
    )
    print(
        "Boundary     : "
        + str(
            project_path(config["paths"]["boundary"])
        )
    )
    print("=" * 62)


# ==========================================================
# 10. DIRECT EXECUTION / SMOKE TEST
# ==========================================================

def main() -> None:
    config = load_config()

    ensure_directories(config)
    logger = setup_logger()

    print_project_summary(config)

    logger.info("Loading Rwanda national boundary")

    boundary = load_rwanda_boundary(config)

    logger.info(
        "Boundary loaded: %s feature(s)",
        len(boundary),
    )
    logger.info(
        "Boundary CRS: %s",
        boundary.crs,
    )
    logger.info(
        "Boundary bounds: %s",
        boundary.total_bounds.tolist(),
    )

    sample_url = chirps_url(
        2023, 10, config
    )

    logger.info(
        "Example CHIRPS URL: %s",
        sample_url,
    )

    logger.info(
        "Project configuration validation successful"
    )


if __name__ == "__main__":
    main()
