
"""
01_collect_chirps.py
Rwanda CHIRPS Rainfall Analysis (1995–2026)

Download monthly CHIRPS v3 Africa GeoTIFFs,
clip to Rwanda, and save georeferenced outputs.

Example:
    python src/01_collect_chirps.py --start-year 2023 --end-year 2023
"""

from __future__ import annotations

import argparse
import csv
import logging
import time
from datetime import date
from pathlib import Path

import numpy as np
import rasterio
import requests

from rasterio.mask import mask
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from common import (
    PROJECT_ROOT,
    load_config,
    ensure_directories,
    setup_logger,
    load_rwanda_boundary,
    iter_year_month,
    is_future_month,
    chirps_filename,
    chirps_url,
    project_path,
)


def build_session(retries: int) -> requests.Session:
    """Create an HTTP session with retry support."""

    retry = Retry(
        total=retries,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )

    session = requests.Session()
    session.headers.update({
        "User-Agent": "Rwanda-CHIRPS-Research/1.0"
    })

    adapter = HTTPAdapter(max_retries=retry)
    session.mount("https://", adapter)
    session.mount("http://", adapter)

    return session


def download_file(
    session: requests.Session,
    url: str,
    destination: Path,
    timeout: int,
    logger: logging.Logger,
) -> bool:
    """
    Download a GeoTIFF safely using a temporary file.

    Return False when the server returns HTTP 404.
    Other HTTP/network problems raise an exception.
    """

    destination.parent.mkdir(
        parents=True, exist_ok=True
    )

    temporary = destination.with_suffix(".tif.part")

    try:
        with session.get(
            url,
            stream=True,
            timeout=(30, timeout),
        ) as response:

            if response.status_code == 404:
                logger.warning(
                    "Not available (HTTP 404): %s", url
                )
                return False

            response.raise_for_status()

            with temporary.open("wb") as output:
                for chunk in response.iter_content(
                    chunk_size=1024 * 1024
                ):
                    if chunk:
                        output.write(chunk)

        if temporary.stat().st_size == 0:
            raise ValueError(
                "Downloaded file is empty."
            )

        # Verify that the file really is a readable raster.
        with rasterio.open(temporary) as src:
            if src.count < 1 or src.crs is None:
                raise ValueError(
                    "Downloaded raster is invalid."
                )

        temporary.replace(destination)
        return True

    finally:
        if temporary.exists():
            temporary.unlink()


def clip_to_rwanda(
    input_file: Path,
    output_file: Path,
    boundary,
    all_touched: bool,
) -> dict:
    """
    Clip CHIRPS raster to Rwanda.

    Keep rainfall in millimetres and preserve
    nodata information.
    """

    output_file.parent.mkdir(
        parents=True, exist_ok=True
    )

    temporary = output_file.with_suffix(".tif.part")

    try:
        with rasterio.open(input_file) as src:

            if src.crs is None:
                raise ValueError(
                    "Input CHIRPS raster has no CRS."
                )

            projected_boundary = boundary.to_crs(
                src.crs
            )

            clipped, transform = mask(
                src,
                projected_boundary.geometry,
                crop=True,
                filled=False,
                all_touched=all_touched,
            )

            # Use a consistent floating-point nodata value.
            nodata = -9999.0

            values = clipped.filled(nodata).astype(
                "float32"
            )

            valid = (
                ~np.ma.getmaskarray(clipped[0])
                & np.isfinite(clipped.data[0])
            )

            if not np.any(valid):
                raise ValueError(
                    "No valid CHIRPS pixels inside Rwanda."
                )

            rainfall = clipped.data[0][valid]

            if np.any(rainfall < 0):
                raise ValueError(
                    "Negative precipitation values found."
                )

            profile = src.profile.copy()

            profile.update(
                driver="GTiff",
                height=values.shape[1],
                width=values.shape[2],
                count=values.shape[0],
                transform=transform,
                dtype="float32",
                nodata=nodata,
                compress="deflate",
                predictor=3,
                tiled=False,
            )

            with rasterio.open(
                temporary, "w", **profile
            ) as dst:
                dst.write(values)
                dst.update_tags(
                    source="CHIRPS v3.0",
                    country="Rwanda",
                    variable="monthly_precipitation",
                    units="mm",
                    spatial_processing="Rwanda boundary mask",
                )

        # Check the saved output.
        with rasterio.open(temporary) as check:
            if check.width == 0 or check.height == 0:
                raise ValueError(
                    "Output raster has invalid dimensions."
                )

        temporary.replace(output_file)

        return {
            "valid_pixels": int(valid.sum()),
            "minimum_mm": float(np.min(rainfall)),
            "maximum_mm": float(np.max(rainfall)),
            "mean_unweighted_mm": float(np.mean(rainfall)),
        }

    finally:
        if temporary.exists():
            temporary.unlink()


def process_month(
    year: int,
    month: int,
    config: dict,
    boundary,
    session: requests.Session,
    logger: logging.Logger,
    force: bool = False,
) -> dict:
    """Download and clip one month."""

    filename = chirps_filename(year, month, config)

    raw_dir = project_path(
        config["paths"]["raw_chirps"]
    )
    processed_dir = project_path(
        config["paths"]["processed"]
    )

    raw_file = raw_dir / filename

    output_file = processed_dir / (
        f"RWA_CHIRPS_{year}_{month:02d}.tif"
    )

    record = {
        "year": year,
        "month": month,
        "status": "",
        "output_file": str(
            output_file.relative_to(PROJECT_ROOT)
        ),
        "valid_pixels": "",
        "minimum_mm": "",
        "maximum_mm": "",
        "message": "",
    }

    if output_file.exists() and not force:
        try:
            with rasterio.open(output_file) as existing:
                if (
                    existing.crs is not None
                    and existing.count == 1
                    and existing.width > 0
                    and existing.height > 0
                    and existing.read(1, masked=True).count() > 0
                ):
                    logger.info(
                        "SKIP existing: %s", output_file.name
                    )
                    record["status"] = "skipped"
                    return record

        except Exception:
            logger.warning(
                "Existing raster is unreadable; "
                "regenerating: %s",
                output_file.name,
            )

    if is_future_month(year, month):
        record["status"] = "future"
        record["message"] = (
            "Month has not yet started."
        )
        return record

    # Avoid treating the current, potentially incomplete
    # calendar month as a finished monthly total.
    today = date.today()
    if (year, month) == (today.year, today.month):
        record["status"] = "current_month"
        record["message"] = (
            "Current month excluded until complete."
        )
        return record

    url = chirps_url(year, month, config)

    logger.info(
        "Processing %04d-%02d",
        year, month
    )

    if force or not raw_file.exists():
        available = download_file(
            session=session,
            url=url,
            destination=raw_file,
            timeout=int(
                config["chirps"]["download_timeout_seconds"]
            ),
            logger=logger,
        )

        if not available:
            record["status"] = "unavailable"
            record["message"] = "HTTP 404"
            return record

    try:
        stats = clip_to_rwanda(
            input_file=raw_file,
            output_file=output_file,
            boundary=boundary,
            all_touched=bool(
                config["spatial_analysis"]["all_touched"]
            ),
        )

    except Exception:
        # Preserve the raw file for inspection.
        raise

    record["status"] = "downloaded"
    record["valid_pixels"] = stats["valid_pixels"]
    record["minimum_mm"] = round(
        stats["minimum_mm"], 3
    )
    record["maximum_mm"] = round(
        stats["maximum_mm"], 3
    )

    logger.info(
        "SAVED %s | min %.2f mm | max %.2f mm",
        output_file.name,
        stats["minimum_mm"],
        stats["maximum_mm"],
    )

    return record


def save_manifest(records: list[dict], path: Path):
    """Save an audit table of attempted months."""

    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "year",
        "month",
        "status",
        "output_file",
        "valid_pixels",
        "minimum_mm",
        "maximum_mm",
        "message",
    ]

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(records)


def main():
    parser = argparse.ArgumentParser(
        description="Download CHIRPS monthly rainfall for Rwanda"
    )

    parser.add_argument(
        "--start-year", type=int, default=None
    )
    parser.add_argument(
        "--end-year", type=int, default=None
    )
    parser.add_argument(
        "--last-month", type=int, default=None
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Redownload and reprocess existing files"
    )

    args = parser.parse_args()

    config = load_config()
    ensure_directories(config)

    logger = setup_logger(
        name="chirps_download",
        logfile="chirps_download.log",
    )

    start_year = (
        args.start_year
        if args.start_year is not None
        else int(config["project"]["start_year"])
    )

    end_year = (
        args.end_year
        if args.end_year is not None
        else int(config["project"]["end_year"])
    )

    last_month = (
        args.last_month
        if args.last_month is not None
        else (
            int(config["project"]["last_requested_month"])
            if end_year == int(config["project"]["end_year"])
            else 12
        )
    )

    # Validate arguments before running.
    if start_year > end_year:
        parser.error(
            "start-year cannot exceed end-year"
        )

    if not 1 <= last_month <= 12:
        parser.error(
            "last-month must be from 1 to 12"
        )

    logger.info(
        "CHIRPS Rwanda download: %d to %d, "
        "last month %02d",
        start_year, end_year, last_month,
    )

    boundary = load_rwanda_boundary(config)

    records = []

    session = build_session(
        retries=int(
            config["chirps"]["download_retries"]
        )
    )

    try:
        for year, month in iter_year_month(
            start_year, end_year, last_month
        ):

            try:
                record = process_month(
                    year=year,
                    month=month,
                    config=config,
                    boundary=boundary,
                    session=session,
                    logger=logger,
                    force=args.force,
                )

            except Exception as exc:
                logger.exception(
                    "FAILED: %04d-%02d", year, month
                )

                record = {
                    "year": year,
                    "month": month,
                    "status": "failed",
                    "output_file": "",
                    "valid_pixels": "",
                    "minimum_mm": "",
                    "maximum_mm": "",
                    "message": str(exc),
                }

            records.append(record)

            # Update progress manifest after every month.
            manifest = project_path(
                config["paths"]["tables"]
            ) / (
                f"chirps_download_manifest_"
                f"{start_year}_{end_year}.csv"
            )

            save_manifest(records, manifest)

    finally:
        session.close()

    counts = {}
    for record in records:
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1

    logger.info("Download summary: %s", counts)

    if counts.get("failed", 0) > 0:
        raise SystemExit(
            "Some months failed. Check the manifest and logs."
        )


if __name__ == "__main__":
    main()
