
"""
Rwanda CHIRPS Rainfall Analysis (1995–2026)
STEP 6 — Build monthly, annual, seasonal and
dry-to-wet transition time series.

Input:
    data/processed/RWA_CHIRPS_YYYY_MM.tif

Outputs:
    outputs/tables/*.csv

Run:
    python src/02_build_series.py
"""

from __future__ import annotations

import argparse
import calendar
import re
from pathlib import Path

import numpy as np
import pandas as pd

from common import (
    load_config,
    ensure_directories,
    setup_logger,
    load_rwanda_boundary,
    project_path,
    raster_area_weighted_mean,
)

PATTERN = re.compile(
    r"^RWA_CHIRPS_(\d{4})_(\d{2})\.tif$"
)


def get_periods(config):
    project = config["project"]
    climate = config["climatology"]

    start = int(project["start_year"])
    end = int(project["end_year"])
    ref_start = int(climate["reference_start_year"])
    ref_end = int(climate["reference_end_year"])

    if not (start <= ref_start <= ref_end <= end):
        raise ValueError(
            "Climatology period must lie inside study period."
        )

    return start, end, ref_start, ref_end


def save_csv(df, filename, output_dir):
    path = output_dir / filename
    df.to_csv(
        path,
        index=False,
        encoding="utf-8-sig",
        float_format="%.6f",
    )
    return path


def build_monthly(config, boundary, logger):
    """Calculate national rainfall for each monthly raster."""

    raw_dir = project_path(config["paths"]["processed"])
    start, end, _, _ = get_periods(config)

    raster_lookup = {}

    for path in sorted(raw_dir.glob("RWA_CHIRPS_*.tif")):
        match = PATTERN.fullmatch(path.name)
        if not match:
            continue

        year, month = map(int, match.groups())

        if start <= year <= end and 1 <= month <= 12:
            raster_lookup[(year, month)] = path

    rows = []
    all_touched = bool(
        config["spatial_analysis"]["all_touched"]
    )

    for year in range(start, end + 1):
        for month in range(1, 13):
            path = raster_lookup.get((year, month))

            value = np.nan
            status = "missing"

            if path is not None:
                try:
                    value = raster_area_weighted_mean(
                        path,
                        boundary,
                        all_touched=all_touched,
                    )
                    status = (
                        "available"
                        if np.isfinite(value)
                        else "no_valid_pixels"
                    )
                except Exception as exc:
                    status = "error"
                    logger.error(
                        "%04d-%02d: %s",
                        year, month, exc
                    )

            rows.append({
                "year": year,
                "month": month,
                "month_name": calendar.month_abbr[month],
                "date": f"{year}-{month:02d}-01",
                "rainfall_mm": value,
                "status": status,
            })

    df = pd.DataFrame(rows)
    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"], errors="coerce"
    )

    if (df["rainfall_mm"] < 0).any():
        raise ValueError(
            "Negative monthly rainfall totals detected."
        )

    logger.info(
        "Monthly observations available: %d / %d",
        df["rainfall_mm"].notna().sum(),
        len(df),
    )

    return df


def build_monthly_climatology(
    monthly, ref_start, ref_end
):
    """Calculate climatological mean and percentiles."""

    reference = monthly[
        monthly["year"].between(ref_start, ref_end)
    ].copy()

    expected_years = ref_end - ref_start + 1

    group = reference.groupby("month")["rainfall_mm"]

    clim = group.agg(
        mean_mm="mean",
        std_mm="std",
        n_years="count",
    ).reset_index()

    for percentile in [10, 20, 50, 80, 90]:
        clim[f"p{percentile}_mm"] = (
            group.quantile(percentile / 100).values
        )

    if (clim["n_years"] != expected_years).any():
        raise ValueError(
            "Reference climatology has missing monthly data."
        )

    return clim


def add_monthly_anomalies(monthly, clim):
    df = monthly.merge(
        clim[["month", "mean_mm"]],
        on="month",
        how="left",
        validate="many_to_one",
    )

    df = df.rename(
        columns={"mean_mm": "climatology_mm"}
    )

    df["anomaly_mm"] = (
        df["rainfall_mm"] - df["climatology_mm"]
    )

    df["anomaly_pct"] = np.where(
        df["climatology_mm"] > 0,
        100.0 * df["anomaly_mm"] /
        df["climatology_mm"],
        np.nan,
    )

    return df


def build_annual(monthly, ref_start, ref_end):
    """Create annual totals only for complete years."""

    annual = monthly.groupby("year").agg(
        months_available=("rainfall_mm", "count"),
        rainfall_sum_mm=("rainfall_mm", "sum"),
    ).reset_index()

    annual["complete"] = (
        annual["months_available"] == 12
    )

    annual.loc[
        ~annual["complete"], "rainfall_sum_mm"
    ] = np.nan

    reference = annual[
        annual["year"].between(ref_start, ref_end)
    ]

    if not reference["complete"].all():
        raise ValueError(
            "Incomplete reference year in annual analysis."
        )

    baseline = reference["rainfall_sum_mm"].mean()

    annual["climatology_mm"] = baseline
    annual["anomaly_mm"] = (
        annual["rainfall_sum_mm"] - baseline
    )

    annual["anomaly_pct"] = (
        100.0 * annual["anomaly_mm"] / baseline
        if baseline > 0 else np.nan
    )

    return annual


def build_seasonal(monthly, config):
    """Sum the specified calendar-year rainfall seasons."""

    rows = []

    for season_name, months in config["seasons"].items():
        months = sorted(set(int(m) for m in months))

        for year, group in monthly.groupby("year"):
            selected = group[
                group["month"].isin(months)
            ]

            count = selected["rainfall_mm"].count()
            complete = count == len(months)

            rainfall = (
                selected["rainfall_mm"].sum()
                if complete
                else np.nan
            )

            rows.append({
                "year": int(year),
                "season": season_name,
                "months_expected": len(months),
                "months_available": int(count),
                "complete": bool(complete),
                "rainfall_mm": rainfall,
            })

    return pd.DataFrame(rows).sort_values(
        ["year", "season"]
    ).reset_index(drop=True)


def seasonal_climatology(
    seasonal, ref_start, ref_end
):
    reference = seasonal[
        seasonal["year"].between(
            ref_start, ref_end
        )
    ].copy()

    expected_years = ref_end - ref_start + 1

    if not reference["complete"].all():
        raise ValueError(
            "Incomplete season in climatology period."
        )

    grouped = reference.groupby("season")["rainfall_mm"]

    clim = grouped.agg(
        mean_mm="mean",
        std_mm="std",
        n_years="count",
    ).reset_index()

    for percentile in [10, 20, 50, 80, 90]:
        clim[f"p{percentile}_mm"] = (
            grouped.quantile(
                percentile / 100
            ).values
        )

    if (clim["n_years"] != expected_years).any():
        raise ValueError(
            "Incomplete seasonal reference."
        )

    return clim


def add_seasonal_anomalies(seasonal, clim):
    df = seasonal.merge(
        clim[
            ["season", "mean_mm",
             "p10_mm", "p20_mm",
             "p80_mm", "p90_mm"]
        ],
        on="season",
        how="left",
        validate="many_to_one",
    )

    df = df.rename(
        columns={"mean_mm": "climatology_mm"}
    )

    df["anomaly_mm"] = (
        df["rainfall_mm"] - df["climatology_mm"]
    )

    df["anomaly_pct"] = np.where(
        df["climatology_mm"] > 0,
        100.0 * df["anomaly_mm"] /
        df["climatology_mm"],
        np.nan,
    )

    return df


def build_transitions(seasonal, config):
    """
    Identify anomalously dry JJAS seasons followed
    by anomalously wet October–November seasons.

    Use empirically calculated seasonal percentiles.
    """

    thresholds = config["drought_analysis"]

    dry_pct = int(thresholds["dry_percentile"])
    extreme_dry_pct = int(
        thresholds["extreme_dry_percentile"]
    )
    wet_pct = int(thresholds["wet_percentile"])
    extreme_wet_pct = int(
        thresholds["extreme_wet_percentile"]
    )

    required = {"JJAS", "ON"}
    present = set(config["seasons"])

    if not required.issubset(present):
        raise ValueError(
            "Both JJAS and ON must be defined."
        )

    # This implementation requires these percentile
    # thresholds to match the calculated quantiles.
    supported = {10, 20, 80, 90}
    requested = {
        dry_pct, extreme_dry_pct,
        wet_pct, extreme_wet_pct,
    }

    if not requested.issubset(supported):
        raise ValueError(
            "Transition thresholds must currently "
            "use the 10, 20, 80 or 90 percentiles."
        )

    jjas = seasonal[
        seasonal["season"] == "JJAS"
    ].copy()

    on = seasonal[
        seasonal["season"] == "ON"
    ].copy()

    jjas = jjas.rename(
        columns={
            "rainfall_mm": "JJAS_rainfall_mm",
            "anomaly_pct": "JJAS_anomaly_pct",
            "complete": "JJAS_complete",
            f"p{dry_pct}_mm": "dry_threshold_mm",
            f"p{extreme_dry_pct}_mm":
                "extreme_dry_threshold_mm",
        }
    )

    on = on.rename(
        columns={
            "rainfall_mm": "ON_rainfall_mm",
            "anomaly_pct": "ON_anomaly_pct",
            "complete": "ON_complete",
            f"p{wet_pct}_mm": "wet_threshold_mm",
            f"p{extreme_wet_pct}_mm":
                "extreme_wet_threshold_mm",
        }
    )

    columns_jjas = [
        "year", "JJAS_rainfall_mm",
        "JJAS_anomaly_pct", "JJAS_complete",
        "dry_threshold_mm",
        "extreme_dry_threshold_mm",
    ]

    columns_on = [
        "year", "ON_rainfall_mm",
        "ON_anomaly_pct", "ON_complete",
        "wet_threshold_mm",
        "extreme_wet_threshold_mm",
    ]

    result = jjas[columns_jjas].merge(
        on[columns_on],
        on="year",
        how="inner",
        validate="one_to_one",
    )

    valid = (
        result["JJAS_complete"]
        & result["ON_complete"]
    )

    dry = (
        result["JJAS_rainfall_mm"]
        < result["dry_threshold_mm"]
    )

    extreme_dry = (
        result["JJAS_rainfall_mm"]
        < result["extreme_dry_threshold_mm"]
    )

    wet = (
        result["ON_rainfall_mm"]
        > result["wet_threshold_mm"]
    )

    extreme_wet = (
        result["ON_rainfall_mm"]
        > result["extreme_wet_threshold_mm"]
    )

    result["unusually_dry_JJAS"] = (
        valid & dry
    )

    result["extremely_dry_JJAS"] = (
        valid & extreme_dry
    )

    result["unusually_wet_ON"] = (
        valid & wet
    )

    result["extremely_wet_ON"] = (
        valid & extreme_wet
    )

    result["dry_to_wet_reversal"] = (
        valid & dry & wet
    )

    result["extreme_reversal"] = (
        valid & extreme_dry & extreme_wet
    )

    result["transition_status"] = np.where(
        valid, "complete", "incomplete"
    )

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if any monthly raster is missing."
    )
    args = parser.parse_args()

    config = load_config()
    ensure_directories(config)

    logger = setup_logger(
        name="rainfall_series",
        logfile="rainfall_series.log",
    )

    start, end, ref_start, ref_end = (
        get_periods(config)
    )

    tables = project_path(
        config["paths"]["tables"]
    )

    logger.info("Loading Rwanda boundary")
    boundary = load_rwanda_boundary(config)

    logger.info("Calculating monthly rainfall")
    monthly = build_monthly(
        config, boundary, logger
    )

    missing = monthly[
        monthly["rainfall_mm"].isna()
    ]

    if args.strict and not missing.empty:
        raise RuntimeError(
            f"{len(missing)} monthly values are missing."
        )

    climatology = build_monthly_climatology(
        monthly, ref_start, ref_end
    )

    monthly = add_monthly_anomalies(
        monthly, climatology
    )

    annual = build_annual(
        monthly, ref_start, ref_end
    )

    seasonal = build_seasonal(
        monthly, config
    )

    season_clim = seasonal_climatology(
        seasonal, ref_start, ref_end
    )

    seasonal = add_seasonal_anomalies(
        seasonal, season_clim
    )

    transitions = build_transitions(
        seasonal, config
    )

    outputs = [
        (
            monthly,
            f"rwanda_monthly_rainfall_{start}_{end}.csv",
        ),
        (
            annual,
            f"rwanda_annual_rainfall_{start}_{end}.csv",
        ),
        (
            seasonal,
            f"rwanda_seasonal_rainfall_{start}_{end}.csv",
        ),
        (
            climatology,
            f"rwanda_monthly_climatology_"
            f"{ref_start}_{ref_end}.csv",
        ),
        (
            season_clim,
            f"rwanda_seasonal_climatology_"
            f"{ref_start}_{ref_end}.csv",
        ),
        (
            transitions,
            f"rwanda_dry_wet_transitions_"
            f"{start}_{end}.csv",
        ),
    ]

    for dataframe, filename in outputs:
        saved = save_csv(
            dataframe, filename, tables
        )
        logger.info("Saved: %s", saved)

    complete_years = int(
        annual["complete"].sum()
    )

    reversal_years = transitions.loc[
        transitions["dry_to_wet_reversal"],
        "year"
    ].tolist()

    logger.info(
        "Complete years: %d",
        complete_years
    )

    logger.info(
        "Dry-to-wet reversal years: %s",
        reversal_years
    )

    logger.info(
        "Monthly values missing: %d",
        len(missing)
    )

    logger.info(
        "Rainfall series analysis completed."
    )


if __name__ == "__main__":
    main()
