"""Rwanda CHIRPS × NOAA ONI/DMI teleconnection analysis, 1995–2026.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import statsmodels.api as sm

from common import load_config, ensure_directories, setup_logger, project_path


def read_inputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    start = int(config["project"]["start_year"])
    end = int(config["project"]["end_year"])
    folder = project_path(config["paths"]["tables"])
    rainfall = pd.read_csv(folder / f"rwanda_monthly_rainfall_{start}_{end}.csv")
    seasonal = pd.read_csv(folder / f"rwanda_seasonal_rainfall_{start}_{end}.csv")
    indices = pd.read_csv(project_path(config["paths"]["climate_indices"]) / "enso_iod_monthly.csv")
    for label, df in [("rainfall", rainfall), ("indices", indices)]:
        if df.duplicated(["year", "month"]).any():
            raise ValueError(f"Duplicate year/month records in {label}")
        if not {"year", "month"}.issubset(df.columns):
            raise ValueError(f"Missing year/month columns in {label}")
    required = {"rainfall_mm"}
    if not required.issubset(rainfall.columns):
        raise ValueError("Missing rainfall_mm in monthly rainfall table")
    if not {"oni", "dmi"}.issubset(indices.columns):
        raise ValueError("Missing oni or dmi column in indices table")
    if not {"year", "season", "rainfall_mm", "complete"}.issubset(seasonal.columns):
        raise ValueError("Missing expected columns in seasonal rainfall table")
    return rainfall, seasonal, indices


def join_monthly(rainfall: pd.DataFrame, indices: pd.DataFrame) -> pd.DataFrame:
    left = rainfall.copy()
    right = indices[["year", "month", "oni", "dmi"]].copy()
    for df in (left, right):
        df["year"] = pd.to_numeric(df["year"], errors="raise").astype(int)
        df["month"] = pd.to_numeric(df["month"], errors="raise").astype(int)
    for c in ("oni", "dmi"):
        right[c] = pd.to_numeric(right[c], errors="coerce")
    left["rainfall_mm"] = pd.to_numeric(left["rainfall_mm"], errors="coerce")
    out = left.merge(right, on=["year", "month"], how="left", validate="one_to_one")
    out["date"] = pd.to_datetime(dict(year=out.year, month=out.month, day=1))
    out = out.sort_values("date").reset_index(drop=True)
    full_dates = pd.date_range(out.date.min(), out.date.max(), freq="MS")
    if len(out) != len(full_dates) or not out.date.eq(full_dates).all():
        raise ValueError("Monthly input has a missing or repeated calendar month")
    return out


def lag_correlations(monthly: pd.DataFrame, max_lag: int) -> pd.DataFrame:
    """Association of ON rainfall anomaly with index from lag months EARLIER.

    lag=0 means same month (explanatory only); lag=2 means index 2 months
    earlier than the rainfall month. Only Oct/Nov rainfall observations are used.
    """
    m = monthly.copy()
    if "anomaly_mm" not in m:
        reference = m[m.year.between(1995, 2024)]
        means = reference.groupby("month").rainfall_mm.mean()
        m["anomaly_mm"] = m.rainfall_mm - m.month.map(means)
    for c in ("rainfall_mm", "anomaly_mm"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    for c in ("oni", "dmi"):
        m[c] = pd.to_numeric(m[c], errors="coerce")
    rows = []
    for idx in ("oni", "dmi"):
        for lag in range(max_lag + 1):
            predictor = m[idx].shift(lag)
            mask = m.month.isin([10, 11]) & m.anomaly_mm.notna() & predictor.notna()
            x = predictor.loc[mask].to_numpy(dtype=float)
            y = m.loc[mask, "anomaly_mm"].to_numpy(dtype=float)
            if len(x) < 8 or np.std(x) == 0 or np.std(y) == 0:
                rp = sp = pp = ps = np.nan
            else:
                rp, pp = pearsonr(x, y)
                sp, ps = spearmanr(x, y)
            rows.append({"index": idx, "lag_months": lag,
                         "meaning": "index precedes Oct/Nov rainfall" if lag else "contemporaneous (not a forecast)",
                         "n_months": len(x), "pearson_r": rp, "pearson_p": pp,
                         "spearman_rho": sp, "spearman_p": ps})
    return pd.DataFrame(rows)


def seasonal_table(monthly: pd.DataFrame, seasonal: pd.DataFrame, config: dict) -> pd.DataFrame:
    """One row per year. August indices are pre-SON rainfall predictor candidates.

    October/November index averages are CONTEMPORANEOUS and are only for
    descriptive relationships, not August-origin forecasting.
    """
    years = pd.DataFrame({"year": sorted(monthly.year.unique())})
    for s in ("JJAS", "ON"):
        tmp = seasonal[seasonal.season == s][["year", "rainfall_mm", "complete"]].copy()
        tmp["complete"] = tmp.complete.astype(str).str.lower().eq("true")
        tmp.loc[~tmp.complete, "rainfall_mm"] = np.nan
        tmp = tmp.rename(columns={"rainfall_mm": f"{s}_rainfall_mm", "complete": f"{s}_complete"})
        years = years.merge(tmp, on="year", how="left", validate="one_to_one")
    aug = monthly[monthly.month == 8][["year", "oni", "dmi"]].rename(
        columns={"oni": "aug_oni", "dmi": "aug_dmi"})
    years = years.merge(aug, on="year", how="left", validate="one_to_one")
    on_indices = (monthly[monthly.month.isin([10, 11])]
                  .groupby("year").agg(on_oni=("oni", "mean"), on_dmi=("dmi", "mean"),
                                       oni_months=("oni", "count"), dmi_months=("dmi", "count"))
                  .reset_index())
    on_indices.loc[on_indices.oni_months != 2, "on_oni"] = np.nan
    on_indices.loc[on_indices.dmi_months != 2, "on_dmi"] = np.nan
    years = years.merge(on_indices, on="year", how="left", validate="one_to_one")
    reference = years[years.year.between(int(config["climatology"]["reference_start_year"]),
                                          int(config["climatology"]["reference_end_year"]))]
    for s in ("JJAS", "ON"):
        ref = reference[f"{s}_rainfall_mm"].dropna()
        expected = (int(config["climatology"]["reference_end_year"]) -
                    int(config["climatology"]["reference_start_year"]) + 1)
        if len(ref) != expected:
            raise ValueError(f"Incomplete {s} climatology; expected {expected} complete years")
        mean = ref.mean()
        years[f"{s}_anomaly_pct"] = 100 * (years[f"{s}_rainfall_mm"] - mean) / mean
        years[f"{s}_anomaly_mm"] = years[f"{s}_rainfall_mm"] - mean
    # Categories describe August index values, NOT official ENSO/IOD episode declarations.
    years["aug_enso_phase"] = np.select(
        [years.aug_oni >= 1.5, years.aug_oni >= 0.5,
         years.aug_oni <= -1.5, years.aug_oni <= -0.5],
        ["warm >=1.5", "warm 0.5-1.5", "cold <=-1.5", "cold -0.5 to -1.5"],
        default="neutral")
    years.loc[years.aug_oni.isna(), "aug_enso_phase"] = "unavailable"
    years["aug_iod_phase"] = np.select(
        [years.aug_dmi >= 0.4, years.aug_dmi <= -0.4],
        ["positive >=0.4", "negative <=-0.4"], default="neutral")
    years.loc[years.aug_dmi.isna(), "aug_iod_phase"] = "unavailable"
    return years


def season_phase_summary(years: pd.DataFrame) -> pd.DataFrame:
    valid = years.dropna(subset=["ON_rainfall_mm", "aug_oni", "aug_dmi"]).copy()
    rows = []
    for phase_col in ("aug_enso_phase", "aug_iod_phase"):
        for phase, sub in valid.groupby(phase_col):
            rows.append({"phase_variable": phase_col, "phase": phase, "n_years": len(sub),
                         "mean_ON_mm": sub.ON_rainfall_mm.mean(),
                         "mean_ON_anomaly_pct": sub.ON_anomaly_pct.mean(),
                         "mean_JJAS_mm": sub.JJAS_rainfall_mm.mean()})
    return pd.DataFrame(rows)


def regression(years: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """HC3-robust descriptive OLS. Same-season indices are NOT forecast features."""
    df = years.dropna(subset=["ON_anomaly_mm", "on_oni", "on_dmi"]).copy()
    if len(df) < 15:
        raise ValueError("Too few complete years for ON regression")
    df["interaction"] = df.on_oni * df.on_dmi
    X = sm.add_constant(df[["on_oni", "on_dmi", "interaction"]], has_constant="add")
    y = df.ON_anomaly_mm
    fit = sm.OLS(y, X).fit(cov_type="HC3")
    coefficients = pd.DataFrame({"term": fit.params.index, "coefficient_mm": fit.params.values,
                                 "robust_se": fit.bse.values, "p_value": fit.pvalues.values,
                                 "ci95_lower": fit.conf_int()[0].values,
                                 "ci95_upper": fit.conf_int()[1].values})
    diagnostics = pd.DataFrame([{"n_years": len(df), "r_squared": fit.rsquared,
                                 "adjusted_r_squared": fit.rsquared_adj,
                                 "model": "OLS HC3; contemporaneous ON ONI/DMI; explanatory only"}])
    return coefficients, diagnostics


def make_plots(years: pd.DataFrame, correlations: pd.DataFrame, config: dict, logger: logging.Logger) -> None:
    figs = project_path(config["paths"]["figures"])
    figs.mkdir(parents=True, exist_ok=True)
    valid = years.dropna(subset=["ON_anomaly_pct", "on_oni", "on_dmi"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, col, name in zip(axes, ["on_oni", "on_dmi"], ["ONI (Oct–Nov mean)", "DMI (Oct–Nov mean)"]):
        ax.scatter(valid[col], valid.ON_anomaly_pct, s=38, alpha=0.75)
        ax.axhline(0, linewidth=1, color="gray")
        ax.axvline(0, linewidth=1, color="gray", linestyle="--")
        for yr in (1997, 2015, 2019, 2023, 2024):
            s = valid[valid.year == yr]
            if len(s):
                ax.annotate(str(yr), (float(s[col].iloc[0]), float(s.ON_anomaly_pct.iloc[0])),
                            xytext=(4, 4), textcoords="offset points", fontsize=8)
        ax.set_xlabel(name)
        ax.set_ylabel("Rwanda ON rainfall anomaly (%)")
        ax.grid(alpha=0.18)
    fig.suptitle("Rwanda October–November rainfall vs climate indices (descriptive)")
    fig.tight_layout()
    fig.savefig(figs / "07_ON_rainfall_ENSO_IOD_scatter.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4.8))
    for idx, symbol in (("oni", "o"), ("dmi", "s")):
        rows = correlations[correlations["index"] == idx]
        ax.plot(rows.lag_months, rows.pearson_r, marker=symbol, label=idx.upper())
    ax.axhline(0, color="gray", linewidth=1)
    ax.set(xlabel="Index lead before Oct/Nov rainfall (months)", ylabel="Pearson correlation",
           title="Lagged monthly ON rainfall-anomaly correlations")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(figs / "08_ON_rainfall_lag_correlations.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved two teleconnection figures")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-lag", type=int, default=6,
                        help="Maximum climate-index lead in months (default: 6)")
    args = parser.parse_args()
    if not 0 <= args.max_lag <= 24:
        parser.error("--max-lag must be 0 to 24")
    config = load_config()
    ensure_directories(config)
    logger = setup_logger("rwanda_teleconnections", "teleconnections.log")
    rainfall, seasonal, indices = read_inputs(config)
    monthly = join_monthly(rainfall, indices)
    logger.info("Combined %d monthly rows; paired rainfall/ONI/DMI: %d", len(monthly),
                int(monthly[["rainfall_mm", "oni", "dmi"]].notna().all(axis=1).sum()))
    corr = lag_correlations(monthly, args.max_lag)
    years = seasonal_table(monthly, seasonal, config)
    phases = season_phase_summary(years)
    coefficients, model_info = regression(years)
    out = project_path(config["paths"]["tables"])
    out.mkdir(parents=True, exist_ok=True)
    result_tables = {
        "rwanda_enso_iod_rainfall_1995_2026.csv": monthly,
        "rwanda_enso_iod_correlations.csv": corr,
        "rwanda_enso_iod_seasonal_comparison.csv": phases,
        "rwanda_enso_iod_annual_features.csv": years,
        "rwanda_enso_iod_regression.csv": coefficients,
        "rwanda_enso_iod_regression_diagnostics.csv": model_info,
    }
    for name, data in result_tables.items():
        data.to_csv(out / name, index=False, float_format="%.5f")
        logger.info("Saved: %s", name)
    make_plots(years, corr, config, logger)
    print("\nREGRESSION DIAGNOSTICS (explanatory, not a forecast)")
    print(model_info.round(4).to_string(index=False))
    print("\nCOEFFICIENTS (HC3 robust errors)")
    print(coefficients.round(4).to_string(index=False))
    print("\nAUGUST 2026 PREDICTOR STATUS")
    print(years.loc[years.year == 2026,
                    ["year", "aug_oni", "aug_dmi", "JJAS_rainfall_mm", "ON_rainfall_mm"]]
          .to_string(index=False))
    print("\nDone. No 2026 rainfall forecast was fitted in this script.")


if __name__ == "__main__":
    main()
