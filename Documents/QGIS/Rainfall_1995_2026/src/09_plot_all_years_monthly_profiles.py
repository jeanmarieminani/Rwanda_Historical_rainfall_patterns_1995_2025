"""
Rwanda CHIRPS Rainfall Analysis (1995–2026)
Plot monthly rainfall profiles
"""

from __future__ import annotations

import argparse
import math

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from common import (
    load_config,
    ensure_directories,
    setup_logger,
    project_path,
)

# ----------------------------------------------------------
# CONFIG
# ----------------------------------------------------------

MONTH_LABELS = [
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"
]

BAR_COLOR = "#2E7DBA"
DRY_COLOR = "#C98732"
CLIM_COLOR = "#C1272D"

JAS_SHADE = "#E8DCC7"
OND_SHADE = "#DCEBF8"

def configure_style():
    plt.rcParams.update({
        "figure.dpi": 120,
        "savefig.dpi": 300,
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

# ----------------------------------------------------------
# LOAD DATA
# ----------------------------------------------------------

def load_tables(config):
    project = config["project"]
    climate = config["climatology"]

    start = int(project["start_year"])
    end = int(project["end_year"])
    ref_start = int(climate["reference_start_year"])
    ref_end = int(climate["reference_end_year"])

    tables_dir = project_path(config["paths"]["tables"])

    monthly_file = tables_dir / (
        f"rwanda_monthly_rainfall_{start}_{end}.csv"
    )
    clim_file = tables_dir / (
        f"rwanda_monthly_climatology_{ref_start}_{ref_end}.csv"
    )

    if not monthly_file.exists():
        raise FileNotFoundError(monthly_file)

    if not clim_file.exists():
        raise FileNotFoundError(clim_file)

    monthly = pd.read_csv(monthly_file)
    climatology = pd.read_csv(clim_file)

    monthly["year"] = pd.to_numeric(monthly["year"]).astype(int)
    monthly["month"] = pd.to_numeric(monthly["month"]).astype(int)
    monthly["rainfall_mm"] = pd.to_numeric(
        monthly["rainfall_mm"], errors="coerce"
    )

    climatology["month"] = pd.to_numeric(
        climatology["month"]
    ).astype(int)

    return monthly, climatology

# ----------------------------------------------------------
# PLOT HELPERS
# ----------------------------------------------------------

def compute_year_summary(subset):
    jas = subset.loc[
        subset["month"].isin([6, 7, 8, 9]),
        "rainfall_mm"
    ].sum(min_count=3)

    ond = subset.loc[
        subset["month"].isin([10, 11, 12]),
        "rainfall_mm"
    ].sum(min_count=3)

    return jas, ond

def draw_one_panel(ax, subset, clim_values, year, ymax):
    months = np.arange(1, 13)

    values = (
        subset.set_index("month")["rainfall_mm"]
        .reindex(months)
        .to_numpy(dtype=float)
    )

    # Shade JAS and OND
    ax.axvspan(5.5, 9.5, color=JAS_SHADE, alpha=0.75, zorder=0)
    ax.axvspan(9.5, 12.5, color=OND_SHADE, alpha=0.45, zorder=0)

    colors = [
        DRY_COLOR if month in [6, 7, 8, 9] else BAR_COLOR
        for month in months
    ]

    ax.bar(
        months,
        values,
        color=colors,
        width=0.72,
        alpha=0.85,
        edgecolor="none",
        zorder=2,
    )

    ax.plot(
        months,
        clim_values,
        color=CLIM_COLOR,
        linestyle="--",
        linewidth=1.4,
        marker="o",
        markersize=2.5,
        zorder=3,
    )

    jas_total, ond_total = compute_year_summary(subset)

    ax.text(
        0.02, 0.92,
        str(year),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10,
        fontweight="bold",
    )

    ax.text(
        0.98, 0.92,
        f"JJAS={jas_total:.1f} mm\nOND={ond_total:.1f} mm"
        if np.isfinite(jas_total) and np.isfinite(ond_total)
        else f"JAS={jas_total:.1f} mm\nOND=incomplete",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox=dict(
            facecolor="white",
            edgecolor="#DDDDDD",
            boxstyle="round,pad=0.25",
            alpha=0.85,
        ),
    )

    ax.set_xlim(0.5, 12.5)
    ax.set_ylim(0, ymax)
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(MONTH_LABELS, fontsize=8)
    ax.grid(axis="y", alpha=0.18)

def save_figure(fig, filename, config, logger):
    out_dir = project_path(config["paths"]["figures"])
    out_dir.mkdir(parents=True, exist_ok=True)

    path = out_dir / filename
    dpi = int(config["visualization"]["figure_dpi"])

    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    logger.info("Saved figure: %s", path)

# ----------------------------------------------------------
# MAIN FIGURES
# ----------------------------------------------------------

def make_all_years_grid(monthly, climatology, config, logger, ncols=4):
    years = sorted(monthly["year"].unique().tolist())
    n_years = len(years)
    nrows = math.ceil(n_years / ncols)

    clim_values = (
        climatology.sort_values("month")["mean_mm"]
        .to_numpy(dtype=float)
    )

    ymax = max(
        float(monthly["rainfall_mm"].max(skipna=True)),
        float(np.nanmax(clim_values))
    ) * 1.15

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(5.0 * ncols, 2.8 * nrows),
        sharex=True,
        sharey=True,
    )

    axes = np.array(axes).reshape(-1)

    for ax, year in zip(axes, years):
        subset = monthly[monthly["year"] == year].copy()
        subset = subset.sort_values("month")

        draw_one_panel(
            ax=ax,
            subset=subset,
            clim_values=clim_values,
            year=year,
            ymax=ymax,
        )

    # Hide unused axes
    for ax in axes[n_years:]:
        ax.axis("off")

    # Label only leftmost and bottom axes for cleanliness
    for i, ax in enumerate(axes[:n_years]):
        row = i // ncols
        col = i % ncols

        if col == 0:
            ax.set_ylabel("Rainfall (mm)")
        if row == nrows - 1:
            ax.set_xlabel("Month")

    fig.suptitle(
        "Rwanda Monthly Rainfall Profiles for All Years\n"
        "CHIRPS v3 versus 1995–2024 climatology",
        fontsize=16,
        fontweight="bold",
        y=0.995,
    )

    legend_handles = [
        Patch(facecolor=BAR_COLOR, label="Monthly rainfall"),
        Patch(facecolor=DRY_COLOR, label="JJAS rainfall bars"),
        Line2D(
            [0], [0],
            color=CLIM_COLOR,
            linestyle="--",
            marker="o",
            markersize=4,
            label="Historical monthly mean",
        ),
        Patch(facecolor=JAS_SHADE, label="JJAS period"),
        Patch(facecolor=OND_SHADE, label="OND period"),
    ]

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=5,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 0.975])

    save_figure(
        fig,
        "07_all_years_monthly_profiles_grid.png",
        config,
        logger,
    )

def make_paged_figures(monthly, climatology, config, logger,
                       years_per_page=8, ncols=2):
    years = sorted(monthly["year"].unique().tolist())
    pages = [
        years[i:i + years_per_page]
        for i in range(0, len(years), years_per_page)
    ]

    clim_values = (
        climatology.sort_values("month")["mean_mm"]
        .to_numpy(dtype=float)
    )

    ymax = max(
        float(monthly["rainfall_mm"].max(skipna=True)),
        float(np.nanmax(clim_values))
    ) * 1.15

    for page_num, page_years in enumerate(pages, start=1):
        n_years = len(page_years)
        nrows = math.ceil(n_years / ncols)

        fig, axes = plt.subplots(
            nrows=nrows,
            ncols=ncols,
            figsize=(7.2 * ncols, 2.9 * nrows),
            sharex=True,
            sharey=True,
        )

        axes = np.atleast_1d(axes).reshape(-1)

        for ax, year in zip(axes, page_years):
            subset = monthly[monthly["year"] == year].copy()
            subset = subset.sort_values("month")

            draw_one_panel(
                ax=ax,
                subset=subset,
                clim_values=clim_values,
                year=year,
                ymax=ymax,
            )

        for ax in axes[n_years:]:
            ax.axis("off")

        for i, ax in enumerate(axes[:n_years]):
            row = i // ncols
            col = i % ncols

            if col == 0:
                ax.set_ylabel("Rainfall (mm)")
            if row == nrows - 1:
                ax.set_xlabel("Month")

        year_min = min(page_years)
        year_max = max(page_years)

        fig.suptitle(
            "Rwanda Monthly Rainfall Profiles\n"
            f"CHIRPS v3 versus 1995–2024 climatology | "
            f"Years {year_min}–{year_max}",
            fontsize=15,
            fontweight="bold",
            y=0.995,
        )

        legend_handles = [
            Patch(facecolor=BAR_COLOR, label="Monthly rainfall"),
            Patch(facecolor=DRY_COLOR, label="JAS rainfall bars"),
            Line2D(
                [0], [0],
                color=CLIM_COLOR,
                linestyle="--",
                marker="o",
                markersize=4,
                label="Historical monthly mean",
            ),
            Patch(facecolor=JAS_SHADE, label="JAS period"),
            Patch(facecolor=OND_SHADE, label="OND period"),
        ]

        fig.legend(
            handles=legend_handles,
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.01),
        )

        fig.tight_layout(rect=[0, 0.05, 1, 0.97])

        save_figure(
            fig,
            f"07_all_years_monthly_profiles_page_{page_num:02d}.png",
            config,
            logger,
        )

# ----------------------------------------------------------
# MAIN
# ----------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Plot Rwanda monthly rainfall profiles for all years"
    )
    parser.add_argument(
        "--years-per-page",
        type=int,
        default=8,
        help="Number of years in each paged figure"
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=4,
        help="Number of columns in full-grid figure"
    )

    args = parser.parse_args()

    config = load_config()
    ensure_directories(config)
    configure_style()

    logger = setup_logger(
        "all_year_profiles",
        "all_year_profiles.log",
    )

    monthly, climatology = load_tables(config)

    logger.info(
        "Loaded %d monthly rainfall records",
        len(monthly),
    )

    make_all_years_grid(
        monthly=monthly,
        climatology=climatology,
        config=config,
        logger=logger,
        ncols=args.grid_cols,
    )

    make_paged_figures(
        monthly=monthly,
        climatology=climatology,
        config=config,
        logger=logger,
        years_per_page=args.years_per_page,
        ncols=2,
    )

    logger.info("Completed all-year monthly profile figures.")

if __name__ == "__main__":
    main()