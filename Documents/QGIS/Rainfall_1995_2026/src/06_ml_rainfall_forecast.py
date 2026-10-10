
"""
Rwanda CHIRPS rainfall forecast: 2026.

Forecasts September, October, November and the
October-November total, using information available
through August of each prediction year.

Models:
- Historical climatology
- Ridge regression
- Random Forest

Uses expanding-window historical backtesting.

Run:
    python src/06_ml_rainfall_forecast.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

from common import (
    load_config,
    ensure_directories,
    setup_logger,
    project_path,
)


FEATURES = [
    "jan", "feb", "mar", "apr",
    "may", "jun", "jul", "aug",
    "jan_aug_total",
    "jja_total",
    "previous_on",
]

TARGETS = [
    "sep", "oct", "nov", "on_total"
]


def load_dataset(config):
    p = config["project"]

    path = project_path(
        config["paths"]["tables"]
    ) / (
        "rwanda_monthly_rainfall_"
        f"{p['start_year']}_{p['end_year']}.csv"
    )

    df = pd.read_csv(path)

    pivot = df.pivot(
        index="year",
        columns="month",
        values="rainfall_mm",
    )

    pivot = pivot.reindex(columns=range(1, 13))

    rows = []

    for year in pivot.index:
        r = pivot.loc[year]

        # Only generate forecasts when all
        # January-August observations exist.
        if r.loc[1:8].isna().any():
            continue

        prev_on = np.nan

        if year - 1 in pivot.index:
            previous = pivot.loc[year - 1]

            if previous.loc[[10, 11]].notna().all():
                prev_on = float(
                    previous.loc[[10, 11]].sum()
                )

        row = {
            "year": int(year),
            "jan_aug_total": float(r.loc[1:8].sum()),
            "jja_total": float(r.loc[6:8].sum()),
            "previous_on": prev_on,
            "sep": r.loc[9],
            "oct": r.loc[10],
            "nov": r.loc[11],
        }

        for month, name in enumerate(
            ["jan", "feb", "mar", "apr",
             "may", "jun", "jul", "aug"],
            start=1
        ):
            row[name] = float(r.loc[month])

        row["on_total"] = (
            float(r.loc[[10, 11]].sum())
            if r.loc[[10, 11]].notna().all()
            else np.nan
        )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("year")


def new_model(name):
    if name == "Ridge":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            Ridge(alpha=100.0),
        )

    if name == "RandomForest":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            RandomForestRegressor(
                n_estimators=300,
                max_depth=3,
                min_samples_leaf=3,
                max_features=0.7,
                random_state=42,
                n_jobs=-1,
            ),
        )

    raise ValueError(name)


def backtest(df, target, min_train=15):
    """
    Expanding-window evaluation.

    For every test year, use only earlier years
    for training and climatological predictions.
    """

    known = df.dropna(subset=[target]).copy()

    records = []

    for i in range(min_train, len(known)):
        train = known.iloc[:i]
        test = known.iloc[i:i + 1]

        actual = float(test[target].iloc[0])

        for model_name in [
            "Climatology",
            "Ridge",
            "RandomForest"
        ]:

            if model_name == "Climatology":
                predicted = float(
                    train[target].mean()
                )
            else:
                model = new_model(model_name)

                model.fit(
                    train[FEATURES],
                    train[target],
                )

                predicted = float(
                    model.predict(
                        test[FEATURES]
                    )[0]
                )

            records.append({
                "target": target,
                "year": int(test["year"].iloc[0]),
                "model": model_name,
                "actual_mm": actual,
                "predicted_mm": max(0, predicted),
            })

    return pd.DataFrame(records)


def calculate_scores(backtests):
    rows = []

    for (target, model), group in backtests.groupby(
        ["target", "model"]
    ):
        mae = mean_absolute_error(
            group["actual_mm"],
            group["predicted_mm"],
        )

        rows.append({
            "target": target,
            "model": model,
            "n_test_years": len(group),
            "MAE_mm": mae,
        })

    return pd.DataFrame(rows)


def forecast_2026(
    df, scores, backtests, config
):
    """
    Select the best historically tested model
    for each target and predict 2026.

    Empirical error bounds are diagnostic,
    not guaranteed calibrated prediction intervals.
    """

    forecast_year = int(
        config["project"]["end_year"]
    )

    future = df[df["year"] == forecast_year]

    if future.empty:
        raise ValueError(
            "No complete January-August data "
            f"for {forecast_year}."
        )

    train = df[df["year"] < forecast_year]

    results = []

    for target in TARGETS:
        valid = train.dropna(subset=[target])

        choices = scores[
            scores["target"] == target
        ].sort_values("MAE_mm")

        winner = choices.iloc[0]["model"]

        if winner == "Climatology":
            prediction = float(
                valid[target].mean()
            )
        else:
            model = new_model(winner)
            model.fit(
                valid[FEATURES],
                valid[target],
            )
            prediction = float(
                model.predict(
                    future[FEATURES]
                )[0]
            )

        historical_errors = backtests[
            (backtests["target"] == target)
            & (backtests["model"] == winner)
        ].copy()

        absolute_errors = np.abs(
            historical_errors["actual_mm"]
            - historical_errors["predicted_mm"]
        )

        # Approximate 80th-percentile absolute
        # backtest error envelope.
        uncertainty = float(
            np.quantile(absolute_errors, 0.80)
        )

        reference = valid[
            valid["year"].between(1995, 2024)
        ][target].mean()

        prediction = max(prediction, 0.0)

        results.append({
            "year": forecast_year,
            "target": target,
            "selected_model": winner,
            "forecast_mm": prediction,
            "historical_mean_mm": reference,
            "forecast_anomaly_pct": (
                100 * (prediction - reference) / reference
                if reference > 0 else np.nan
            ),
            "lower_error_band_mm": max(
                0, prediction - uncertainty
            ),
            "upper_error_band_mm": (
                prediction + uncertainty
            ),
            "backtest_MAE_mm": float(
                choices.iloc[0]["MAE_mm"]
            ),
        })

    return pd.DataFrame(results)


def main():
    config = load_config()
    ensure_directories(config)

    logger = setup_logger(
        "ml_rainfall",
        "ml_rainfall.log",
    )

    df = load_dataset(config)

    logger.info(
        "Available annual feature records: %d",
        len(df),
    )

    backtests = []

    for target in TARGETS:
        logger.info(
            "Backtesting target: %s",
            target,
        )

        backtests.append(
            backtest(df, target)
        )

    backtests = pd.concat(
        backtests,
        ignore_index=True,
    )

    scores = calculate_scores(backtests)

    # Separate model-selection and evaluation
    # is required for an unbiased assessment
    # of the selected winner; these scores
    # compare individual candidate models.
    forecast = forecast_2026(
        df, scores, backtests, config
    )

    output = project_path(
        config["paths"]["tables"]
    )

    backtests.to_csv(
        output / "ml_backtest_predictions.csv",
        index=False,
    )

    scores.to_csv(
        output / "ml_model_comparison.csv",
        index=False,
    )

    forecast.to_csv(
        output / "ml_2026_rainfall_forecast.csv",
        index=False,
    )

    print("\nMODEL COMPARISON")
    print(
        scores.round(2).to_string(index=False)
    )

    print("\n2026 FORECAST")
    print(
        forecast.round(2).to_string(index=False)
    )

    logger.info(
        "ML rainfall forecast completed."
    )


if __name__ == "__main__":
    main()
