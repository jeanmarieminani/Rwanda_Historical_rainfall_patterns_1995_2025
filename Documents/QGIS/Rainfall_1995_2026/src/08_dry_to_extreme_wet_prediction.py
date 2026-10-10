"""Rwanda CHIRPS: dry JAS -> extreme wet ON historical risk and 2026 forecast.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import fisher_exact, beta
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

from common import load_config, ensure_directories, project_path, setup_logger

FEATURES = ['jul_aug_mm', 'jan_aug_mm', 'jul_oni', 'aug_dmi']


def load_inputs(config):
    start, end = int(config['project']['start_year']), int(config['project']['end_year'])
    monthly_path = project_path(config['paths']['tables']) / f'rwanda_monthly_rainfall_{start}_{end}.csv'
    indices_path = project_path(config['paths']['climate_indices']) / 'enso_iod_monthly.csv'
    for path in (monthly_path, indices_path):
        if not path.is_file():
            raise FileNotFoundError(f'Missing input: {path}')
    rain = pd.read_csv(monthly_path)
    idx = pd.read_csv(indices_path)
    for frame, value_cols in ((rain, ['rainfall_mm']), (idx, ['oni', 'dmi'])):
        frame['year'] = pd.to_numeric(frame['year'], errors='raise').astype(int)
        frame['month'] = pd.to_numeric(frame['month'], errors='raise').astype(int)
        if frame.duplicated(['year', 'month']).any():
            raise ValueError('Duplicate year/month records found')
        for col in value_cols:
            frame[col] = pd.to_numeric(frame[col], errors='coerce')
    return rain, idx


def build_yearly(rain, indices):
    rainfall = rain.pivot(index='year', columns='month', values='rainfall_mm').reindex(columns=range(1, 13))
    oni = indices.pivot(index='year', columns='month', values='oni').reindex(columns=range(1, 13))
    dmi = indices.pivot(index='year', columns='month', values='dmi').reindex(columns=range(1, 13))
    rows = []
    for year in rainfall.index:
        r = rainfall.loc[year]
        def complete_sum(months):
            x = r.loc[months]
            return float(x.sum()) if x.notna().all() else np.nan
        rows.append(dict(
            year=int(year),
            jul_aug_mm=complete_sum([7, 8]),
            jan_aug_mm=complete_sum(list(range(1, 9))),
            jas_mm=complete_sum([7, 8, 9]),
            on_mm=complete_sum([10, 11]),
            jul_oni=float(oni.loc[year, 7]) if year in oni.index else np.nan,
            aug_dmi=float(dmi.loc[year, 8]) if year in dmi.index else np.nan,
        ))
    return pd.DataFrame(rows).sort_values('year').reset_index(drop=True)


def jeffreys_interval(success, total, credibility=0.95):
    if total == 0:
        return (np.nan, np.nan)
    alpha = (1 - credibility) / 2
    return (float(beta.ppf(alpha, success + .5, total - success + .5)),
            float(beta.ppf(1-alpha, success + .5, total - success + .5)))


def event_table(yearly, ref_start, ref_end, dry_q=.20, wet_q=.90):
    reference = yearly[yearly.year.between(ref_start, ref_end)]
    if reference[['jas_mm', 'jul_aug_mm', 'on_mm']].isna().any().any():
        raise ValueError('Incomplete climatology period')
    thresholds = {
        'jas_dry_p20_mm': float(reference.jas_mm.quantile(dry_q)),
        'jul_aug_dry_p20_mm': float(reference.jul_aug_mm.quantile(dry_q)),
        'on_wet_p90_mm': float(reference.on_mm.quantile(wet_q)),
        'on_wet_p80_mm': float(reference.on_mm.quantile(.80)),
        'on_climatology_mm': float(reference.on_mm.mean()),
    }
    df = yearly.copy()
    df['jas_dry'] = pd.Series(np.where(df.jas_mm.notna(), df.jas_mm < thresholds['jas_dry_p20_mm'], pd.NA), dtype='boolean')
    df['jul_aug_dry'] = pd.Series(np.where(df.jul_aug_mm.notna(), df.jul_aug_mm < thresholds['jul_aug_dry_p20_mm'], pd.NA), dtype='boolean')
    df['on_exceptional'] = pd.Series(np.where(df.on_mm.notna(), df.on_mm > thresholds['on_wet_p90_mm'], pd.NA), dtype='boolean')
    df['on_unusually_wet'] = pd.Series(np.where(df.on_mm.notna(), df.on_mm > thresholds['on_wet_p80_mm'], pd.NA), dtype='boolean')
    df['extreme_transition'] = df.jas_dry & df.on_exceptional
    df['on_anomaly_pct'] = (df.on_mm / thresholds['on_climatology_mm'] - 1) * 100
    return df, thresholds


def contingency(events, dry_col, wet_col, label):
    valid = events.dropna(subset=[dry_col, wet_col]).copy()
    dry = valid[dry_col].astype(bool)
    wet = valid[wet_col].astype(bool)
    a = int((dry & wet).sum())
    b = int((dry & ~wet).sum())
    c = int((~dry & wet).sum())
    d = int((~dry & ~wet).sum())
    odds, p = fisher_exact([[a, b], [c, d]])
    p_dry = a/(a+b) if (a+b) else np.nan
    p_notdry = c/(c+d) if (c+d) else np.nan
    lo1, hi1 = jeffreys_interval(a, a+b)
    lo2, hi2 = jeffreys_interval(c, c+d)
    return dict(comparison=label, n_years=len(valid), dry_and_wet=a, dry_notwet=b,
                notdry_and_wet=c, notdry_notwet=d, p_wet_given_dry=p_dry,
                dry_ci95_low=lo1, dry_ci95_high=hi1, p_wet_given_notdry=p_notdry,
                notdry_ci95_low=lo2, notdry_ci95_high=hi2,
                risk_difference=p_dry-p_notdry, odds_ratio=odds, fisher_p=p)


def build_model():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=.25, solver='liblinear', max_iter=2000, random_state=42))


def backtest(yearly, thresholds, minimum_training=15):
    # Historical 1995-2025 only. Label threshold fixed from 1995-2024
    # for descriptive 2026 analysis; for honest historical backtests,
    # recompute the training-only p90 threshold at every forecast origin.
    data = yearly[yearly.year <= 2025].copy()
    data = data.dropna(subset=FEATURES + ['on_mm']).reset_index(drop=True)
    records = []
    for i in range(minimum_training, len(data)):
        train, test = data.iloc[:i], data.iloc[i:i+1]
        threshold = float(train.on_mm.quantile(.90))
        y = (train.on_mm > threshold).astype(int)
        actual = int(test.on_mm.iloc[0] > threshold)
        # Training climatology is the base rate for this event definition.
        baseline = float((y.sum()+.5)/(len(y)+1))
        model_probability = np.nan
        if y.nunique() == 2 and y.sum() >= 2:
            model = build_model()
            model.fit(train[FEATURES], y)
            model_probability = float(model.predict_proba(test[FEATURES])[0, 1])
        records.append(dict(year=int(test.year.iloc[0]), threshold_mm=threshold,
                            actual_extreme=actual, climatology_probability=baseline,
                            ml_probability=model_probability))
    return pd.DataFrame(records)


def forecast_2026(yearly, thresholds):
    historical = yearly[yearly.year <= 2025].dropna(subset=FEATURES+['on_mm']).copy()
    future = yearly[yearly.year == 2026].copy()
    if len(future) != 1 or future[FEATURES].isna().any().any():
        raise ValueError('2026 forecast inputs missing: require Jan-Aug rainfall, July ONI, August DMI')
    y = (historical.on_mm > thresholds['on_wet_p90_mm']).astype(int)
    prior = float((int(y.sum())+.5)/(len(y)+1))
    pred = np.nan
    if y.nunique() == 2 and int(y.sum()) >= 2:
        model = build_model()
        model.fit(historical[FEATURES], y)
        pred = float(model.predict_proba(future[FEATURES])[0, 1])
    n_extreme = int(y.sum())
    return dict(year=2026, issue_basis='CHIRPS observed through Aug 2026; ONI centered Jul; DMI Aug',
                jas_complete=False, on_complete=False,
                jul_aug_mm=float(future.jul_aug_mm.iloc[0]),
                july_august_dry=bool(future.jul_aug_mm.iloc[0] < thresholds['jul_aug_dry_p20_mm']),
                july_oni=float(future.jul_oni.iloc[0]),
                august_dmi=float(future.aug_dmi.iloc[0]),
                on_p90_threshold_mm=thresholds['on_wet_p90_mm'],
                historical_extreme_years=n_extreme,
                historical_years=len(historical),
                climatology_probability=prior,
                ml_probability=pred,
                warning='Exploratory estimate; use only if out-of-sample skill improves on climatology. July ONI availability must be checked against actual publication date.')


def make_figures(events, comparisons, forecasts, config):
    out = project_path(config['paths']['figures'])
    out.mkdir(parents=True, exist_ok=True)
    hist = events[(events.year <= 2025) & events.jas_mm.notna() & events.on_mm.notna()].copy()
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(hist.jas_mm, hist.on_mm, c='#417ca9', alpha=.85)
    for _, row in hist.iterrows():
        if bool(row.extreme_transition) or row.year in (1997, 2019, 2023, 2024):
            ax.annotate(str(int(row.year)), (row.jas_mm, row.on_mm), xytext=(5,5), textcoords='offset points', fontsize=9)
    dry = float(comparisons['jas_dry_p20_mm'])
    wet = float(comparisons['on_wet_p90_mm'])
    ax.axvline(dry, ls='--', color='#bc6c25', label=f'JAS dry p20: {dry:.1f} mm')
    ax.axhline(wet, ls='--', color='#b22d34', label=f'ON extreme p90: {wet:.1f} mm')
    ax.set(xlabel='July–September rainfall (mm)', ylabel='October–November rainfall (mm)',
           title='Rwanda: historical JAS dryness vs exceptional ON rainfall (1995–2025)')
    ax.legend(fontsize=9)
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'08_JAS_ON_extreme_transition.png', dpi=300)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 5))
    years = hist.year.astype(int).to_numpy()
    colors = np.where(hist.on_exceptional.fillna(False).to_numpy(dtype=bool), '#b22d34', '#417ca9')
    ax.bar(years, hist.on_mm, color=colors)
    ax.axhline(wet, color='#b22d34', linestyle='--', label=f'Historical ON p90: {wet:.1f} mm')
    ax.set(xlabel='Year', ylabel='October–November rainfall (mm)',
           title='Rwanda October–November rainfall and extreme threshold')
    ax.legend()
    ax.grid(axis='y', alpha=.2)
    fig.tight_layout()
    fig.savefig(out/'09_ON_extreme_rainfall_years.png', dpi=300)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--min-train', type=int, default=15)
    args = parser.parse_args()
    config = load_config()
    ensure_directories(config)
    logger = setup_logger('dry_wet_prediction', 'dry_wet_prediction.log')
    rainfall, indices = load_inputs(config)
    yearly = build_yearly(rainfall, indices)
    c = config['climatology']
    ref_start, ref_end = int(c['reference_start_year']), int(c['reference_end_year'])
    events, thresholds = event_table(yearly, ref_start, ref_end)
    hist = events[events.year <= 2025]
    comparisons = pd.DataFrame([
        contingency(hist, 'jas_dry', 'on_exceptional', 'JAS p20 -> ON p90'),
        contingency(hist, 'jas_dry', 'on_unusually_wet', 'JAS p20 -> ON p80'),
        contingency(hist, 'jul_aug_dry', 'on_exceptional', 'Jul-Aug p20 -> ON p90'),
    ])
    bt = backtest(yearly, thresholds, minimum_training=args.min_train)
    if bt.empty:
        raise ValueError('Insufficient years for historical backtesting')
    score_records = []
    for key, name in [('climatology_probability', 'Climatology'), ('ml_probability', 'LogisticRegression')]:
        valid = bt.dropna(subset=[key])
        if not valid.empty:
            score_records.append(dict(model=name, n_test_years=len(valid),
                                      brier_score=brier_score_loss(valid.actual_extreme, valid[key])))
    scores = pd.DataFrame(score_records)
    forecast = pd.DataFrame([forecast_2026(yearly, thresholds)])
    folder = project_path(config['paths']['tables'])
    folder.mkdir(parents=True, exist_ok=True)
    tables = {
        'rwanda_JAS_ON_extreme_events.csv': events,
        'rwanda_JAS_ON_event_probabilities.csv': comparisons,
        'rwanda_JAS_ON_backtest.csv': bt,
        'rwanda_JAS_ON_backtest_scores.csv': scores,
        'rwanda_2026_extreme_ON_forecast.csv': forecast,
        'rwanda_JAS_ON_thresholds.csv': pd.DataFrame([thresholds]),
    }
    for name, table in tables.items():
        table.to_csv(folder/name, index=False, encoding='utf-8-sig', float_format='%.6f')
        logger.info('Saved %s', name)
    make_figures(events, thresholds, forecast, config)
    logger.info('Saved two figures')
    print('\nEVENT COUNTS 1995-2025')
    print('Complete years:', int(hist.on_mm.notna().sum()))
    print('Extreme ON events:', int(hist.on_exceptional.fillna(False).sum()))
    print('Extreme JAS -> ON transitions:', int(hist.extreme_transition.fillna(False).sum()))
    print('\nTHRESHOLDS (1995-2024):')
    print(pd.DataFrame([thresholds]).round(2).to_string(index=False))
    print('\nHISTORICAL CONDITIONAL EVENT FREQUENCIES:')
    print(comparisons[['comparison', 'n_years', 'dry_and_wet', 'p_wet_given_dry', 'p_wet_given_notdry', 'fisher_p']].round(3).to_string(index=False))
    print('\nBACKTEST BRIER SCORES (lower is better):')
    print(scores.round(4).to_string(index=False))
    print('\n2026 PROVISIONAL FORECAST:')
    print(forecast[['jul_aug_mm','july_august_dry','on_p90_threshold_mm','climatology_probability','ml_probability']].round(4).to_string(index=False))
    logger.info('Completed; ML probability is exploratory, not calibrated')


if __name__ == '__main__':
    main()