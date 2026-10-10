
"""
COLLECT HISTORICAL ENSO AND IOD INDICES

Sources:
    NOAA CPC ONI version 6 (HTML)
    NOAA CPC DMI (plain-text monthly dataset)

Output:
    data/indices/enso_iod_monthly.csv
    data/indices/sources_metadata.json
"""

from __future__ import annotations

import json
import math
import re

from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path

import pandas as pd
import requests

from common import (
    load_config,
    ensure_directories,
    project_path,
    setup_logger,
)



SOURCES = {
    "oni": (
        "https://www.cpc.ncep.noaa.gov/products/"
        "analysis_monitoring/enso/oni/v6/index.php"
    ),

    "dmi": (
        "https://www.cpc.ncep.noaa.gov/products/"
        "international/ocean_monitoring/IODMI/"
        "mnth.ersstv6.clim19912020.dmi_current.txt"
    ),
}


class TableParser(HTMLParser):
    """
    Extract rows from NOAA's HTML tables.
    """

    def __init__(self):
        super().__init__()
        self.rows = []
        self.row = None
        self.cell = None

    def handle_starttag(self, tag, attrs):

        if tag == "tr":
            self.row = []

        elif tag in ("td", "th"):
            if self.row is not None:
                self.cell = []

    def handle_data(self, data):

        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag):

        if tag in ("td", "th"):

            if self.cell is not None:
                value = "".join(self.cell).strip()
                self.row.append(value)
                self.cell = None

        elif tag == "tr":

            if self.row is not None:
                if self.row:
                    self.rows.append(self.row)

                self.row = None



def download_text(url: str) -> str:
    """
    Download an HTML or plain-text NOAA dataset.
    """

    response = requests.get(
        url,
        timeout=(20, 60),
        headers={
            "User-Agent": "Rwanda-Climate-Research/1.0"
        },
    )

    response.raise_for_status()

    if not response.text.strip():
        raise ValueError(
            f"Empty NOAA response: {url}"
        )

    return response.text



def parse_oni_table(
    html: str,
    start_year: int,
    end_year: int,
) -> dict:
    """
    Parse ONI v6 from NOAA HTML.

    NOAA's ONI table provides 12 overlapping
    three-month seasons:

    DJF, JFM, FMA, MAM, AMJ, MJJ,
    JJA, JAS, ASO, SON, OND, NDJ

    These are assigned to central months
    January through December.
    """

    parser = TableParser()
    parser.feed(html)

    records = {}

    for row in parser.rows:

        if len(row) < 2:
            continue

        try:
            year = int(row[0].strip())
        except (ValueError, TypeError):
            continue

        if not start_year <= year <= end_year:
            continue

        available_months = min(
            len(row) - 1,
            12,
        )

        for month in range(1, available_months + 1):

            raw = row[month].strip()

            try:
                value = float(raw)
            except (ValueError, TypeError):
                continue

            if not math.isfinite(value):
                continue

            if abs(value) > 10:
                continue

            key = (year, month)

            if key in records and records[key] != value:
                raise ValueError(
                    f"Conflicting ONI values: {key}"
                )

            records[key] = value

    if not records:
        raise ValueError(
            "No ONI observations parsed from NOAA."
        )

    return records



def parse_dmi_table(
    text: str,
    start_year: int,
    end_year: int,
) -> dict:
    """
    Parse NOAA monthly DMI text.

    Source columns:
        Year Month WTIO SETIO DMI

    Example:
        1999 1 -0.34 0.00 -0.34

    Only the fifth column (DMI) is retained.
    """

    records = {}

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        parts = re.split(r"\s+", line)

        if len(parts) != 5:
            continue

        try:
            year = int(parts[0])
            month = int(parts[1])

            wtio = float(parts[2])
            setio = float(parts[3])
            dmi = float(parts[4])

        except (ValueError, TypeError):
            continue

        if not start_year <= year <= end_year:
            continue

        if not 1 <= month <= 12:
            raise ValueError(
                f"Invalid DMI month: {year}-{month}"
            )

        if not all(
            math.isfinite(v)
            for v in (wtio, setio, dmi)
        ):
            raise ValueError(
                f"Non-finite DMI record: {year}-{month}"
            )

        # Values are rounded in NOAA's text table.
        # Allow small rounding differences.
        calculated = wtio - setio

        if abs(calculated - dmi) > 0.03:
            raise ValueError(
                f"Inconsistent DMI components at "
                f"{year}-{month:02d}: "
                f"WTIO-SETIO={calculated:.3f}, "
                f"DMI={dmi:.3f}"
            )

        key = (year, month)

        if key in records:
            raise ValueError(
                f"Duplicate DMI record: {key}"
            )

        records[key] = dmi

    if not records:
        raise ValueError(
            "No DMI observations parsed from NOAA."
        )

    return records



def validate_historical_data(
    records: dict,
    name: str,
    start_year: int,
    end_year: int,
) -> None:
    """
    Require complete monthly observations for
    all historical years through 2025.

    Current-year and future-month records
    may be incomplete.
    """

    historical_end = min(end_year, 2025)

    missing = []

    for year in range(
        start_year,
        historical_end + 1,
    ):
        for month in range(1, 13):

            if (year, month) not in records:
                missing.append(
                    f"{year}-{month:02d}"
                )

    if missing:
        raise ValueError(
            f"{name.upper()} historical data incomplete. "
            f"Missing {len(missing)} months: "
            + ", ".join(missing[:25])
        )



def validate_reference_values(
    oni: dict,
    dmi: dict,
) -> None:
    """
    Check selected known observations from
    the NOAA historical index tables.

    These checks help detect parsing shifts.
    """

    oni_checks = {
        (1997, 10): 2.2,
        (2015, 11): 2.5,
        (2023, 11): 1.9,
    }

    dmi_checks = {
        (1999, 1): -0.34,
        (1999, 2): 0.01,
        (1999, 12): -0.06,
        (2019, 10): 1.17,
        (2023, 10): 1.01,
    }

    for key, expected in oni_checks.items():

        if key not in oni:
            continue

        actual = oni[key]

        if abs(actual - expected) > 0.011:
            raise ValueError(
                f"ONI reference mismatch at {key}: "
                f"{actual} != {expected}"
            )

    for key, expected in dmi_checks.items():

        if key not in dmi:
            continue

        actual = dmi[key]

        if abs(actual - expected) > 0.011:
            raise ValueError(
                f"DMI reference mismatch at {key}: "
                f"{actual} != {expected}"
            )



def build_monthly_dataframe(
    oni: dict,
    dmi: dict,
    start_year: int,
    end_year: int,
) -> pd.DataFrame:
    """
    Build a complete calendar grid.

    Missing climate observations are NaN,
    never automatically replaced with zero.
    """

    rows = []

    for year in range(
        start_year,
        end_year + 1,
    ):
        for month in range(1, 13):

            rows.append({
                "year": year,
                "month": month,
                "date": f"{year}-{month:02d}-01",
                "oni": oni.get((year, month)),
                "dmi": dmi.get((year, month)),
            })

    df = pd.DataFrame(rows)

    df["oni"] = pd.to_numeric(
        df["oni"],
        errors="coerce",
    )

    df["dmi"] = pd.to_numeric(
        df["dmi"],
        errors="coerce",
    )

    return df



def save_outputs(
    df: pd.DataFrame,
    config: dict,
    logger,
) -> None:
    """
    Save validated CSV and metadata.

    Validation happens before existing
    outputs are replaced.
    """

    index_dir = project_path(
        config["paths"]["climate_indices"]
    )

    index_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        index_dir / "enso_iod_monthly.csv"
    )

    metadata_file = (
        index_dir / "sources_metadata.json"
    )

    metadata = {
        "downloaded_utc": datetime.now(
            timezone.utc
        ).isoformat(),

        "start_year": int(
            config["project"]["start_year"]
        ),

        "end_year": int(
            config["project"]["end_year"]
        ),

        "calendar_records": int(len(df)),

        "oni_available": int(
            df["oni"].notna().sum()
        ),

        "dmi_available": int(
            df["dmi"].notna().sum()
        ),

        "sources": SOURCES,

        "oni_definition": (
            "NOAA CPC ONI v6: overlapping "
            "three-month mean Niño-3.4 "
            "SST anomaly assigned to the "
            "central calendar month."
        ),

        "dmi_definition": (
            "NOAA CPC Dipole Mode Index "
            "calculated from WTIO minus SETIO "
            "sea-surface temperature anomalies."
        ),

        "units": {
            "oni": "degrees Celsius",
            "dmi": "degrees Celsius",
        },

        "missing_values": (
            "Blank CSV cells indicate missing "
            "or unpublished observations."
        ),

        "data_type": "observed_indices",

        "notes": (
            "Latest published values may be revised. "
            "This dataset contains observations, "
            "not future ENSO or IOD forecasts."
        ),
    }

    csv_temporary = (
        index_dir / "enso_iod_monthly.csv.tmp"
    )

    json_temporary = (
        index_dir / "sources_metadata.json.tmp"
    )

    try:
        df.to_csv(
            csv_temporary,
            index=False,
            encoding="utf-8-sig",
            float_format="%.3f",
        )

        json_temporary.write_text(
            json.dumps(
                metadata,
                indent=2,
            ),
            encoding="utf-8",
        )

        # Validate the saved temporary CSV.
        check = pd.read_csv(csv_temporary)

        if len(check) != len(df):
            raise ValueError(
                "Saved CSV row count mismatch."
            )

        if (
            check["oni"].count()
            != df["oni"].count()
        ):
            raise ValueError(
                "Saved ONI count mismatch."
            )

        if (
            check["dmi"].count()
            != df["dmi"].count()
        ):
            raise ValueError(
                "Saved DMI count mismatch."
            )

        # Replace existing outputs only after
        # successful data and file validation.
        csv_temporary.replace(output_file)
        json_temporary.replace(metadata_file)

    finally:
        csv_temporary.unlink(
            missing_ok=True
        )
        json_temporary.unlink(
            missing_ok=True
        )

    logger.info(
        "Saved: %s",
        output_file,
    )

    logger.info(
        "Saved: %s",
        metadata_file,
    )

    print("\nDATASET SUMMARY")
    print(df.count().to_string())

    print("\n2026 MONTHLY OBSERVATIONS")
    print(
        df[df["year"] == 2026]
        .to_string(index=False)
    )

    print("\nSELECTED HISTORICAL YEARS")

    selected = df[
        df["year"].isin([
            1997, 1999, 2015,
            2019, 2023, 2024,
        ])
        & df["month"].isin([10, 11])
    ]

    print(
        selected.to_string(index=False)
    )

    print("\nDISK USAGE")

    for path in (
        output_file,
        metadata_file,
    ):
        print(
            f"{path.name}: "
            f"{path.stat().st_size / 1024:.2f} KB"
        )



def main() -> None:

    config = load_config()
    ensure_directories(config)

    logger = setup_logger(
        "climate_indices",
        "climate_indices.log",
    )

    start_year = int(
        config["project"]["start_year"]
    )

    end_year = int(
        config["project"]["end_year"]
    )


    logger.info(
        "Downloading NOAA ONI v6"
    )

    oni_text = download_text(
        SOURCES["oni"]
    )

    oni = parse_oni_table(
        oni_text,
        start_year,
        end_year,
    )

    validate_historical_data(
        oni,
        "ONI",
        start_year,
        end_year,
    )

    logger.info(
        "ONI observations parsed: %d",
        len(oni),
    )


    logger.info(
        "Downloading NOAA DMI text dataset"
    )

    dmi_text = download_text(
        SOURCES["dmi"]
    )

    dmi = parse_dmi_table(
        dmi_text,
        start_year,
        end_year,
    )

    validate_historical_data(
        dmi,
        "DMI",
        start_year,
        end_year,
    )

    logger.info(
        "DMI observations parsed: %d",
        len(dmi),
    )


    validate_reference_values(
        oni,
        dmi,
    )

    logger.info(
        "NOAA reference checks passed"
    )


    df = build_monthly_dataframe(
        oni,
        dmi,
        start_year,
        end_year,
    )

    expected_rows = (
        end_year - start_year + 1
    ) * 12

    if len(df) != expected_rows:
        raise ValueError(
            "Unexpected monthly record count."
        )


    save_outputs(
        df,
        config,
        logger,
    )

    logger.info(
        "Historical ENSO-IOD collection completed."
    )


if __name__ == "__main__":
    main()
