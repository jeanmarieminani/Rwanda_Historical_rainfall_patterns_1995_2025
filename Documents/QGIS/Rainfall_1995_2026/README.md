\# Rwanda Historical Rainfall Patterns (1995–2025)



\*\*CHIRPS rainfall variability, dry-to-wet seasonal transitions, ENSO/IOD teleconnections, and exploratory forecasts for 2026\*\*



This reproducible Python/QGIS research project studies Rwanda's historical monthly rainfall from \*\*1995–2025\*\* and uses available \*\*2026 observations\*\* for ongoing monitoring and exploratory forecasting. Particular attention is given to the \*\*June–September (JJAS) dry season\*\*, the \*\*July–September (JAS) dryness indicator\*\*, and the \*\*October–November (ON)\*\* and \*\*October–December (OND)\*\* rainy-season periods.



\## Research question



> Can unusually dry conditions during July–September in Rwanda be followed by exceptionally heavy rainfall during October–November or October–December, and do ENSO and the Indian Ocean Dipole (IOD) help predict that transition?



A dry season \*\*does not automatically imply\*\* an unusually wet following season. This project tests that proposition statistically rather than assuming it.



\## Data



| Data | Period | Role | Source |

|---|---|---|---|

| CHIRPS v3 monthly precipitation, Africa grid | 1995–2025; available months of 2026 | Rwanda rainfall totals and anomalies | \[Climate Hazards Center](https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/) |

| Rwanda national boundary (`RWA\_adm0.\*`) | Static | Spatial clipping and aggregation | See `data/boundaries/` and provenance note below |

| NOAA CPC ONI v6 | 1995–available 2026 months | ENSO index (three-month running mean) | \[NOAA CPC ONI](https://www.cpc.ncep.noaa.gov/products/analysis\_monitoring/enso/oni/v6/index.php) |

| NOAA CPC monthly DMI | 1995–available 2026 months | Indian Ocean Dipole index | \[NOAA CPC IOD](https://www.cpc.ncep.noaa.gov/products/international/ocean\_monitoring/IODMI/) |



\*\*Climatological reference period:\*\* 1995–2024 (30 years). Seasonal rainfall is measured in \*\*mm\*\*. Monthly Rwanda values are derived from CHIRPS grid cells using area weighting; boundary-grid-cell coverage is approximated (not fractional-area weighted).



\*\*Data availability as of 10 October 2026:\*\* CHIRPS and the saved ONI/DMI datasets contain observations from January 1995 through August 2026 (380 months). September–December 2026 are missing in the saved datasets and \*\*must not be interpreted as zero or as observed forecasts\*\*. Download availability can change.



\## Selected results



\- Rwanda's \*\*1995–2024 mean annual precipitation\*\* in the processed CHIRPS data is \*\*1,285.84 mm\*\*.

\- The annual rainfall trend for \*\*1995–2025\*\* is approximately \*\*+3.86 mm/year\*\*; it is \*\*not statistically significant\*\* (linear-regression \*p\* ≈ 0.207; \*R²\* ≈ 0.054).

\- Under a national-scale \*\*JJAS below 20th percentile / ON above 80th percentile\*\* definition, \*\*2024\*\* was the only identified reversal in 1995–2025.

\- Under the stricter \*\*JAS below 20th percentile / ON above 90th percentile\*\* definition, \*\*0\*\* transitions were detected among 31 complete years. There were \*\*3 exceptional ON rainfall years\*\* in total.

\- The contemporaneous October–November rainfall association model with ONI, DMI, and their interaction had \*\*R² = 0.0883\*\* and \*\*adjusted R² = −0.013\*\* (31 annual observations). Neither individual index nor their interaction had a statistically significant coefficient in that model.

\- Exploratory ML predictions of exceptional ON rainfall \*\*did not outperform climatology\*\* in historical testing: Brier scores \*\*0.1586\*\* (logistic regression) vs \*\*0.1105\*\* (climatology), lower is better. Do not interpret the exploratory 2026 ML probability as a calibrated operational forecast.



These national, monthly/seasonal results \*\*do not constitute flood forecasts\*\*. Localized flooding and landslide hazards require daily/sub-daily rainfall, topography, soil/river conditions, and exposure information.



\## Figures



The repository retains the principal figures in \[`outputs/figures`](outputs/figures). Large paged copies of the all-years grid are generated locally but are not tracked by default.



\### All-year monthly profiles



!\[Monthly Rwanda rainfall profiles (1995–2026)](outputs/figures/07\_all\_years\_monthly\_profiles\_grid.png)



\### Rainfall seasonality and changes



!\[Monthly rainfall climatology](outputs/figures/01\_monthly\_rainfall\_climatology.png)



!\[Annual rainfall trend](outputs/figures/02\_annual\_rainfall\_trend.png)



!\[Monthly rainfall anomaly heatmap](outputs/figures/03\_monthly\_rainfall\_anomaly\_heatmap.png)



\### Dry-to-wet transitions and teleconnections



!\[JJAS and ON rainfall anomalies](outputs/figures/04\_JJAS\_ON\_rainfall\_anomalies.png)



!\[JAS–ON extreme transition](outputs/figures/08\_JAS\_ON\_extreme\_transition.png)



!\[ON rainfall / ENSO / IOD associations](outputs/figures/07\_ON\_rainfall\_ENSO\_IOD\_scatter.png)



\*Panel colors, thresholds, and periods follow each figure's caption and generating code. JJAS, JAS, ON, and OND are not interchangeable seasons.\*



\## Project structure



```text

Rainfall\_1995\_2026/

├── config/settings.json

├── src/

│   ├── common.py

│   ├── 01\_collect\_chirps.py

│   ├── 02\_build\_series.py

│   ├── 03\_plots.py

│   ├── 04\_teleconnections.py

│   ├── 05\_event\_timeline.py

│   ├── 06\_ml\_rainfall\_forecast.py

│   ├── 07\_collect\_enso\_iod.py

│   ├── 08\_dry\_to\_extreme\_wet\_prediction.py

│   └── 09\_plot\_all\_years\_monthly\_profiles.py

├── data/

│   ├── boundaries/                  # Boundary files, subject to source license

│   ├── indices/                     # Compact monthly ONI/DMI and metadata

│   ├── raw/chirps/                  # Downloaded rasters (not in Git)

│   └── processed/                   # Clipped GeoTIFFs (not in Git)

├── outputs/

│   ├── figures/                     # Selected PNG figures tracked

│   ├── tables/                      # Lightweight CSV results tracked

│   └── maps/

├── logs/                            # Not in Git

├── tests/

├── requirements.txt

└── .gitignore

```



Some optional placeholder scripts (for example `05\_event\_timeline.py`) may not yet be implemented. The reproducible core is the CHIRPS and indices collection, seasonal analysis, plotting, teleconnections, and exploratory forecasting scripts.



\## Setup and reproduction (Windows PowerShell)



```powershell

\# From the cloned repository root

py -3.12 -m venv .venv

.\\.venv\\Scripts\\Activate.ps1

python -m pip install --upgrade pip

python -m pip install -r requirements.txt

```



The CHIRPS rasters are \*\*not distributed in Git\*\* because they can be downloaded from the original provider. Make sure `data/boundaries/RWA\_adm0.shp` and its companion files are present, with correct source/licensing documentation, before running the pipeline.



```powershell

\# Download and clip available CHIRPS months for Rwanda

python .\\src\\01\_collect\_chirps.py --start-year 1995 --end-year 2026



\# Create monthly, annual, and seasonal rainfall tables

python .\\src\\02\_build\_series.py



\# Rebuild primary plots

python .\\src\\03\_plots.py



\# Collect compact ONI and DMI index files

python .\\src\\07\_collect\_enso\_iod.py



\# Historical ENSO/IOD associations

python .\\src\\04\_teleconnections.py



\# Exploratory rainfall amount ML forecast

python .\\src\\06\_ml\_rainfall\_forecast.py



\# JAS-to-ON extreme-event frequencies and classifier

python .\\src\\08\_dry\_to\_extreme\_wet\_prediction.py



\# All-years monthly rainfall profiles

python .\\src\\09\_plot\_all\_years\_monthly\_profiles.py

```



Note that scripts use existing input CSVs and reference paths defined in `config/settings.json`. Their outputs are not replacements for official early-warning advisories. Date cutoffs and index publication lags matter in historical forecasting experiments.



\## Definitions and limitations



\- \*\*JJAS:\*\* June–September; \*\*JAS:\*\* July–September; \*\*ON:\*\* October–November; \*\*OND:\*\* October–December.

\- \*Unusually dry\* and \*exceptionally wet\* are defined using empirical percentiles of the study's rainfall climatology. Percentile definitions are study-specific and can be changed.

\- Seasonal associations and regression coefficients \*\*do not establish causation\*\*.

\- The 2026 September–December period is incomplete in the saved observations; avoid treating it as a complete season.

\- Seasonal averages can hide localized rainfall extremes; national results should not be interpreted as district-level hazard estimates.

\- Very few observed extreme ON events (3 over 31 years) limit classification performance and confidence.



\## Data rights and citation



CHIRPS and NOAA data remain credited to their respective providers. Before committing `data/boundaries/RWA\_adm0.\*`, verify the original boundary provider's license permits redistribution and add its proper attribution. If redistribution is not permitted, \*\*exclude boundary files from Git\*\* and provide acquisition instructions instead.



Please cite the original CHIRPS and NOAA datasets in any publication and record the extraction date, version, baseline and processing choices used.



\## Author and repository



Repository: \[jeanmarieminani/Rwanda\_Historical\_rainfall\_patterns\_1995\_2025](https://github.com/jeanmarieminani/Rwanda\_Historical\_rainfall\_patterns\_1995\_2025)



Research and Python analysis: Jean-Marie Vianney Minani.



\*\*Status:\*\* Research code and exploratory analysis; not an operational flood-warning service.



