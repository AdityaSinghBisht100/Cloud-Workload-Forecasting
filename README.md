# Cloud Workload Forecasting

**Author:** Aditya Singh Bisht

Forecasting and workload analysis for a curated Microsoft Azure VM dataset. The project predicts the next **24 five-minute intervals** of fleet-average CPU utilization and observed VM count, then examines forecast failures and their implications for capacity planning.

The final CPU model is a **shared recursive LightGBM**: it learns from individual VM histories, predicts each VM's next CPU value, and averages those predictions into a fleet forecast. Observed VM count uses the latest valid population count.

**Start here:** [Final report](reports/report.md) · [Model methodology](reports/model_methodology.md) · [Architecture](architecture/ARCHITECTURE.md) · [Final predictions](predictions/prediction.csv)

## Main findings

The release contains **10,000 VMs and 1,340,000 readings**. The first 110 observations provide training and chronological validation history; the remaining 24 define the forecast horizon.

| Training-only validation | Final LightGBM | Persistence benchmark |
| --- | ---: | ---: |
| CPU MAE, percentage points | **0.19881** | 0.26550 |
| CPU RMSE, percentage points | **0.26889** | 0.31736 |
| MAE improvement over persistence | **25.12%** | — |

These results use seven expanding-window folds. The final model improves pooled accuracy, but underpredicts **93.33% of retrospectively identified high-load cases**. Its high-load MAE is **0.37269 percentage points**. The holdout was already inspected before this model's experiment, so an independent final-model holdout result is not claimed.

Workload analysis also finds that low average CPU can coexist with bursts. Bonus experiments investigate CPU-busy counts, activity-event probabilities, low-use review candidates and headroom. Their findings support planning and human review; they do not establish safe automatic provisioning or shutdown.

## Quick start

Use **Python 3.11 or newer**. Run commands from this repository's root directory.

### 1. Install the project

In PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

### 2. Prepare the dataset

Extract the supplied challenge release into this structure:

```text
challenge_data/
├── vm_cpu_readings.csv.gz
├── vm_metadata.csv.gz
├── DATA_CARD.md
└── SHA256SUMS
```

See the [dataset contract](DATASET.md), [release instructions](DATA_RELEASE.md) and [source attribution](SOURCES.md). The prediction command checks supplied release checksums. Raw data files are excluded from Git.

### 3. Verify and generate the forecast

```powershell
python -m pytest -q
python -m cwfc.predict
```

The forecast is written to **`predictions/prediction.csv`**. The command also saves individual VM forecasts, feature gains and a provenance manifest under `artifacts/final_model/`.

To specify paths explicitly:

```powershell
python -m cwfc.predict --data challenge_data --config configs/final_model.yaml --output predictions/prediction.csv
```

## Final prediction file

The CSV contains exactly **24 rows** and these three columns:

| Column | Meaning |
| --- | --- |
| `timestamp` | Relative seconds identifying the forecast observation |
| `active_vm_count` | Forecast number of distinct VMs with valid observations |
| `aggregate_cpu_utilization` | Unweighted mean of individual VM CPU forecasts, in percent |

CPU predictions are bounded to `[0, 100]`; counts are nonnegative. On this complete-trace release, observed count remains 10,000. That count describes the dataset population and is separate from CPU-busy demand or required provisioned capacity.

## Explore the notebooks

Open JupyterLab after activating the environment:

```powershell
jupyter lab
```

Read or run the notebooks in order:

| Notebook | What it contains |
| --- | --- |
| [01 — Data understanding](notebooks/01_data_understanding.ipynb) | Checksums, missing values, duplicates, CPU validity, coverage, preprocessing and leakage checks |
| [02 — Workload analysis](notebooks/02_workload_analysis.ipynb) | Distributions, bursts, autocorrelation, change scans, segments, concentration, peak windows and their contributors |
| [03 — Model comparison](notebooks/03_model_comparison.ipynb) | Actual baseline, statistical and segment-model fitting; LightGBM feature construction, final training, chronological refits, count experiments and direct/activity-model comparisons |
| [04 — Error analysis](notebooks/04_error_analysis.ipynb) | Horizon/fold errors, regime transitions, segment and individual errors, calibration, candidate policies, headroom and quantile readiness |

The notebooks retain explanations, executable code, tables and plots. Notebook 03 performs model fitting when run. Its full direct-model and activity-classifier experiment has an explicit `RETRAIN_OPERATIONAL` switch because it requires additional fitting. The recorded operational evidence is included for inspection.

## Reproduce validation and supporting analysis

These commands serve different purposes:

| Command | Result |
| --- | --- |
| `python -m cwfc.predict` | Fit the final model and export the final forecast |
| `python -m cwfc.validation` | Refit the final model at seven chronological origins and compare with persistence |
| `python -m cwfc.analysis` | Rebuild residual diagnostics, failure windows, segment summaries, review-policy summaries, headroom comparisons and figures from retained evidence |
| `python -m cwfc.capacity` | Recalculate category quantile bounds and calibration-readiness scenarios |
| `python -m cwfc.operational` | Refit the supplementary direct CPU models and activity classifiers, with outputs under `artifacts/operational/` |

Run notebook 03 for the complete aggregate/statistical/segment candidate comparison. Supplementary experiment outputs remain under `artifacts/`; the final submission stays in `predictions/prediction.csv`.

## Repository guide

```text
architecture/     System diagrams, data flow and workflow explanations
configs/          Model and operational scenario settings
notebooks/        Four complete analysis and training notebooks
predictions/      Final prediction.csv
reports/          report.md, model_methodology.md and figures
src/cwfc/         Data preparation, forecasting, comparisons and analysis
tests/            Preprocessing, leakage, validation and pipeline tests
artifacts/        Supporting evidence, generated forecasts and provenance
challenge_data/   Supplied dataset and its release metadata
```

Compact evidence tables and report figures are retained for review. Large input files, fitted models, individual prediction artifacts and generated caches are excluded from Git. The forecast manifest records data, configuration, source and output hashes.

## How to interpret the results

Validation windows overlap, VM observations are correlated, and model selection used the same limited CV history. Complete-trace selection hides real-world churn. Resource buckets do not identify exact core demand or physical-host placement.

Read the [report](reports/report.md) for the full workload findings, failure analysis and operational conclusions. Use the [methodology](reports/model_methodology.md) for dataset-column usage, feature equations, recursive forecasting, validation rules and supplementary experiment definitions.
