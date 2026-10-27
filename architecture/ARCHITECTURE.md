# End-to-End System and Workflow Architecture

This document maps the complete data preparation, analysis, training, validation,
forecasting and operational workflows to the current repository. The final CPU
model is recursive VM LightGBM; the final count rule is observed-count persistence.
The sole submission export is [prediction.csv](../predictions/prediction.csv).

Read the [report](../reports/report.md) for findings and the
[methodology](../reports/model_methodology.md) for equations and column usage.
All model fitting and calibration use eligible historical observations. The viewed
holdout does not provide an independent final-model test.

## 1. High-Level System Architecture

The following diagram illustrates the complete data lifecycle: from raw telemetry ingestion to quality checks, feature extraction, cross-validation, and holdout prediction generation.

```mermaid
flowchart TD
    subgraph S1["1. Raw Data Ingestion"]
        R1[("vm_cpu_readings.csv.gz<br>1,340,000 rows")]
        R2[("vm_metadata.csv.gz<br>10,000 rows")]
    end

    subgraph S2["2. Notebook 01: Data Understanding & Cleaning"]
        C1["Data Integrity & Leakage Filter<br>(Exclude 30-day stats & deletion times)"]
        C2["Time Grid Aggregation<br>(10,000 VMs -> 134 Timestamps at 300s)"]
        T1["Target 1: active_vm_count"]
        T2["Target 2: aggregate_cpu_utilization"]
        SPLIT{"Chronological Split"}
        TR["Train Horizon: 110 steps (~9.1 hrs)"]
        HO["Holdout Horizon: 24 steps (2.0 hrs)"]
    end

    subgraph S3["3. Notebook 02: Workload Analysis (EDA)"]
        EDA1["Distributions: 46% Idle, 45% Steady, 22% Hot"]
        EDA2["Autocorrelation: Aggregate 0.66 vs Median VM 0.06"]
        EDA3["Regime Detection: Elevated load at steps 53-72"]
        EDA4["Segment Drivers: 27 Category x Core x Memory combos"]
    end

    subgraph S4["4. Notebook 03: Forecasting Strategies"]
        direction TB
        B1["03_model_comparison.ipynb<br>Baselines: Persistence, Moving Averages, Drift"]
        B2["03_model_comparison.ipynb<br>Aggregate ML (Ridge, ETS, Segment-LightGBM)"]
        B3["03_model_comparison.ipynb<br>Recursive VM LightGBM: 980k examples"]
    end

    subgraph S5["5. Evaluation & Final Deliverables"]
        OUT["predictions/prediction.csv<br>(24 rows x 3 cols)"]
        REP["reports/report.md"]
    end

    R1 & R2 --> C1 --> C2 --> T1 & T2 --> SPLIT
    SPLIT --> TR & HO
    TR --> EDA1 & EDA2 & EDA3 & EDA4
    TR --> B1 & B2 & B3
    B1 & B2 & B3 --> CV["Training-only CV comparison"]
    B3 --> OUT --> REP
    CV --> REP
    HO -. "Future timestamps only" .-> OUT
```

---

## 2. Notebook 01: Data Understanding & Preprocessing
📁 **Location:** `notebooks/01_data_understanding.ipynb`  
📁 **Module Code:** `src/cwfc/quality.py`, `src/cwfc/targets.py`

### Workflow Diagram

```mermaid
flowchart LR
    subgraph Raw["Raw Files"]
        RC["vm_cpu_readings.csv.gz<br>timestamp, vm_id, min_cpu, max_cpu, avg_cpu"]
        RM["vm_metadata.csv.gz<br>11 columns"]
    end

    subgraph Filter["Leakage Filter"]
        LEAK["Excluded (Future Leakage):<br>- metadata avg_cpu, max_cpu, p95_max_cpu<br>- timestamp_vm_deleted"]
        KEEP["Retained for Modeling:<br>- vm_category<br>- virtual_core_bucket<br>- memory_gb_bucket"]
    end

    subgraph Agg["Aggregation (Every 300s Grid)"]
        A1["active_vm_count = Count(distinct vm_id)"]
        A2["aggregate_cpu_utilization = Mean(avg_cpu)"]
    end

    subgraph Output["Output Series (134 rows)"]
        TRN["Train Horizon (Timestamps 0 - 109)"]
        HLD["Holdout Horizon (Timestamps 110 - 133)"]
    end

    RC --> Filter
    RM --> Filter
    Filter --> Agg
    Agg --> Output
```

### Column Specifications

| Input Source | Columns Inspected | Action / Status |
|---|---|---|
| `vm_cpu_readings.csv.gz` | `timestamp`, `vm_id`, `min_cpu`, `max_cpu`, `avg_cpu` | Verified: 0 missing values, strict 300-second cadence, $0 \le \min \le \text{avg} \le \max \le 100$. |
| `vm_metadata.csv.gz` | `vm_category`, `virtual_core_bucket`, `memory_gb_bucket` | Retained as static categorization features. |
| `vm_metadata.csv.gz` | `avg_cpu`, `max_cpu`, `p95_max_cpu`, `timestamp_vm_deleted` | **Excluded from predictors:** CPU summaries extend beyond the cutoff; deletion metadata can reveal future information. |

**Constructed Targets (134 time steps):**
- `active_vm_count`: Distinct VMs with a valid reading in the interval.
- `aggregate_cpu_utilization`: Mean of VM `avg_cpu` readings across the cluster.

---

## 3. Notebook 02: Workload Analysis (EDA)
📁 **Location:** `notebooks/02_workload_analysis.ipynb`  
📁 **Module Code:** `src/cwfc/eda.py`

### Workflow Diagram

```mermaid
flowchart TD
    D110["110 Training Timestamps Only<br>(Holdout strictly excluded)"] --> W1 & W2 & W3 & W4

    W1["Distribution Profiling:<br>- 46.35% Idle (<5% CPU)<br>- 44.95% Steady<br>- 22.35% Bursting (>80% peak)"]
    W2["Temporal Dynamics:<br>- Aggregate Mean: 9.41% CPU (SD: 0.19 pp)<br>- Lag-1 ACF: 0.662 (Aggregate)<br>- Lag-1 ACF: 0.061 (Median VM)"]
    W3["Regime Detection:<br>- Rise detected at index 53 (+0.43 pp)<br>- Drop detected at index 71 (-0.32 pp)"]
    W4["Segment Decomposition:<br>- 27 Category x Core x Memory groups<br>- Interactive: 11.51% CPU<br>- Delay-insensitive: 8.26% CPU"]
```

### Key Analytical Takeaways
1. **The Aggregation Effect:** Individual VM traces have median lag-one autocorrelation $\approx 0.061$, whereas the aggregate has $\text{ACF}_1 = 0.662$. Aggregation smooths individual variation; these statistics alone do not establish an irreducible noise floor.
2. **Mean-Reversion:** Workload moves within a narrow band (mean 9.41%, SD 0.19 pp) and exhibits mean-reverting behavior after transient spikes.

---

## 4. Notebook 03: Baselines and Validation
📁 **Location:** `notebooks/03_model_comparison.ipynb`  
📁 **Module Code:** `src/cwfc/baselines.py`, `src/cwfc/validation.py`

### Expanding-Window Backtest (7 Folds)

| Fold | History indices | Validation indices | Forecast steps |
| --- | --- | --- | ---: |
| 1 | 0–47 | 48–71 | 24 |
| 2 | 0–53 | 54–77 | 24 |
| 3 | 0–59 | 60–83 | 24 |
| 4 | 0–65 | 66–89 | 24 |
| 5 | 0–71 | 72–95 | 24 |
| 6 | 0–77 | 78–101 | 24 |
| 7 | 0–83 | 84–107 | 24 |

Each fold refits using only its history. The 168 decisions cover 60 distinct
observations; overlapping windows and shared VM histories are correlated.


### Baseline Performance on Training-Only CV

| Baseline Model | Forecasting Rule | CPU CV MAE (pp) | CPU CV RMSE (pp) |
|---|---|---:|---:|
| **Moving Average (k=3)** | $\hat{y}_{t+h} = \frac{1}{3}\sum_{i=0}^2 y_{t-i}$ | 0.2439 | 0.2953 |
| **Persistence (Naive)** | $\hat{y}_{t+h} = y_t$ | 0.2655 | 0.3174 |
| **Seasonal Naive (24)** | $\hat{y}_{t+h} = y_{t+h-24}$ | 0.3210 | 0.3611 |

---

## 5. Notebook 03: Aggregate Features and Candidate Models
📁 **Location:** `notebooks/03_model_comparison.ipynb`  
📁 **Module Code:** `src/cwfc/features.py`, `src/cwfc/models.py`

### Feature Pipeline & Model Comparison

```mermaid
flowchart LR
    subgraph FeatEng["Feature Extraction at Forecast Origin (t)"]
        L["Lags: lag_1 to lag_12"]
        R["Rolling Statistics: mean_w, std_w for w in (3, 6, 12)"]
        S["Rolling Slopes: slope_6, slope_12"]
        C["Segment Mix: category CPU means & population shares"]
    end

    subgraph Models["Candidate Models Tested"]
        RIDGE["Direct Ridge Regression<br>(39 features per horizon step)"]
        ETS["Damped ETS / SES"]
        THETA["Theta Method"]
        LGB_SEG["Segment LightGBM<br>(27 pooled segment series)"]
        ENS["Sequential Ensemble"]
    end

    FeatEng --> Models
    Models --> RES["Compare MAE, RMSE and persistence skill on identical cases;<br>the final forecast uses recursive VM LightGBM."]
```

### Feature Inventory (39 Predictors Available at Origin $t$)
1. **Historical Lags:** $y_{t-1}, y_{t-2}, \dots, y_{t-12}$ (past 60 minutes).
2. **Rolling Aggregates:** Mean and sample standard deviation across windows $w \in \{3, 6, 12\}$.
3. **Trend Slopes:** OLS linear slopes over 6 and 12 steps.
4. **Category Breakdown:** Real-time observed mean CPU and proportion of VMs in `Interactive`, `Delay-insensitive`, and `Unknown`.

---

## 6. Notebook 03: Final Recursive VM LightGBM
📁 **Location:** `notebooks/03_model_comparison.ipynb`  
📁 **Module Code:** `src/cwfc/model.py`

### End-to-End Per-VM Workflow

```mermaid
flowchart TD
    subgraph DataPrep["1. Panel Matrix Construction"]
        RAW["10,000 VMs x 110 Training Timestamps<br>= 1,100,000 raw readings"]
        ALIGN["Pivot to Matrix: 10,000 rows x 110 cols"]
    end

    subgraph TrainSamples["2. Supervised Training (Anchor indices 11 to 108)"]
        X["Features (25 per VM per timestamp):<br>- VM lags 1..12<br>- VM rolling mean & std (3, 6, 12)<br>- VM slopes (6, 12)<br>- vm_history_mean<br>- aggregate_cpu_context<br>- Metadata: vm_category, core_bucket, memory_bucket"]
        Y["Target: next_vm_cpu - vm_history_mean<br>(980,000 training pairs)"]
    end

    subgraph Training["3. LightGBM Training"]
        LGBM["LightGBM Regressor<br>120 trees, lr=0.05, 31 leaves<br>Objective: Regression (L2)"]
    end

    subgraph Infer["4. Recursive Inference (Holdout Steps 1 to 24)"]
        STEP["For step h = 1..24:<br>1. Query 25 features for each VM<br>2. Predict next step CPU for all 10,000 VMs<br>3. Clip to [0, 100]<br>4. Append to matrix and recurse"]
        AGG["5. Bottom-Up Aggregation:<br>Aggregate CPU(h) = Mean(10,000 VM predictions at h)"]
    end

    DataPrep --> TrainSamples
    TrainSamples --> Training
    Training --> Infer
    Infer --> STEP --> AGG
```

### Training vs. Inference Column Breakdown

```
Input Features (25):
├── Telemetry and Context (22)
│   ├── lag_1 to lag_12         (Individual VM past CPU values)
│   ├── mean_3, std_3           (3-step rolling statistics)
│   ├── mean_6, std_6           (6-step rolling statistics)
│   ├── mean_12, std_12         (12-step rolling statistics)
│   ├── slope_6, slope_12       (Recent trend trajectory)
│   ├── vm_history_mean         (Individual VM's historical baseline)
│   └── aggregate_cpu_context   (Overall cluster CPU level at anchor)
└── Static Metadata (3)
    ├── vm_category             (Interactive, Delay-insensitive, Unknown)
    ├── virtual_core_bucket     (Bucketed vCPU cores)
    └── memory_gb_bucket        (Bucketed RAM)

Target (1):
└── residual = next_vm_cpu - vm_history_mean
```

### Feature Importance Ranking (Empirical Gain)

```
vm_history_mean        ██████████████████████████████ 70.4M
lag_1                  █████████████████████████ 60.5M
slope_12               █████████████████ 40.2M
mean_12                ██ 6.6M
mean_3                 ██ 5.8M
lag_2                  ██ 5.8M
...
memory_gb_bucket       ▍ 1.35M
vm_category            ▎ 0.65M
virtual_core_bucket    ▏ 0.08M  (Lowest importance)
```

---

## 7. Modeling Units and Final Model Selection

All fitting workflows are retained in [notebook 03](../notebooks/03_model_comparison.ipynb).

| Approach | Modeling unit | Inputs | Forecast strategy |
| --- | --- | --- | --- |
| Baselines | Aggregate target histories | Observed target values | Persistence, moving averages, drift and seasonal negative control |
| Aggregate statistical/linear models | Fleet series | Historical levels or origin-safe predictors | SES, ETS, Theta and per-horizon ridge |
| Segment LightGBM | 27 category/core/memory series | Segment dynamics, static labels, aggregate context and horizon | Shared direct multi-horizon model, population-weighted aggregation |
| Final VM LightGBM | 10,000 VM histories | 25 VM/context/category features | Shared next-step residual model, recursive prediction and unweighted aggregation |
| Direct VM ablations | Individual VM/anchor/horizon examples | Base, peak/history additions or category removal | Horizon-conditioned residual prediction without recursive CPU feedback |

The final VM model has 980,000 supervised examples on this release. Those
examples do not create additional independent observation periods. Settings are
120 trees, learning rate 0.05, 31 leaves, minimum 100 child samples and seed 42.

| CPU forecast / benchmark | CV MAE (pp) | CV RMSE (pp) |
| --- | ---: | ---: |
| Final recursive VM LightGBM | 0.19881 | 0.26889 |
| Persistence benchmark | 0.26550 | 0.31736 |
| Moving-average benchmark (3) | 0.24385 | 0.29532 |
| Segment LightGBM comparison | 0.26926 | 0.30750 |

Selection uses pooled chronological CV MAE, with RMSE and model-name tie breaks.
The current final recipe remains [final_model.yaml](../configs/final_model.yaml).
Observed-count persistence is implemented in [predict.py](../src/cwfc/predict.py).
The Holt-count comparator is retained in [count_model.py](../src/cwfc/count_model.py)
and notebook 03; its zero error on this constant population does not establish churn skill.

## 8. Notebook 04: Error and Failure Analysis

**Notebook:** [04_error_analysis.ipynb](../notebooks/04_error_analysis.ipynb)  
**Code:** [analysis.py](../src/cwfc/analysis.py), [validation.py](../src/cwfc/validation.py)

```mermaid
flowchart TD
    CV["Saved final-model CV cases"] --> CHECK["Align actuals to training histories"]
    TRAIN["Training-only target histories"] --> CHECK
    CHECK --> METRIC["MAE, RMSE, sMAPE, bias and origin-scaled MASE"]
    CHECK --> LOAD["Origin-specific Q80 load labels"]
    CHECK --> WINDOW["Training-defined onset, reversion and last-CV windows"]
    METRIC --> BREAK["Horizon and fold diagnostics"]
    LOAD --> BREAK
    WINDOW --> STORIES["Transition failures and signed bias"]
    VM["Saved individual CV forecasts"] --> SEG["Category/core/memory aggregation"]
    SEG --> RECON["Reconcile segment-weighted predictions to fleet forecasts"]
    BREAK & STORIES & RECON --> FIG["Tables and figures under artifacts/analysis and reports/figures"]
    FIG --> REPORT["reports/report.md"]
```

Origin-safe Q80 labels differ from the full-training retrospective high-load
labels. Negative bias means underprediction. MASE uses each origin's historical
one-step naive error scale; zero scales are undefined. The transition windows
are selected from workload changes rather than residuals. Per-VM and behavior
errors expose mistakes that may cancel in the fleet mean.

Previously recorded persistence holdout observations are retained as benchmark
context. They are not final-model holdout results and do not choose the final
model. No new holdout evaluation occurs in this analysis path.

## 9. Supplementary Activity, Risk and Capacity Workflows

**Training code:** [direct.py](../src/cwfc/direct.py),
[operational.py](../src/cwfc/operational.py)  
**Evaluation code:** [operational_evaluation.py](../src/cwfc/operational_evaluation.py),
[capacity.py](../src/cwfc/capacity.py), [uncertainty.py](../src/cwfc/uncertainty.py)

```mermaid
flowchart TD
    PAST["Origin-bounded VM histories"] --> DIRECT["Matured direct CPU / horizon examples"]
    DIRECT --> ABLATE["Base, peaks, history and no-category CPU variants"]
    DIRECT --> BUSY["Binary LightGBM: CPU-busy probability"]
    PAST --> SCREEN["Last 12 averages below 1% and maxima below 5%"]
    SCREEN --> EVENT["Binary LightGBM: any crossing in complete next 24 slots"]
    BUSY & EVENT --> OOF["Earlier-origin probabilities with matured outcomes"]
    OOF --> CAL["Optional Platt calibration"]
    CAL --> EVAL["Brier / log loss, reliability and expected busy counts"]
    EVENT --> REVIEW["Risk budgets, matched-size rules and hysteresis"]
    ABLATE --> CPUCV["Shared chronological CPU comparison"]
    CPUCV --> HEAD["Selected-model fixed / empirical headroom scenarios"]
    EVAL & REVIEW & HEAD --> NB["Notebook 04 operational interpretation"]
```

Point-busy labels mature at each forecast observation. Activity-event labels
require the entire 24-step future window. Calibration excludes current-fold and
unmatured outcomes. Both classifiers exclude workload category.

The optional full training run is controlled by `RETRAIN_OPERATIONAL` in notebook
03 or `python -m cwfc.operational`. Supporting CPU/activity forecasts, candidate
lists, model fits and diagnostics stay under `artifacts/operational/`.

Category quantile scenarios are a separate training-only persistence experiment.
[capacity.yaml](../configs/capacity.yaml) defines cost-ratio quantiles, margins,
fixed-buffer comparators and a slope-trigger scenario. Exact-horizon P90 requires
nine matured residuals; seven folds cannot support it. Unsupported bounds use
100% CPU and explicitly expose the calibration gap. This differs from the
selected-model empirical margin, which pools six-step horizon bands and uses a
0.5 pp warmup margin.

The operational results support demand planning and human review. CPU shortfalls
are not observed SLA breaches, candidate VM-hours overlap, and resource buckets
do not identify exact cores or safe shutdown eligibility.

## 10. Outputs and Reproduction

| Output | Purpose |
| --- | --- |
| [predictions/prediction.csv](../predictions/prediction.csv) | Sole final 24-row, three-column forecast |
| `artifacts/final_model/` | Final-model cases, individual predictions, feature gains and provenance |
| `artifacts/comparison/` | Retained candidate-comparison evidence |
| `artifacts/model_selection/` | Notebook-generated statistical/segment comparison and ensemble evidence |
| `artifacts/operational/` | Busy/risk predictions, candidate decisions, headroom and category bounds |
| `artifacts/analysis/` | Rebuilt final-model residual and policy summaries |
| [reports/report.md](../reports/report.md) | Complete findings, failures and operational recommendations |
| [reports/model_methodology.md](../reports/model_methodology.md) | Column usage, equations and methodological definitions |

```powershell
python -m cwfc.predict
python -m cwfc.validation
python -m cwfc.analysis
python -m cwfc.capacity
```

Run the four notebooks in order for the complete exploratory and training
workflow. See [README](../README.md) for installation and data preparation.
The forecast manifest records input, configuration, source and output hashes.
The final CSV contains `timestamp`, `active_vm_count` and
`aggregate_cpu_utilization`; counts are nonnegative and CPU is in [0,100].
