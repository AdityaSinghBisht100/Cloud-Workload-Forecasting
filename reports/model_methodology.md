# Final model methodology: dataset columns and equations

**Scope:** The final recursive VM LightGBM CPU forecast and observed-count
persistence rule, configured in
[final_model.yaml](../configs/final_model.yaml) and generating
[prediction.csv](../predictions/prediction.csv).

This document explains the implemented calculations, from source columns to
individual predictions and fleet aggregation. Measured results and operational
interpretations are in [report.md](report.md). The
[model-comparison notebook](../notebooks/03_model_comparison.ipynb) displays
fitting evidence, feature gains and forecast output; the
[error-analysis notebook](../notebooks/04_error_analysis.ipynb) shows failure
patterns.

**Reading the equations:** formulas use LaTeX inside Markdown `$...$` and
`$$...$$` delimiters. Each displayed formula also has a plain-text equivalent.

## 1. What one observation means

`vm_id` identifies a virtual machine, not an individual physical CPU. The same
anonymous identifier connects that VM's readings over time and its metadata.
For a VM $v$ and five-minute time index $t$, define:

$$
y_{v,t}=\operatorname{avg\_cpu}_{v,t},\qquad
m_{v,t}=\operatorname{min\_cpu}_{v,t},\qquad
M_{v,t}=\operatorname{max\_cpu}_{v,t}.
$$

Plain-text equivalent:

```text
y[v,t] = avg_cpu of VM v at time t
m[v,t] = min_cpu of VM v at time t
M[v,t] = max_cpu of VM v at time t
```

All three are utilization percentages for the observation interval. Relative
timestamps increase by 300 seconds; no absolute time-of-day is assumed.
Let $V_t$ be the set of VMs with a valid observation at time $t$.
The assignment's two targets are:

$$
N_t=|V_t|,\qquad A_t=\frac{1}{N_t}\sum_{v\in V_t}y_{v,t}.
$$

Plain-text equivalent:

```text
N[t] = number of distinct VMs with valid readings at time t
A[t] = sum(avg_cpu of those VMs at t) / N[t]
```

The assignment calls $A_t$ `aggregate_cpu_utilization`. Its mathematical
operation is an **unweighted mean across VMs**, not a sum, core-weighted mean,
or total CPU consumption. Bucketed core counts are not used as weights.
For timestamp 0, this calculation gives **9.396897973701%** across 10,000 VMs.

## 2. How every readings column is used

| Readings column | Implemented use | Direct current LightGBM predictor? |
|---|---|---|
| `timestamp` | Chronological ordering, five-minute grid, lags and forecast cutoffs | No raw timestamp feature |
| `vm_id` | Separate each VM's history; join metadata; label individual forecasts | No raw ID feature |
| `avg_cpu` | CPU labels, lag/rolling/trend features, historical baseline and population context | Yes, through the features below |
| `min_cpu` | Validate CPU ordering and completeness | No |
| `max_cpu` | Validate ordering/completeness; peak and hot-VM analysis | No |

Accepted records must satisfy:

$$
0\le m_{v,t}\le y_{v,t}\le M_{v,t}\le100.
$$

Plain-text equivalent:

```text
0 <= min_cpu[v,t] <= avg_cpu[v,t] <= max_cpu[v,t] <= 100
```

Missing keys and invalid CPU values are rejected. A row missing any CPU field
is excluded without imputing its label. Duplicate VM/timestamp records are
averaged separately for each CPU column. For duplicate group $D_{v,t}$:

$$
x^{\mathrm{clean}}_{v,t}
=\frac{1}{|D_{v,t}|}\sum_{r\in D_{v,t}}x_r,
\qquad x\in\{m,y,M\}.
$$

Plain-text equivalent:

```text
clean_value[v,t] = sum(values in duplicate VM/timestamp group) / group_size
Apply separately to min_cpu, avg_cpu and max_cpu.
```

The full five-minute grid is retained. A wholly unobserved timestamp has zero
observed VMs and missing CPU, rather than an invented zero CPU measurement.
See [quality.py](../src/cwfc/quality.py) and [targets.py](../src/cwfc/targets.py).

`max_cpu` also defines descriptive operational targets:

$$
\operatorname{HotVMShare}_t
=\frac1{N_t}\sum_{v\in V_t}\mathbf1(M_{v,t}>80),
\qquad
\operatorname{MeanPeakCPU}_t=\frac1{N_t}\sum_{v\in V_t}M_{v,t}.
$$

Plain-text equivalent:

```text
HotVMShare[t] = count(VMs with max_cpu > 80% at t) / N[t]
MeanPeakCPU[t] = sum(max_cpu of observed VMs at t) / N[t]
```

These are descriptive analysis outputs. The final VM LightGBM uses historical
interval-average CPU for its numerical predictors; interval minimum and maximum
CPU support validation and workload analysis.

## 3. How every metadata column is used

| Metadata column | Current role | Reason or calculation |
|---|---|---|
| `vm_id` | Join key and VM-history identifier | Match readings to the same VM's metadata, not by row position |
| `vm_category` | Categorical predictor | Workload category context |
| `virtual_core_bucket` | Categorical predictor | VM size context; bucket label is not treated as an exact core weight |
| `memory_gb_bucket` | Categorical predictor | VM memory-size context |
| `subscription_id` | Descriptive concentration analysis | Group VMs by subscription; not a current predictor |
| `deployment_id` | Descriptive concentration analysis | Group VMs by deployment; not a current predictor |
| `timestamp_vm_created` | Lifetime/censoring audit | All released values are zero; excluded from forecasting |
| `timestamp_vm_deleted` | Lifetime audit | Future deletion information is excluded from forecasting |
| `avg_cpu` | Excluded from model inputs | Metadata lifetime summary extends beyond the training cutoff |
| `max_cpu` | Excluded from model inputs | Same lifetime-information leakage risk |
| `p95_max_cpu` | Excluded from model inputs | Same lifetime-information leakage risk |

The metadata `avg_cpu` is different from the timestamped readings `avg_cpu`.
The model uses the timestamped training readings and computes its own past-only
historical mean; it does not use the lifetime metadata average.

For each of the three categorical predictors, the implementation assigns a
category code and tells LightGBM that the column is categorical:

$$
C_v=[\operatorname{code}(\text{category}_v),
     \operatorname{code}(\text{core bucket}_v),
     \operatorname{code}(\text{memory bucket}_v)].
$$

Plain-text equivalent:

```text
C[v] = [category_code[v], core_bucket_code[v], memory_bucket_code[v]]
```

The codes are identifiers, not numerical magnitudes. One model learns across
VMs, while each training row retains the features of its own VM.

For descriptive subscription/deployment concentration, group $g$'s share is:

$$
S_g=\frac{|\{v:v\text{ belongs to }g\}|}{|V|}.
$$

Plain-text equivalent:

```text
GroupShare[g] = number of VMs in group g / total number of VMs
```

The lifetime audit includes duration and a post-window deletion flag:

$$
D_v=d_v-c_v,\qquad I_v^{\mathrm{after}}=\mathbf1(d_v>T_{\mathrm{window}}),
$$

Plain-text equivalent:

```text
Duration[v] = deletion_timestamp[v] - creation_timestamp[v]
DeletedAfterWindow[v] = 1 if deletion_timestamp[v] > window_end; otherwise 0
```

where $c_v$ and $d_v$ are the supplied creation/deletion timestamps.
These audit calculations do not enter forecasting features. Source creation
times of zero do not establish that those VMs were newly created at trace start.

## 4. The 25 VM-level forecasting inputs

Features at anchor $t$ predict $t+1$. Feature calculations may forward-fill
missing historical CPU from earlier observations only; write this feature
series as $\tilde y$. Observed CPU labels remain unfilled.

**Twelve lags** retain individual recent observations:

$$
L_{v,t,k}=\tilde y_{v,t-k+1},\qquad k=1,\ldots,12.
$$

Plain-text equivalent:

```text
Lag[v,t,k] = past-filled avg_cpu[v,t-k+1], for k = 1,...,12
Lag 1 is the observation at the current anchor t.
```

Here lag 1 is the last value available at the prediction anchor.

**Six rolling statistics** summarize level and variability, for
$w\in\{3,6,12\}$:

$$
\mu_{v,t,w}=\frac1w\sum_{j=0}^{w-1}\tilde y_{v,t-j},
$$

Plain-text equivalent:

```text
RollingMean[v,t,w] = sum(last w past-filled CPU values of VM v) / w
w = 3, 6 or 12
```

$$
\sigma_{v,t,w}=\sqrt{
\frac{\sum_{j=0}^{w-1}(\tilde y_{v,t-j}-\mu_{v,t,w})^2}{w-1}}.
$$

Plain-text equivalent:

```text
RollingSD[v,t,w] = sqrt(sum((value - RollingMean[v,t,w])^2) / (w-1))
Sum over the last w past-filled CPU values; w = 3, 6 or 12.
```

**Two slopes** measure recent direction, for $w\in\{6,12\}$. Define
$z_j=\tilde y_{v,t-w+1+j}$ and $\bar j=(w-1)/2$:

$$
b_{v,t,w}=\frac{\sum_{j=0}^{w-1}(j-\bar j)z_j}
{\sum_{j=0}^{w-1}(j-\bar j)^2}.
$$

Plain-text equivalent:

```text
TrendSlope[v,t,w] = sum((j - mean_j) * CPU_at_position_j)
                     / sum((j - mean_j)^2)
j = 0,...,w-1 within the trailing window; mean_j = (w-1)/2; w = 6 or 12
```

Slope units are CPU percentage points per five-minute step.

**Two context values** are the VM's historical mean and the current population
mean. For observed times $O_{v,t}$ through the anchor:

$$
H_{v,t}=\frac1{|O_{v,t}|}\sum_{\tau\in O_{v,t}}y_{v,\tau},
\qquad A_t=\frac1{N_t}\sum_{v\in V_t}y_{v,t}.
$$

Plain-text equivalent:

```text
H[v,t] = sum(observed CPU values of VM v through t) / observed_value_count
A[t]   = sum(observed VM CPU values at t) / observed_VM_count_at_t
```

Unlike rolling feature inputs, $H_{v,t}$ does not count forward-filled values
as additional observations. The final **three inputs** are $C_v$.
Thus the feature count is:

$$
12+6+2+2+3=25.
$$

Plain-text equivalent:

```text
25 inputs = 12 lags + 6 rolling statistics + 2 slopes
            + 2 context values + 3 metadata categories
```

## 5. What LightGBM learns

At every eligible historical anchor, the training label is the next VM CPU
minus that VM's historical mean:

$$
r_{v,t+1}=y_{v,t+1}-H_{v,t}.
$$

Plain-text equivalent:

```text
TrainingResidual[v,t+1] = actual_avg_cpu[v,t+1] - historical_mean[v,t]
```

The shared LightGBM learns $F(X_{v,t})$ to reduce squared residual error:

$$
\mathcal L(F)=\sum_{(v,t)\in\mathcal E}
\left[r_{v,t+1}-F(X_{v,t})\right]^2.
$$

Plain-text equivalent:

```text
Loss = sum((TrainingResidual[v,t+1] - LightGBM(features[v,t]))^2)
Sum over eligible VM/anchor pairs with observed next-step labels.
```

$\mathcal E$ contains observed next-step labels that have arrived by the
fit cutoff. This is a sum over individual VM examples, not a loss fitted to
110 population averages. The forecast is:

$$
\hat y_{v,t+1}=\operatorname{clip}
\left(H_{v,t}+F(X_{v,t}),0,100\right).
$$

Plain-text equivalent:

```text
PredictedCPU[v,t+1] = clip(historical_mean[v,t]
                          + LightGBM(features[v,t]), 0, 100)
```

Fixed settings are 120 trees, learning rate 0.05, 31 leaves, minimum 100 child
samples and seed 42. No VM or eligible anchor is subsampled. No validation
label is used for early stopping, and no holdout label tunes these settings.

| Final training quantity | Value |
|---|---:|
| VM histories | 10,000 |
| Training timestamps per VM | 110 |
| Raw training readings | 1,100,000 |
| Initial lag-history observations per VM | 12 |
| Supervised next-step labels per VM | 98 |
| Total supervised examples | 980,000 |

**Interpretation:** the first 12 observations provide the initial history;
the following 98 observations become labels. All training readings contribute
as context and/or labels. Pooling increases the number of VM examples but does
not create additional independent time periods.

## 6. How 240,000 VM forecasts become 24 output rows

At cutoff $T$, forecast VMs observed at $T$. For each future step:

1. Build features from that VM's available history.
2. Predict its next CPU using the equation above.
3. Append the prediction, rather than the actual future CPU, to its history.
4. Recompute features and repeat through step 24.

Beyond $T$, historical means, rolling statistics and aggregate context are
updated using those appended predictions. This is **recursive forecasting**,
not 24 separately trained horizon models. Errors can propagate between steps.

The required CPU output is:

$$
\hat A_{T+h}=\frac1{|V_T|}\sum_{v\in V_T}\hat y_{v,T+h},
\qquad h=1,\ldots,24.
$$

Plain-text equivalent:

```text
PredictedAggregateCPU[T+h] = sum(predicted CPU of cutoff VMs at step h)
                            / number of VMs observed at cutoff T
h = 1,...,24
```

Future membership is held fixed at the cutoff population. The count
rule does not assign forecast arrivals/departures to particular VM IDs.
The individual CPU forecasts are saved in
[`artifacts/final_model/individual_predictions.csv.gz`](../artifacts/final_model/individual_predictions.csv.gz).
These generated artifacts are ignored by Git and recreated by the pipeline.

## 7. How observed VM count is forecast

CPU prediction and count prediction are separate. The count series is
$N_t=|V_t|$, derived from distinct valid-observation VM IDs at each grid slot.
The final pipeline applies observed-count persistence:

$$
\hat N_{T+h}=N_T,\qquad h=1,\ldots,24.
$$

Plain-text equivalent:

```text
PredictedCount[T+h] = number of distinct VMs with valid readings at cutoff T
h = 1,...,24
```

The supplied complete-trace population has 10,000 observed VMs at every slot,
so all 24 count forecasts are 10,000. The value comes from the observations;
it is not a constant embedded in the implementation. The count rule does not
infer unseen arrivals or departures. Zero error on a constant population does
not establish churn forecasting ability.

The historical VM CPU level $H_{v,t}$ in sections 4–5 remains part of the
LightGBM residual calculation and is recomputed within the recursive forecast.

## 8. Chronological validation, metrics and limitations

Each fold refits the fixed CPU model using only readings available at its
origin. On the supplied release, the first 110 observations provide training
and validation history; the last 24 supply future forecast timestamps.
Origins are 48, 54, 60, 66, 72, 78 and 84, each forecasting 24 steps. The
first fit sees rows 0–47 and forecasts rows 48–71; the last sees rows 0–83
and forecasts rows 84–107. Each next-step supervised label must already be
observed within that fold's history.

The full fit uses 1.10 million historical readings. Earlier folds use fewer
readings because their cutoffs occur earlier. The 168 origin/horizon decisions
cover 60 distinct timestamps; overlapping windows and cross-VM correlation
limit statistical independence.

For actual fleet CPU $A_i$ and prediction $\hat A_i$:

$$
\operatorname{MAE}=\frac1K\sum_{i=1}^{K}|A_i-\hat A_i|,
\qquad
\operatorname{RMSE}=\sqrt{\frac1K\sum_{i=1}^{K}(A_i-\hat A_i)^2}.
$$

Plain-text equivalent:

```text
MAE  = sum(abs(actual_CPU[i] - predicted_CPU[i])) / K
RMSE = sqrt(sum((actual_CPU[i] - predicted_CPU[i])^2) / K)
K = number of scored forecast cases
```

Signed bias and MAE skill are:

$$
\operatorname{Bias}=\frac1K\sum_{i=1}^{K}(\hat A_i-A_i),
\qquad
\operatorname{Skill}=1-\frac{\operatorname{MAE}_{model}}
{\operatorname{MAE}_{persistence}}.
$$

Plain-text equivalent:

```text
Bias  = mean(predicted_CPU - actual_CPU)
Skill = 1 - model_MAE / persistence_MAE
```

Negative bias indicates underprediction. Skill requires a nonzero benchmark
MAE. CPU errors are percentage points. The persistence benchmark repeats the
last observed fleet CPU and is scored on identical validation cases.

| CPU forecast / benchmark | CV MAE (pp) | CV RMSE (pp) | MAE skill |
|---|---:|---:|---:|
| Final recursive VM LightGBM | 0.19881 | 0.26889 | 25.12% |
| Persistence benchmark | 0.26550 | 0.31736 | 0% |

Feature gains show fitted tree usage. They do not establish causal importance
or the validation benefit of each metadata column. Inspect individual VM error
percentiles and workload-group errors alongside aggregate scores because
averaging can cancel individual errors.

High-load diagnostics use full-training Q80 = 9.55449% retrospectively. This is
not an online threshold available at every earlier origin. In those cases,
final-model MAE is 0.37269 pp, RMSE 0.41569 pp, bias −0.36405 pp, and the
underprediction fraction is 93.33%. Better pooled CV accuracy does not establish
better peak performance.

The final 24 slots contain 240,000 VM readings excluded from this model's
fitting. The holdout had already been inspected before this model's experiment,
so no untouched final test is claimed. No new holdout score is used to tune or
select this model. Selection on the same limited CV evidence can be optimistic;
independent confirmation needs unseen operating periods. The short relative
history, complete-trace population and missing physical-host/application
constraints also limit deployment conclusions.

## 9. Reproduce and inspect

Run from the repository root after installing the project:

```powershell
python -m cwfc.predict --data challenge_data --config configs/final_model.yaml --output predictions/prediction.csv
python -m cwfc.validation
```

The prediction command fits the final CPU model and applies count persistence.
It writes a 24-row CSV with exactly `timestamp`, `active_vm_count` and
`aggregate_cpu_utilization`. Supporting outputs include individual CPU
predictions, feature gains, fitting diagnostics and a provenance manifest under
`artifacts/final_model/`. The manifest records configuration, source and input
hashes, supervised sample counts and the final export hash.

The validation command refits the same model at the chronological origins and
regenerates backtest cases, scores and horizon/fold/regime diagnostics. It does
not score the holdout or choose a replacement method. Saved notebook evidence
can be inspected without repeating expensive fitting.

Implementation references:

- [model.py](../src/cwfc/model.py): VM histories, the 25 features, residual fitting and recursion.
- [predict.py](../src/cwfc/predict.py): cutoff selection, count persistence, CSV validation and provenance.
- [validation.py](../src/cwfc/validation.py): chronological folds and benchmark diagnostics.
- [quality.py](../src/cwfc/quality.py) and [targets.py](../src/cwfc/targets.py): preparation and target construction.
- [report.md](report.md): measured results and operational interpretation.

## 10. Supplementary comparison and operational methods

The final CPU model above remains recursive LightGBM. Notebook 03 also retains executable aggregate baselines, statistical/ridge/segment training, a count comparator, direct CPU ablations and activity classifiers. These methods support comparison and bonus operational findings in the report.

Direct variants predict `clip(H[i,t] + F(features[i,t], h), 0, 100)` without recursively appending predictions. Historical anchors advance six slots, all eligible VMs contribute, labels must mature, and total weight is equal per horizon. Variants add peaks/burst summaries, expanding max/P95, then remove category. Numerical feature filling uses earlier observations; labels are unfilled.

Busy labels are `1[avg_cpu >= 1%]`; summing predicted probabilities estimates expected busy count. The screen requires all preceding 12 averages <1% and maxima <5%. A separate classifier predicts any crossing during the complete next 24 slots. Event labels require the full future window. Both classifiers exclude category. Platt calibration fits `sigmoid(a*logit(raw_p)+b)` only on matured earlier-origin out-of-fold outcomes; fewer than 30 cases or one observed class serves raw probabilities. No candidate-event labels conservatively serves risk 1; one-class histories use a smoothed rate.

Candidate policies compare fixed risk budgets, matched-size lowest-average/lowest-peak rules and hysteresis (entry 0.1, exit 0.2, continued screen required). Candidate precision means no threshold crossing across the complete future window. Candidate VM-hours overlap and are not realized savings.

Headroom compares fixed buffers with empirical P90 residual margins in six-step horizon bands, using only matured earlier-origin errors. Fewer than 12 residuals uses 0.5 pp warmup. Cost is `5*max(actual-capacity,0) + max(capacity-actual,0)` and capacity is capped at 100%. Coefficients are assumptions; P90 is not the linear 5:1 cost-optimal quantile, which is 5/6. The separate exact-horizon category scenarios expose unsupported P90 ranks with seven folds. No empirical or conformal coverage guarantee is claimed under serial dependence and repeated selection.

The optional Holt count comparator uses fixed alpha=0.3, beta=0.1, phi=0.98 and a zero initial trend. It ties persistence on this constant population; the final count rule remains observed-count persistence. All supplementary findings retain their evidence scope and do not create additional final prediction CSVs.

Supporting code: [baselines.py](../src/cwfc/baselines.py), [features.py](../src/cwfc/features.py), [models.py](../src/cwfc/models.py), [direct.py](../src/cwfc/direct.py), [operational.py](../src/cwfc/operational.py), [operational_evaluation.py](../src/cwfc/operational_evaluation.py), and [analysis.py](../src/cwfc/analysis.py).

Category quantile calculations are executable in [capacity.py](../src/cwfc/capacity.py) with [uncertainty.py](../src/cwfc/uncertainty.py), using [capacity.yaml](../configs/capacity.yaml). Run `python -m cwfc.capacity` to rebuild their CV/deployment bounds and readiness figure from retained category histories.
