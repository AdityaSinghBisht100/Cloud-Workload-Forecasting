"""Training-only workload summaries and figures for exploratory workload analysis.

CPU contributions are sums of utilization percentages, not core-weighted CPU
consumption. No lifetime CPU statistic or deletion field is used here.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from statsmodels.tsa.stattools import acf, pacf

from cwfc.quality import observation_grid, prepare_readings

SEGMENTS = ["vm_category", "virtual_core_bucket", "memory_gb_bucket"]
LABELS = ["vm_id", *SEGMENTS, "subscription_id", "deployment_id"]


def load_windows(series, threshold, above=True):
    """Contiguous above/below-threshold samples; strict comparisons, no filling."""
    mask = (series.aggregate_cpu_utilization.gt(threshold) if above
            else series.aggregate_cpu_utilization.lt(threshold)).to_numpy()
    rows = []
    start = None
    for position in range(len(mask) + 1):
        selected = position < len(mask) and mask[position]
        if selected and start is None:
            start = position
        if not selected and start is not None:
            window = series.iloc[start:position]
            rows.append({"window_id": len(rows) + 1, "start_index": start,
                         "end_index": position - 1, "start_seconds": int(window.timestamp.iloc[0]),
                         "end_seconds": int(window.timestamp.iloc[-1]), "samples": len(window),
                         "mean_cpu": float(window.aggregate_cpu_utilization.mean()),
                         "max_cpu": float(window.aggregate_cpu_utilization.max()),
                         "min_cpu": float(window.aggregate_cpu_utilization.min())})
            start = None
    return pd.DataFrame(rows, columns=["window_id", "start_index", "end_index", "start_seconds",
                                      "end_seconds", "samples", "mean_cpu", "max_cpu", "min_cpu"])


def change_scan(series, window=6):
    """Adjacent complete windows: right mean minus left mean, in CPU pp.

    This is a retrospective scan, not an online detector or a significance test.
    A candidate boundary is the first timestamp of the right window.
    """
    if window < 2:
        raise ValueError("Change-scan window must be at least two samples")
    y = series.aggregate_cpu_utilization.to_numpy(dtype=float)
    rows = []
    for boundary in range(window, len(y) - window + 1):
        left, right = y[boundary - window:boundary], y[boundary:boundary + window]
        if np.isfinite(np.r_[left, right]).all():
            rows.append((boundary, int(series.timestamp.iloc[boundary]), right.mean() - left.mean()))
    scan = pd.DataFrame(rows, columns=["boundary_index", "timestamp", "change_pp"])
    scan["absolute_change_pp"] = scan.change_pp.abs()
    if len(scan):
        # Local maxima keep nearby boundaries from being reported as separate events.
        values = scan.absolute_change_pp.to_numpy()
        candidates, _ = find_peaks(np.r_[-np.inf, values, -np.inf], distance=window)
        selected = candidates - 1
        selected = selected[(selected >= 0) & (selected < len(scan))]
        top = scan.iloc[selected].nlargest(6, "absolute_change_pp").sort_values("boundary_index")
    else:
        top = scan.copy()
    return scan, top.reset_index(drop=True)


def correlation_table(series, clean, max_lag=12):
    """Standard mean-centered ACF at each lag, plus aggregate Yule-Walker PACF.

    VM ACF uses lagged centered products / total centered sum of squares, the
    same unadjusted definition as the aggregate. Constant VMs are undefined and
    excluded from the median. Missing slots keep their original time positions.
    PACF is omitted if the aggregate is incomplete, constant, or too short.
    """
    wide = clean.pivot(index="timestamp", columns="vm_id", values="avg_cpu").reindex(series.timestamp)
    values = wide.to_numpy(dtype=float)
    centers = wide.mean(axis=0).to_numpy(dtype=float)
    centered = values - centers
    denominator = np.nansum(centered ** 2, axis=0)
    usable = (denominator > 1e-12) & (np.isfinite(values).sum(axis=0) >= 3)
    y = series.aggregate_cpu_utilization.to_numpy(dtype=float)
    n_lags = min(max_lag, max(0, len(series) // 2 - 1))
    aggregate_acf = np.full(n_lags + 1, np.nan)
    aggregate_pacf = aggregate_acf.copy()
    if np.isfinite(y).sum() >= 3 and np.nanstd(y) > 1e-12:
        aggregate_acf = acf(y, nlags=n_lags, fft=False, missing="conservative")
        if np.isfinite(y).all() and n_lags >= 1:
            aggregate_pacf = pacf(y, nlags=n_lags, method="ywm")
    rows = []
    for lag in range(1, n_lags + 1):
        pairs = np.isfinite(centered[lag:]) & np.isfinite(centered[:-lag])
        valid = usable & (pairs.sum(axis=0) >= 2)
        estimates = np.nansum(centered[lag:] * centered[:-lag], axis=0)[valid] / denominator[valid]
        rows.append((lag, lag * 5, aggregate_acf[lag], aggregate_pacf[lag],
                     float(np.median(estimates)) if len(estimates) else np.nan, int(valid.sum())))
    table = pd.DataFrame(rows, columns=["lag", "minutes", "aggregate_acf", "aggregate_pacf", "median_vm_acf", "eligible_vms"])
    return table, int((~usable).sum())


def segment_summary(panel, columns, high_timestamps):
    per_time = panel.groupby(["timestamp", *columns], observed=True).agg(
        mean_cpu=("avg_cpu", "mean"), cpu_sum=("avg_cpu", "sum"), observed_vms=("vm_id", "nunique")
    ).reset_index()
    summary = per_time.groupby(columns, observed=True).agg(
        mean_cpu=("mean_cpu", "mean"), volatility_pp=("mean_cpu", "std"),
        timestamps_observed=("timestamp", "nunique"),
    )
    summary["vm_count"] = panel.groupby(columns, observed=True).vm_id.nunique()
    peak = per_time.loc[per_time.timestamp.isin(high_timestamps)].groupby(columns, observed=True).cpu_sum.sum()
    summary["high_load_cpu_share_pct"] = peak.reindex(summary.index, fill_value=0) / peak.sum() * 100 if peak.sum() > 0 else np.nan
    return summary.reset_index().sort_values("mean_cpu", ascending=False).reset_index(drop=True), per_time


def concentration(metadata):
    rows, curves = [], {}
    for column in ("subscription_id", "deployment_id"):
        counts = metadata.groupby(column, observed=True).vm_id.nunique().sort_values(ascending=False)
        shares = counts / counts.sum()
        rows.append({"group": column, "distinct_groups": len(counts),
                     "largest_group_vms": int(counts.iloc[0]), "largest_group_share_pct": shares.iloc[0] * 100,
                     "top_10_group_share_pct": shares.head(10).sum() * 100,
                     "hhi": float((shares ** 2).sum()), "effective_group_count": float(1 / (shares ** 2).sum())})
        curves[column] = pd.DataFrame({"rank": np.arange(1, len(counts) + 1), "cumulative_vm_share_pct": shares.cumsum().to_numpy() * 100})
    return pd.DataFrame(rows), curves


def peak_drivers(panel, series, windows, baseline_samples=6):
    """Additive contribution to aggregate change vs preceding 6 sample slots.

    For each timestamp divide a VM's avg_cpu by the observed VM count. Average
    those contributions over valid timestamps in each window, then subtract the
    baseline. Contributions sum to the aggregate mean change, even with changing
    observed populations. An absent VM contributes zero to the decomposition;
    this is not imputation of its reading. Offsetting declines can make positive
    contribution shares exceed 100%; shares are not clipped.
    """
    counts = panel.groupby("timestamp", observed=True).vm_id.transform("nunique")
    weighted = panel.assign(contribution=panel.avg_cpu / counts)
    labels = panel[["vm_id", "vm_category"]].drop_duplicates().set_index("vm_id")
    output, vm_rows, category_rows = [], [], []
    for row in windows.to_dict("records"):
        start, end = row["start_index"], row["end_index"]
        before = series.iloc[max(0, start - baseline_samples):start]
        event = series.iloc[start:end + 1]
        baseline_times = before.loc[before.aggregate_cpu_utilization.notna(), "timestamp"]
        event_times = event.loc[event.aggregate_cpu_utilization.notna(), "timestamp"]
        row.update(baseline_samples=len(baseline_times), baseline_cpu=np.nan, increase_pp=np.nan,
                   top_vm=None, top_category=None, top_20_vm_share_of_increase_pct=np.nan)
        if len(baseline_times) and len(event_times):
            base = weighted.loc[weighted.timestamp.isin(baseline_times)].groupby("vm_id", observed=True).contribution.sum() / len(baseline_times)
            event_values = weighted.loc[weighted.timestamp.isin(event_times)].groupby("vm_id", observed=True).contribution.sum() / len(event_times)
            difference = event_values.subtract(base, fill_value=0).rename("contribution_pp").to_frame().join(labels).reset_index()
            difference["window_id"] = row["window_id"]
            difference = difference.sort_values("contribution_pp", ascending=False)
            category = difference.groupby("vm_category", observed=True).contribution_pp.sum().sort_values(ascending=False).reset_index()
            category["window_id"] = row["window_id"]
            row["baseline_cpu"] = float(before.aggregate_cpu_utilization.mean())
            row["increase_pp"] = row["mean_cpu"] - row["baseline_cpu"]
            row["top_vm"] = difference.vm_id.iloc[0]
            row["top_category"] = category.vm_category.iloc[0]
            if row["increase_pp"] > 1e-12:
                row["top_20_vm_share_of_increase_pct"] = difference.head(20).contribution_pp.sum() / row["increase_pp"] * 100
            if not np.isclose(difference.contribution_pp.sum(), row["increase_pp"], atol=1e-10):
                raise AssertionError("VM contributions must reconcile to aggregate change")
            vm_rows.append(difference)
            category_rows.append(category)
        output.append(row)
    vm_columns = ["vm_id", "contribution_pp", "vm_category", "window_id"]
    category_columns = ["vm_category", "contribution_pp", "window_id"]
    return (pd.DataFrame(output), pd.concat(vm_rows, ignore_index=True) if vm_rows else pd.DataFrame(columns=vm_columns),
            pd.concat(category_rows, ignore_index=True) if category_rows else pd.DataFrame(columns=category_columns))


def analyze_workload(readings, metadata, train_targets):
    """Analyze only the supplied training timestamps, regardless of extra input rows.

    Assumptions fixed before inspection: idle mean <5%; steady mean >=5% and
    coefficient of variation <=0.5; hot any max_cpu >80%; rolling/scan window 6;
    high > training Q80 and low < training Q20. VM indicators may overlap.
    """
    series = train_targets[["timestamp", "aggregate_cpu_utilization", "active_vm_count"]].copy().reset_index(drop=True)
    grid = observation_grid(series)
    if not np.array_equal(grid.to_numpy(), series.timestamp.to_numpy()):
        raise ValueError("Training targets require a sorted, complete five-minute grid")
    clean = prepare_readings(readings.loc[readings.timestamp.isin(grid)])
    if clean.empty:
        raise ValueError("Workload analysis requires valid training readings")
    missing = set(LABELS) - set(metadata)
    if missing:
        raise ValueError(f"EDA metadata missing columns: {sorted(missing)}")
    labels = metadata[LABELS].copy()
    if labels.isna().any().any() or labels.vm_id.duplicated().any():
        raise ValueError("EDA metadata labels must be nonmissing with unique VM IDs")
    labels = labels.loc[labels.vm_id.isin(clean.vm_id.unique())]
    if set(labels.vm_id) != set(clean.vm_id):
        raise ValueError("EDA metadata must cover every observed training VM")
    panel = clean.merge(labels, on="vm_id", validate="many_to_one")
    # Verify that supplied target values agree with the readings being analyzed.
    grouped = clean.groupby("timestamp", observed=True)
    expected_cpu = grouped.avg_cpu.mean().reindex(grid)
    expected_count = grouped.vm_id.nunique().reindex(grid, fill_value=0)
    if not (np.allclose(series.aggregate_cpu_utilization, expected_cpu, equal_nan=True)
            and np.array_equal(series.active_vm_count, expected_count)):
        raise ValueError("Training targets do not match supplied readings")
    vm = clean.groupby("vm_id", observed=True).agg(mean_cpu=("avg_cpu", "mean"),
        std_cpu=("avg_cpu", "std"), peak_cpu=("max_cpu", "max"), observations=("timestamp", "nunique"),
        cpu_sum=("avg_cpu", "sum"))
    vm["cv"] = vm.std_cpu / vm.mean_cpu.replace(0, np.nan)
    vm["idle"] = vm.mean_cpu.lt(5)
    vm["steady"] = vm.mean_cpu.ge(5) & vm.cv.le(.5)
    vm["hot"] = vm.peak_cpu.gt(80)
    vm = vm.reset_index()
    indicators = pd.DataFrame([
        ("Idle", "Historical CPU average avg_cpu < 5%", int(vm.idle.sum()), vm.idle.mean() * 100),
        ("Steady", "Historical CPU average >= 5% and std/mean <= 0.5", int(vm.steady.sum()), vm.steady.mean() * 100),
        ("Hot", "At least one training max_cpu > 80%", int(vm.hot.sum()), vm.hot.mean() * 100),
    ], columns=["indicator", "rule", "vm_count", "share_pct"])
    high = float(series.aggregate_cpu_utilization.quantile(.8))
    low = float(series.aggregate_cpu_utilization.quantile(.2))
    high_times = series.loc[series.aggregate_cpu_utilization.gt(high), "timestamp"]
    detailed, segment_series = segment_summary(panel, SEGMENTS, high_times)
    category, category_series = segment_summary(panel, ["vm_category"], high_times)
    group_summary, curves = concentration(labels)
    scan, changes = change_scan(series)
    correlations, excluded_vms = correlation_table(series, clean)
    peaks, drivers, category_drivers = peak_drivers(panel, series, load_windows(series, high))
    lows = load_windows(series, low, above=False)
    temporal = series.copy()
    temporal["rolling_mean"] = series.aggregate_cpu_utilization.rolling(6, min_periods=3).mean()
    temporal["rolling_variance"] = series.aggregate_cpu_utilization.rolling(6, min_periods=3).var(ddof=1)
    eligible = peaks.loc[peaks.baseline_samples.gt(0)] if len(peaks) else peaks
    primary = int(eligible.sort_values("mean_cpu", ascending=False).window_id.iloc[0]) if len(eligible) else None
    top_n = max(1, int(np.ceil(len(vm) * .01)))
    metrics = {"training_samples": len(series), "vm_count": len(vm), "high_threshold": high,
               "low_threshold": low, "mean_cpu": float(series.aggregate_cpu_utilization.mean()),
               "aggregate_std_pp": float(series.aggregate_cpu_utilization.std(ddof=1)),
               "aggregate_cv": float(series.aggregate_cpu_utilization.std() / series.aggregate_cpu_utilization.mean()),
               "median_vm_cv": float(vm.cv.median()), "acf_excluded_vms": excluded_vms,
               "idle_and_hot_vms": int((vm.idle & vm.hot).sum()),
               "top_1pct_vm_count": top_n,
               "top_1pct_cpu_share_pct": float(vm.cpu_sum.nlargest(top_n).sum() / vm.cpu_sum.sum() * 100) if vm.cpu_sum.sum() else np.nan,
               "primary_peak_window_id": primary}
    return {"metrics": metrics, "series": series, "vm_statistics": vm, "indicators": indicators,
            "temporal": temporal, "correlations": correlations, "change_scan": scan, "change_candidates": changes,
            "segments": detailed, "segment_series": segment_series, "categories": category,
            "category_series": category_series, "concentration": group_summary, "concentration_curves": curves,
            "peaks": peaks, "lows": lows, "vm_drivers": drivers, "category_drivers": category_drivers}


def plot_distributions(analysis):
    vm = analysis["vm_statistics"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    for ax, column, title in zip(axes, ["mean_cpu", "peak_cpu"], ["Per-VM mean CPU", "Per-VM peak CPU"]):
        positive = vm.loc[vm[column].gt(0), column]
        if len(positive):
            lower = max(float(positive.min()), 1e-9)
            upper = max(float(positive.max()), lower * 1.01)
            ax.hist(positive, bins=np.geomspace(lower, upper, 36), color="#246b91", edgecolor="white")
            ax.set_xscale("log")
        ax.set(title=f"{title}\nZero-valued VMs shown separately: {vm[column].eq(0).sum():,}",
               xlabel="CPU utilization (%) â€” logarithmic axis", ylabel="VMs")
    return fig


def plot_temporal(analysis):
    frame, metrics = analysis["temporal"], analysis["metrics"]
    x, y = frame.timestamp / 3600, frame.aggregate_cpu_utilization
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True, constrained_layout=True)
    mean, sd = metrics["mean_cpu"], metrics["aggregate_std_pp"]
    axes[0].axhspan(mean - 2 * sd, mean + 2 * sd, alpha=.12, color="#246b91", label="Historical CPU average Â±2 sample SD (descriptive)")
    axes[0].plot(x, y, color="#246b91", label="Aggregate CPU")
    axes[0].axhline(metrics["high_threshold"], color="#bf6421", ls="--", label="Training Q80")
    axes[0].set(ylabel="CPU (%)", title="Training aggregate: level and high-load threshold")
    axes[0].legend(fontsize=8, loc="upper left")
    axes[1].plot(x, y, color=".75", label="Observed")
    axes[1].plot(x, frame.rolling_mean, color="#246b91", label="Trailing 6-sample mean")
    axes[1].set(ylabel="CPU (%)", title="Trailing mean and variance use current/past observations only")
    axes[1].legend(fontsize=8, loc="upper left")
    variance_axis = axes[1].twinx()
    variance_axis.plot(x, frame.rolling_variance, color="#bf6421", ls=":", label="Trailing sample variance")
    variance_axis.set_ylabel("Variance (ppÂ²)", color="#bf6421")
    variance_axis.legend(fontsize=8, loc="upper right")
    scan = analysis["change_scan"]
    axes[2].plot(scan.timestamp / 3600, scan.change_pp, color="#246b91")
    axes[2].axhline(0, color=".5", lw=.8)
    for row in analysis["change_candidates"].itertuples():
        axes[2].scatter(row.timestamp / 3600, row.change_pp, color="#bf6421")
        axes[2].annotate(str(row.boundary_index), (row.timestamp / 3600, row.change_pp), xytext=(3, 5), textcoords="offset points", fontsize=8)
    axes[2].set(ylabel="Right âˆ’ left mean (pp)", xlabel="Hours since relative trace start",
                title="Retrospective change scan: adjacent 6-sample windows; labels are training indices")
    return fig


def plot_correlations(analysis):
    table = analysis["correlations"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    axes[0].plot(table.minutes, table.aggregate_acf, marker="o", label="Aggregate ACF")
    axes[0].plot(table.minutes, table.median_vm_acf, marker="s", label="Median VM ACF (nonconstant VMs)")
    axes[0].set(title="Persistence at aggregate vs VM level", xlabel="Lag (minutes)", ylabel="Autocorrelation")
    axes[0].legend(fontsize=8)
    axes[1].bar(table.minutes, table.aggregate_pacf, width=3, color="#246b91")
    axes[1].set(title="Aggregate PACF (Yuleâ€“Walker)", xlabel="Lag (minutes)", ylabel="Partial autocorrelation")
    for ax in axes:
        ax.axhline(0, color=".5", lw=.8)
        ax.set_ylim(-1, 1)
    return fig


def plot_segments(analysis):
    table = analysis["segments"].nlargest(12, "high_load_cpu_share_pct").sort_values("high_load_cpu_share_pct")
    names = table[SEGMENTS].astype(str).agg(" / ".join, axis=1)
    fig, axes = plt.subplots(1, 3, figsize=(15, 6), sharey=True, constrained_layout=True)
    for ax, col, title, color in zip(axes,
            ["mean_cpu", "volatility_pp", "high_load_cpu_share_pct"],
            ["Mean CPU (%)", "Time-series SD (pp)", "Share of high-load CPU sum (%)"],
            ["#246b91", "#608936", "#bf6421"]):
        ax.barh(names, table[col], color=color)
        ax.set_xlabel(title)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Top 12 category / core / memory combinations by high-load contribution", fontsize=12)
    return fig


def plot_concentration(analysis):
    fig, ax = plt.subplots(figsize=(9, 4), constrained_layout=True)
    for label, curve in analysis["concentration_curves"].items():
        ax.plot(curve["rank"], curve.cumulative_vm_share_pct, label=label.replace("_id", ""))
    ax.set(xlabel="Groups ranked from largest to smallest", ylabel="Cumulative share of selected VMs (%)",
           title="Selected-population concentration by subscription and deployment", ylim=(0, 102))
    ax.legend()
    return fig


def plot_peak_drivers(analysis):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    window_id = analysis["metrics"]["primary_peak_window_id"]
    if window_id is None:
        for ax in axes:
            ax.text(.5, .5, "No high-load window with a preceding baseline", ha="center", transform=ax.transAxes)
        return fig
    categories = analysis["category_drivers"].query("window_id == @window_id").sort_values("contribution_pp")
    vms = analysis["vm_drivers"].query("window_id == @window_id").nlargest(10, "contribution_pp").sort_values("contribution_pp")
    axes[0].barh(categories.vm_category.astype(str), categories.contribution_pp, color="#246b91")
    axes[0].set_title("Category contributions (all categories)")
    labels = [f"VM {rank}" for rank in range(len(vms), 0, -1)]
    axes[1].barh(labels, vms.contribution_pp, color="#bf6421")
    axes[1].set_title("Ten largest VM contributions; full IDs in notebook table")
    for ax in axes:
        ax.axvline(0, color=".5", lw=.8)
        ax.set_xlabel("Contribution to aggregate mean change (pp)")
    fig.suptitle(f"High-load window {window_id}: change vs immediately preceding 6 sample slots")
    return fig


def save_figure(figure, directory, name):
    path = Path(directory) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, facecolor="white")
    return path


def plot_validation_windows(backtest):
    """Draw expanding histories and validation windows solely from backtest rows."""
    from matplotlib.patches import Patch

    from cwfc.validation import summarize_folds

    folds = summarize_folds(backtest)
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    for row in folds.itertuples():
        ax.barh(row.fold, row.origin, left=0, color="#246b91", height=.6)
        ax.barh(row.fold, row.validation_steps, left=row.origin, color="#bf6421", height=.6)
        ax.text(row.origin + .5, row.fold, str(row.origin), va="center", fontsize=8, color="white")
    ax.set(yticks=folds.fold, yticklabels=[f"Fold {f}" for f in folds.fold],
           xlabel="Row boundaries (origin = number of training observations)",
           title="Expanding-window validation: the next 24 steps at each origin")
    ax.invert_yaxis()
    ax.set_ylim(float(folds.fold.max()) + 1.8, .3)
    ax.legend(handles=[Patch(color="#246b91", label="Available history"),
                       Patch(color="#bf6421", label="Validation horizon")], loc="lower left")
    return fig


def plot_baseline_scores(backtest, target="aggregate_cpu_utilization"):
    """Compare pooled CV MAE/RMSE from the shared evidence table."""
    from cwfc.validation import score_backtest

    scores = score_backtest(backtest)
    scores = scores.loc[scores.target.eq(target)].sort_values("mae", ascending=False)
    labels = scores.model.map({"persistence": "Persistence", "drift": "Drift",
        "seasonal_naive_24": "Seasonal naive (24)",
        **{f"moving_average_{k}": f"Moving average ({k})" for k in (3, 6, 12, 24)}})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True, constrained_layout=True)
    unit = "CPU percentage points" if target == "aggregate_cpu_utilization" else "VM count"
    for ax, metric, color in zip(axes, ["mae", "rmse"], ["#246b91", "#bf6421"]):
        ax.barh(labels, scores[metric], color=color)
        ax.set_xlabel(f"{metric.upper()} ({unit})")
        ax.set_xlim(0, max(float(scores[metric].max()) * 1.23, 1e-6))
        for index, value in enumerate(scores[metric]):
            ax.text(value, index, f" {value:.4f}", va="center", fontsize=8)
    fig.suptitle("Baseline comparison â€” pooled training-only rolling-origin forecasts")
    return fig


def plot_model_comparison(backtest):
    """Main models and references, scored solely from the shared CV table."""
    from cwfc.validation import score_backtest

    scores = score_backtest(backtest)
    scores = scores.loc[scores.target.eq("aggregate_cpu_utilization")].sort_values("mae", ascending=False)
    labels = scores.model.map({"persistence": "Persistence", "drift": "Drift",
        "seasonal_naive_24": "Seasonal naive (24)",
        "ses": "Simple exponential smoothing", "ets_damped": "Damped-trend ETS",
        "theta": "Theta", "ridge_direct": "Direct ridge",
        "global_panel_lightgbm": "Global segment LightGBM", "cv_weighted_ensemble": "Past-error weighted ensemble",
        **{f"moving_average_{k}": f"Moving average ({k})" for k in (3, 6, 12, 24)}})
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True, constrained_layout=True)
    for ax, metric, color in zip(axes, ["mae", "rmse"], ["#246b91", "#bf6421"]):
        ax.barh(labels, scores[metric], color=color)
        ax.set_xlabel(f"{metric.upper()} (CPU percentage points)")
        ax.set_xlim(0, float(scores[metric].max()) * 1.22)
        for index, value in enumerate(scores[metric]):
            ax.text(value, index, f" {value:.4f}", va="center", fontsize=8)
    fig.suptitle("All CPU candidates â€” identical training-only rolling-origin cases")
    return fig
