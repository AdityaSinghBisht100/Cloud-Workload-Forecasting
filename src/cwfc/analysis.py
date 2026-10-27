"""Rebuild final-model diagnostics and bonus figures from retained CV evidence."""
from pathlib import Path
import numpy as np
import pandas as pd

MODEL = "global_vm_lightgbm"


def summarize(cases, dimensions=()):
    """Score origin/step decisions; MASE uses each origin's history scale."""
    rows = []
    keys = [*dimensions, "model"]
    for values, group in cases.groupby(keys, sort=True):
        values = values if isinstance(values, tuple) else (values,)
        error = group.yhat.to_numpy() - group.y.to_numpy()
        scale = group.naive_scale.to_numpy() if "naive_scale" in group else np.full(len(group), np.nan)
        valid = np.isfinite(scale) & (scale > 0)
        rows.append({**dict(zip(keys, values)), "cases": len(group),
            "mae": np.abs(error).mean(), "rmse": np.sqrt(np.mean(error ** 2)),
            "smape_pct": np.mean(200*np.abs(error)/np.maximum(np.abs(group.y)+np.abs(group.yhat), 1e-8)),
            "mase": np.mean(np.abs(error[valid])/scale[valid]) if valid.any() else np.nan,
            "bias": error.mean(), "underforecast_fraction": (error < 0).mean()})
    return pd.DataFrame(rows)


def enrich(cases, training):
    """Attach origin-safe load labels and scales without future-history pooling."""
    result = cases.copy()
    for origin, group in result.groupby("origin"):
        history = training.iloc[:int(origin)]
        if not (group.timestamp > history.timestamp.iloc[-1]).all():
            raise ValueError("Forecast cases must follow their history cutoff")
        if not group.timestamp.isin(training.timestamp).all():
            raise ValueError("CV cases must remain inside training")
        actual = group.timestamp.map(training.set_index("timestamp").aggregate_cpu_utilization)
        if not np.allclose(actual, group.y, atol=1e-8, rtol=0):
            raise ValueError("CV actuals differ from retained training observations")
        values = history.aggregate_cpu_utilization
        result.loc[group.index, "high_threshold"] = values.quantile(.8)
        result.loc[group.index, "naive_scale"] = values.diff().abs().dropna().mean()
    result["load_period"] = np.where(result.y > result.high_threshold, "high_load", "normal")
    return result


def run(root=None):
    root = Path(root) if root is not None else Path(__file__).resolve().parents[2]
    final = root / "artifacts/final_model"
    bonus = root / "artifacts/operational"
    output = root / "artifacts/analysis"
    output.mkdir(parents=True, exist_ok=True)
    training = pd.read_csv(root / "artifacts/workload_analysis/series.csv")
    cases = enrich(pd.read_csv(final / "cpu_backtest.csv"), training)
    tables = {"cases": cases, "overall": summarize(cases)}
    for name, keys in [("by_horizon", ["step"]), ("by_fold", ["fold", "origin"]),
                       ("by_load", ["load_period"])]:
        tables[name] = summarize(cases, keys)
    smooth = training.aggregate_cpu_utilization.rolling(3, min_periods=3).mean()
    change = smooth.diff(3)
    last = int(training.index[training.timestamp.eq(cases.timestamp.max())][0])
    windows = {"regime_onset": int(change.idxmax()), "mean_reversion": int(change.idxmin()),
               "last_cv_window": last}
    failures = []
    for event, index in windows.items():
        stamps = training.timestamp.iloc[max(0, index-2):index+1]
        scored = summarize(cases.loc[cases.timestamp.isin(stamps)])
        scored["event"] = event
        scored["start_index"] = max(0, index-2)
        scored["end_index"] = index
        failures.append(scored)
    tables["failure_windows"] = pd.concat(failures, ignore_index=True)
    segments = pd.read_csv(final / "segment_cases.csv")
    tables["segments"] = summarize(segments, ["category", "core_bucket", "memory_bucket"])
    policies = pd.read_csv(bonus / "candidate_policy.csv")
    summary = policies.groupby("policy", as_index=False).agg(
        candidate_decisions=("candidate_decisions", "sum"),
        future_activity_events=("future_activity_events", "sum"),
        stable_candidate_decisions=("stable_candidate_decisions", "sum"),
        candidate_vm_hours=("candidate_vm_hours", "sum"), switches=("switches_since_previous_origin", "sum"))
    summary["candidate_precision"] = summary.stable_candidate_decisions / summary.candidate_decisions.replace(0, np.nan)
    tables["candidate_policy_summary"] = summary
    headroom = pd.read_csv(bonus / "headroom_backtest.csv")
    tables["headroom_summary"] = headroom.assign(under=headroom.underforecast.gt(0)).groupby("policy", as_index=False).agg(
        cases=("under", "size"), underforecast_fraction=("under", "mean"),
        mean_shortfall_pp=("underforecast", "mean"), mean_excess_headroom_pp=("excess_headroom_pp", "mean"),
        mean_cost_proxy=("cost_proxy", "mean"))
    for name, frame in tables.items():
        frame.to_csv(output / f"{name}.csv", index=False)
    plot(tables, bonus, root / "reports/figures")
    return tables


def plot(tables, bonus, directory):
    import matplotlib.pyplot as plt
    directory.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for model, group in tables["by_horizon"].groupby("model"):
        axes[0, 0].plot(group.step, group.mae, label=model)
        axes[0, 1].plot(group.step, group.bias, label=model)
    for model, group in tables["by_fold"].groupby("model"):
        axes[1, 0].plot(group.origin, group.mae, marker="o", label=model)
    failure = tables["failure_windows"]
    for offset, model in enumerate(failure.model.unique()):
        group = failure.loc[failure.model.eq(model)]
        axes[1, 1].bar(np.arange(len(group))+offset*.35, group.bias, width=.35, label=model)
    axes[1, 1].set_xticks(np.arange(3)+.175, ["Onset", "Reversion", "Last CV"])
    for ax, title, xlabel, ylabel in zip(axes.flat,
            ["Horizon error", "Signed horizon bias", "Fold error", "Transition-window bias"],
            ["Forecast step", "Forecast step", "History rows", "Window"],
            ["MAE (pp)", "Bias (pp)", "MAE (pp)", "Bias (pp)"]):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(directory / "final_error_analysis.png", dpi=150); plt.close(fig)
    reliability = pd.read_csv(bonus / "probability_reliability.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, target in zip(axes, reliability.target.unique()):
        subset = reliability.loc[reliability.target.eq(target)].copy()
        subset["predicted_sum"] = subset.mean_probability * subset.cases
        subset["observed_sum"] = subset.observed_event_fraction * subset.cases
        grouped = subset.groupby(["stream", "probability_bin"], as_index=False)[["cases", "predicted_sum", "observed_sum"]].sum()
        for stream, group in grouped.groupby("stream"):
            ax.plot(group.predicted_sum/group.cases, group.observed_sum/group.cases, marker="o", label=stream)
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(title=target, xlabel="Mean predicted probability", ylabel="Observed event fraction")
        ax.legend()
    fig.tight_layout(); fig.savefig(directory / "probability_calibration.png", dpi=150); plt.close(fig)
    activity = pd.read_csv(bonus / "activity_forecast.csv")
    history = pd.read_csv(bonus / "activity_history.csv")
    candidates = pd.read_csv(bonus / "low_use_review_candidates.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(history.timestamp/300, history.cpu_busy_count_ge_1pct, label="Observed CPU-busy count")
    axes[0].plot(activity.timestamp/300, activity.expected_cpu_busy_count, label="Expected busy count")
    axes[0].plot(activity.timestamp/300, activity.predicted_cpu_threshold_count, label="Thresholded CPU forecast")
    axes[0].set(xlabel="Relative observation index", ylabel="VM count"); axes[0].legend(fontsize=7)
    axes[1].hist(candidates.future_activity_risk, bins=np.linspace(0,1,21))
    axes[1].axvline(.1, color="red", label="Review-risk budget")
    axes[1].set(xlabel="Activity-event probability", ylabel="Screened VMs"); axes[1].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(directory / "operational_activity.png", dpi=150); plt.close(fig)


if __name__ == "__main__":
    result = run()
    print(result["overall"].to_string(index=False))
