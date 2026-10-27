"""Reproduce category quantile readiness using only training histories."""
from pathlib import Path
import numpy as np
import pandas as pd
import yaml
from cwfc.uncertainty import cpu_bounds

def load_policy(path):
    settings = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(settings, dict) or settings.get("schema_version") != 1:
        raise ValueError("Expected schema_version=1 capacity settings")
    if not settings.get("segments"):
        raise ValueError("Capacity settings require category scenarios")
    for scenario in settings["segments"].values():
        for key in ("under_cost", "over_cost", "margin"):
            value = scenario[key]
            if not np.isfinite(value) or value < 0 or (key != "margin" and value == 0):
                raise ValueError("Costs must be positive and margins nonnegative")
    levels = settings["cost_curve_levels"]
    if not levels or any(not np.isfinite(x) or not 0 < x < 1 for x in levels):
        raise ValueError("Cost-curve levels must be in (0,1)")
    extras = [*settings["fixed_headroom_pp"], *settings["peak_trigger"].values()]
    if not settings["fixed_headroom_pp"] or any(not np.isfinite(x) or x < 0 for x in extras):
        raise ValueError("Headroom and peak-trigger parameters must be nonnegative")
    return settings


def validate_evidence(evidence, histories):
    if (histories.empty or "timestamp" not in histories
            or not np.isfinite(histories.timestamp.to_numpy(float)).all()
            or (histories.timestamp % 1 != 0).any()
            or not histories.timestamp.diff().dropna().eq(300).all()):
        raise ValueError("Histories require a complete sorted 300-second grid")
    required = {"fold", "origin", "step", "model", "target", "timestamp", "y", "yhat"}
    if evidence.empty or not required.issubset(evidence):
        raise ValueError("Expected nonempty long-format CPU evidence")
    if evidence.duplicated(["target", "model", "origin", "step"]).any():
        raise ValueError("Duplicate calibration decisions")
    if not np.isfinite(evidence[["origin", "step", "timestamp", "y", "yhat"]].to_numpy(float)).all():
        raise ValueError("Calibration evidence must be finite")
    if not evidence[["y", "yhat"]].apply(lambda x: x.between(0, 100)).all().all():
        raise ValueError("CPU evidence must be in [0,100]")
    for (target, origin), part in evidence.groupby(["target", "origin"]):
        if (target not in histories or origin != int(origin) or origin < 1
                or part.step.lt(1).any() or (part.step % 1 != 0).any()):
            raise ValueError("Unknown target or invalid origin/step")
        indices = int(origin)+part.step.to_numpy(int)-1
        if indices.max() >= len(histories):
            raise ValueError("Calibration evidence extends outside training")
        actual = histories.iloc[indices]
        if (not np.array_equal(actual.timestamp, part.timestamp)
                or not np.allclose(actual[target], part.y, rtol=1e-10, atol=1e-10)):
            raise ValueError("Calibration timestamps or labels differ from training")


def capacity_decisions(cases, evidence, histories, settings):
    """Use exact-step earlier forecasts only after their labels have arrived.

    Histories end before the holdout. Each target has a fixed forecasting rule;
    evidence is further restricted to that same model. At origin o, observation
    o-1 is the last available row. Peak triggers inspect four historical values.
    """
    validate_evidence(evidence, histories)
    rows = []
    for case in cases.itertuples(index=False):
        origin, step = int(case.origin), int(case.step)
        if (origin != case.origin or step != case.step or step < 1
                or not 1 <= origin <= len(histories) or case.target not in histories):
            raise ValueError("Invalid capacity forecast origin/step/target")
        history = histories.iloc[:origin]
        cutoff = int(history.timestamp.iloc[-1])
        if case.timestamp != cutoff+300*step:
            raise ValueError("Capacity forecast timestamp differs from origin/step")
        scenario = settings["segments"].get(case.target)
        if scenario is None:
            raise ValueError(f"No explicit cost/margin scenario for {case.target}")
        values = history[case.target].to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError("Capacity histories must be finite; no target imputation")
        matured = evidence.loc[evidence.target.eq(case.target) & evidence.model.eq(case.model)
            & evidence.step.eq(step) & evidence.origin.lt(origin) & evidence.timestamp.le(cutoff)]
        residuals = (matured.y-matured.yhat).to_numpy()
        slope = float((values[-1]-values[-4])/3) if len(values) >= 4 else 0.
        trigger = slope > settings["peak_trigger"]["slope_pp_per_step"]
        tau = scenario["under_cost"]/(scenario["under_cost"]+scenario["over_cost"])
        policies = [(f"quantile_{level:g}", level) for level in settings["cost_curve_levels"]]
        policies += [("cost_ratio_default", tau), ("cost_ratio_peak_trigger", tau)]
        for name, level in policies:
            bounds = cpu_bounds(case.yhat, residuals, level)
            capacity = bounds["upper_cpu"]*(1+scenario["margin"])
            if name == "cost_ratio_peak_trigger" and trigger:
                capacity += settings["peak_trigger"]["extra_headroom_pp"]
            rows.append({**case._asdict(), **bounds, "policy": name, "level": level,
                "capacity_cpu": float(np.clip(capacity, 0, 100)), "slope_pp_per_step": slope,
                "peak_triggered": trigger, "cutoff_timestamp": cutoff,
                "latest_calibration_timestamp": matured.timestamp.max() if len(matured) else np.nan,
                "margin": scenario["margin"], "under_cost": scenario["under_cost"],
                "over_cost": scenario["over_cost"]})
        for headroom in settings["fixed_headroom_pp"]:
            rows.append({**case._asdict(), "policy": f"fixed_{headroom:g}pp", "level": np.nan,
                "capacity_cpu": float(np.clip(case.yhat+headroom, 0, 100)),
                "calibration_n": 0, "upper_supported": False, "interval_supported": False,
                "slope_pp_per_step": slope, "peak_triggered": trigger, "cutoff_timestamp": cutoff,
                "margin": 0., "under_cost": scenario["under_cost"], "over_cost": scenario["over_cost"]})
    return pd.DataFrame(rows)


def summarize_policy(decisions, dimensions=()):
    """Equal weight per origin/step, retaining correlated overlapping decisions."""
    data = decisions.copy()
    data["shortfall"] = (data.y-data.capacity_cpu).clip(lower=0)
    data["excess"] = (data.capacity_cpu-data.y).clip(lower=0)
    data["cost"] = data.shortfall*data.under_cost+data.excess*data.over_cost
    data["pinball"] = np.maximum(data.level*(data.y-data.upper_cpu),
                                  (data.level-1)*(data.y-data.upper_cpu))
    data["interval_hit"] = np.where(data.level.notna(),
        data.y.between(data.lower_cpu, data.interval_upper_cpu), np.nan)
    data["upper_hit"] = np.where(data.level.notna(), data.y.le(data.upper_cpu), np.nan)
    rows = []
    for keys, part in data.groupby(["target", "model", "policy", *dimensions], sort=True):
        rows.append({**dict(zip(["target", "model", "policy", *dimensions], keys)),
            "forecast_cases": len(part), "unique_timestamps": part.timestamp.nunique(),
            "under_provision_rate": part.shortfall.gt(0).mean(),
            "mean_shortfall_pp": part.shortfall.mean(), "mean_over_provision_pp": part.excess.mean(),
            "mean_cost_proxy": part.cost.mean(), "upper_coverage": part.upper_hit.mean(),
            "interval_coverage": part.interval_hit.mean(), "mean_pinball_pp": part.pinball.mean(),
            "supported_fraction": part.upper_supported.mean(),
            "mean_calibration_n": part.calibration_n.mean(),
            "peak_trigger_fraction": part.peak_triggered.mean()})
    return pd.DataFrame(rows)


def plot_policy(summary, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for target, part in summary.loc[summary.policy.str.startswith("quantile_")].groupby("target"):
        levels = part.policy.str.removeprefix("quantile_").astype(float)
        axes[0].plot(levels, part.mean_cost_proxy, marker="o", label=target)
        axes[1].plot(levels, part.supported_fraction, marker="o", label=target)
    axes[0].set(xlabel="One-sided quantile level", ylabel="Mean cost proxy (weighted CPU pp)",
                title="Cost curve including physical-bound fallback")
    axes[1].set(xlabel="One-sided quantile level", ylabel="Fraction of CV decisions with finite rank",
                ylim=(-.03, 1.03), title="Exact-step calibration readiness")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(root=None):
    root=Path(root) if root is not None else Path(__file__).resolve().parents[2]
    settings=load_policy(root/'configs/capacity.yaml')
    history=pd.read_csv(root/'artifacts/workload_analysis/category_series.csv').pivot(index='timestamp',columns='vm_category',values='mean_cpu').reset_index()
    from cwfc.validation import rolling_origin_index
    rows=[]
    for fold,split in enumerate(rolling_origin_index(len(history)),1):
        for target in history.columns.drop('timestamp'):
            for step in range(1,25):
                index=split.origin+step-1
                rows.append(dict(fold=fold,origin=split.origin,step=step,model='category_persistence',target=target,
                    timestamp=history.timestamp.iloc[index],y=history[target].iloc[index],yhat=history[target].iloc[split.origin-1]))
    evidence=pd.DataFrame(rows)
    decisions=capacity_decisions(evidence,evidence,history,settings)
    summary=summarize_policy(decisions)
    deployment=[]
    for target in history.columns.drop('timestamp'):
        for step in range(1,25):
            deployment.append(dict(origin=len(history),step=step,model='category_persistence',target=target,
                timestamp=history.timestamp.iloc[-1]+300*step,yhat=history[target].iloc[-1]))
    bounds=capacity_decisions(pd.DataFrame(deployment),evidence,history,settings)
    directory=root/'artifacts/operational'
    summary.to_csv(directory/'category_capacity_summary.csv',index=False)
    bounds.to_csv(directory/'category_capacity_deployment.csv',index=False)
    plot_policy(summary,root/'reports/figures/category_capacity.png')
    return summary,bounds

if __name__=='__main__':
    summary,_=run()
    print(summary.to_string(index=False))
