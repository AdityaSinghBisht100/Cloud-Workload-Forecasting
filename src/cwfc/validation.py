from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import pandas as pd

from cwfc.baselines import baseline_forecasts
from cwfc.metrics import mae, rmse, smape
from cwfc.quality import observation_grid


@dataclass(frozen=True)
class TimeSplit:
    train_end: int | pd.Timestamp
    validation_start: int | pd.Timestamp
    validation_end: int | pd.Timestamp


@dataclass(frozen=True)
class IndexSplit:
    """Half-open [train_start, train_end) and [validation_start, validation_end)."""

    train_start: int
    train_end: int
    validation_start: int
    validation_end: int

    @property
    def origin(self):
        """Number of training rows; also the first validation row index."""
        return self.train_end


def rolling_origin_index(n=110, horizon=24, min_train=48, step=6):
    """Expanding histories; seven default origins at 48,54,...,84.

    The next origin is 90, whose validation end (114) would exceed n=110.
    Validation windows may overlap; each origin/step remains a separate forecast.
    """
    for name, value in (("n", n), ("horizon", horizon), ("min_train", min_train), ("step", step)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if n < min_train + horizon:
        raise ValueError(f"Need at least {min_train + horizon} rows for one fold; received {n}")
    return [IndexSplit(0, origin, origin, origin + horizon)
            for origin in range(min_train, n - horizon + 1, step)]


def rolling_origin_splits(
    frame: pd.DataFrame,
    timestamp_col: str = "timestamp",
    horizon: int = 24,
    min_train_size: int = 48,
    step: int | None = None,
) -> list[TimeSplit]:
    """Compatibility wrapper preserving relative seconds or datetime timestamps.

    Timestamp endpoints are inclusive for the existing TimeSplit API. For row
    slicing use rolling_origin_index and its half-open bounds instead.
    """
    ts = frame[timestamp_col]
    if ts.isna().any() or not ts.is_monotonic_increasing or not ts.is_unique:
        raise ValueError("timestamps must be nonmissing, sorted and unique")
    if len(ts) > 2 and ts.diff().dropna().nunique() != 1:
        raise ValueError("timestamps must be evenly sampled")
    step = horizon if step is None else step
    return [TimeSplit(ts.iloc[s.train_end - 1], ts.iloc[s.validation_start], ts.iloc[s.validation_end - 1])
            for s in rolling_origin_index(len(frame), horizon, min_train_size, step)]


def backtest_baselines(training_series, targets=("active_vm_count", "aggregate_cpu_utilization"),
                       horizon=24, min_train=48, step=6):
    """Evaluate fixed candidates only on supplied public-training observations.

    Call after split_train_holdout. Forecast functions receive an origin-limited
    history and a template containing timestamps only, never validation labels.
    No target imputation, fitting on holdout, random shuffling or tuning here.
    The same domain clipping applies to every candidate before scoring.
    """
    frame = training_series.reset_index(drop=True)
    if not targets or len(set(targets)) != len(targets):
        raise ValueError("targets must contain unique target names")
    required = {"timestamp", *targets}
    if not required.issubset(frame):
        raise ValueError(f"Missing backtest columns: {sorted(required - set(frame))}")
    grid = observation_grid(frame)
    if not np.array_equal(grid.to_numpy(), frame.timestamp.to_numpy()):
        raise ValueError("Backtest input must have sorted, unique, complete five-minute timestamps")
    if not np.isfinite(frame[list(targets)].to_numpy(dtype=float)).all():
        raise ValueError("Backtest targets contain missing/nonfinite values; targets are not imputed")
    splits = rolling_origin_index(len(frame), horizon, min_train, step)
    if min_train < 24:
        raise ValueError("The shared candidate set requires min_train >= 24 (MA24 and seasonal-naive 24)")
    rows = []
    for fold, split in enumerate(splits, start=1):
        history = frame.iloc[split.train_start:split.train_end]
        validation = frame.iloc[split.validation_start:split.validation_end]
        template = validation[["timestamp"]].copy()
        for target in targets:
            # Restrict inputs to the single target and its past timestamps.
            forecasts = baseline_forecasts(history[["timestamp", target]].copy(), template, target)
            for model, forecast in forecasts.items():
                predictions = forecast.to_numpy(dtype=float)
                if target == "aggregate_cpu_utilization":
                    predictions = np.clip(predictions, 0, 100)
                elif target == "active_vm_count":
                    predictions = np.maximum(predictions, 0)
                if len(predictions) != horizon or not np.isfinite(predictions).all():
                    raise ValueError(f"{model}: expected {horizon} finite predictions")
                for offset, (timestamp, actual, predicted) in enumerate(
                        zip(validation.timestamp, validation[target], predictions), start=1):
                    rows.append((fold, split.origin, offset, model, target, int(timestamp), float(actual), float(predicted)))
    return pd.DataFrame(rows, columns=["fold", "origin", "step", "model", "target", "timestamp", "y", "yhat"])


def score_backtest(backtest):
    """Pooled origin/step metrics from the single long-format evidence table.

    Overlapping windows represent distinct forecast decisions, not independent
    timestamps. Skill is undefined (NaN) when persistence MAE is zero, including
    ties at zero error; no artificial percentage improvement is reported.
    """
    if backtest.empty:
        raise ValueError("Cannot score an empty backtest")
    rows = []
    for (target, model), group in backtest.groupby(["target", "model"], sort=True):
        rows.append({"target": target, "model": model, "forecast_cases": len(group),
                     "folds": group.fold.nunique(), "mae": mae(group.y, group.yhat),
                     "rmse": rmse(group.y, group.yhat), "smape_pct": smape(group.y, group.yhat)})
    scores = pd.DataFrame(rows)
    reference = scores.loc[scores.model.eq("persistence")].set_index("target").mae
    if set(scores.target) - set(reference.index):
        raise ValueError("Persistence rows are required to compute skill")
    scores["persistence_mae"] = scores.target.map(reference)
    scores["skill_vs_persistence"] = np.nan
    nonzero = scores.persistence_mae.gt(0)
    scores.loc[nonzero, "skill_vs_persistence"] = 1 - scores.loc[nonzero, "mae"] / scores.loc[nonzero, "persistence_mae"]
    return scores.sort_values(["target", "mae", "rmse", "model"]).reset_index(drop=True)


def select_moving_average(scores):
    """Choose k by pooled CV MAE per target; break exact ties toward smaller k."""
    candidates = scores.loc[scores.model.str.fullmatch(r"moving_average_(3|6|12|24)")].copy()
    if candidates.empty:
        raise ValueError("No moving-average CV scores available")
    candidates["window"] = candidates.model.str.rsplit("_", n=1).str[-1].astype(int)
    return (candidates.sort_values(["target", "mae", "window"])
            .groupby("target", sort=True).head(1).reset_index(drop=True))


def summarize_folds(backtest):
    """Recover fold boundaries from the long table, without another evidence source."""
    table = backtest.groupby(["fold", "origin"], as_index=False).agg(
        validation_start_seconds=("timestamp", "min"), validation_end_seconds=("timestamp", "max"),
        validation_steps=("step", "nunique"))
    table["train_start_seconds"] = table.validation_start_seconds - table.origin * 300
    table["train_end_seconds"] = table.validation_start_seconds - 300
    return table[["fold", "origin", "train_start_seconds", "train_end_seconds",
                  "validation_start_seconds", "validation_end_seconds", "validation_steps"]]


def backtest_models(training_series, segment_panel, horizon=24, min_train=48, step=6):
    """Run planned candidates with the same folds/columns as baseline evidence.

    Ensemble weights at an origin use earlier-origin predictions only when their
    observed labels have already arrived. First-fold weights are uniform over
    all available CPU candidates, fixed without seeing validation errors.
    """
    from cwfc.models import (MODEL_NAMES, TARGET, count_churn_forecast,
                             inverse_error_weights, panel_forecast, ridge_forecast,
                             statistical_forecast)

    frame = training_series.reset_index(drop=True)
    baseline = backtest_baselines(frame, horizon=horizon, min_train=min_train, step=step)
    rows, diagnostics, panel_rows, weight_rows = [], [], [], []
    components = sorted(set(baseline.loc[baseline.target.eq(TARGET), "model"]) | set(MODEL_NAMES))
    for fold, split in enumerate(rolling_origin_index(len(frame), horizon, min_train, step), start=1):
        history = frame.iloc[:split.origin].copy().reset_index(drop=True)
        validation = frame.iloc[split.validation_start:split.validation_end]
        cutoff = int(history.timestamp.iloc[-1])
        # Models never receive validation labels, future context or future panel rows.
        forecasts = {name: statistical_forecast(history, horizon, name)
                     for name in ("ses", "ets_damped", "theta")}
        forecasts["ridge_direct"], ridge_info = ridge_forecast(history, horizon)
        ridge_info = ridge_info.assign(fold=fold, origin=split.origin, model="ridge_direct")
        diagnostics.append(ridge_info)
        panel_history = segment_panel.loc[segment_panel.timestamp.le(cutoff)].copy()
        forecasts["global_panel_lightgbm"], detail, info = panel_forecast(history, panel_history, horizon)
        detail = detail.assign(fold=fold, origin=split.origin)
        panel_rows.append(detail)
        diagnostics.append(pd.DataFrame([{**info, "fold": fold, "origin": split.origin,
                                          "model": "global_panel_lightgbm", "step": 0}]))
        for model, values in forecasts.items():
            values = np.clip(np.asarray(values, dtype=float), 0, 100)
            if values.shape != (horizon,) or not np.isfinite(values).all():
                raise ValueError(f"{model}: expected {horizon} finite predictions")
            for offset, (timestamp, actual, predicted) in enumerate(zip(validation.timestamp, validation[TARGET], values), start=1):
                rows.append((fold, split.origin, offset, model, TARGET, int(timestamp), float(actual), float(predicted)))
        combined = pd.concat([baseline, pd.DataFrame(rows, columns=baseline.columns)], ignore_index=True)
        matured = combined.loc[combined.target.eq(TARGET) & combined.model.isin(components)
                               & combined.origin.lt(split.origin) & combined.timestamp.le(cutoff)]
        weights = inverse_error_weights(matured, components)
        current = combined.loc[combined.target.eq(TARGET) & combined.origin.eq(split.origin)
                               & combined.model.isin(weights)]
        weighted = current.assign(weight=current.model.map(weights))
        ensemble = (weighted.yhat * weighted.weight).groupby(weighted.step).sum().reindex(range(1, horizon + 1))
        for model, weight in weights.items():
            source = matured.loc[matured.model.eq(model)]
            weight_rows.append({"fold": fold, "origin": split.origin, "model": model, "weight": weight,
                                "available_error_cases": len(source),
                                "latest_error_timestamp": int(source.timestamp.max()) if len(source) else np.nan,
                                "cutoff_timestamp": cutoff})
        for offset, (timestamp, actual, predicted) in enumerate(zip(validation.timestamp, validation[TARGET], ensemble), start=1):
            rows.append((fold, split.origin, offset, "cv_weighted_ensemble", TARGET, int(timestamp), float(actual), float(predicted)))
        count_values = count_churn_forecast(history, horizon)
        for offset, (timestamp, actual, predicted) in enumerate(zip(validation.timestamp, validation.active_vm_count, count_values), start=1):
            rows.append((fold, split.origin, offset, "count_churn", "active_vm_count", int(timestamp), float(actual), float(predicted)))
    evidence = pd.concat([baseline, pd.DataFrame(rows, columns=baseline.columns)], ignore_index=True)
    return {"backtest": evidence, "fit_diagnostics": pd.concat(diagnostics, ignore_index=True),
            "panel_predictions": pd.concat(panel_rows, ignore_index=True),
            "ensemble_weights": pd.DataFrame(weight_rows)}


def backtest_final(readings, metadata, training_series, horizon=24, min_train=48, step=6, params=None):
    """Evaluate the fixed model and persistence on identical chronological cases."""
    from cwfc.model import MODEL, vm_forecast
    rows, diagnostics = [], []
    for fold, split in enumerate(rolling_origin_index(len(training_series),horizon,min_train,step),1):
        history = training_series.iloc[:split.origin]
        future = training_series.iloc[split.origin:split.validation_end]
        cpu, _, info, _ = vm_forecast(readings,metadata,history.timestamp,horizon,params)
        diagnostics.append({"fold":fold,"origin":split.origin,**info})
        for model,values in [(MODEL,cpu),("vm_persistence",np.repeat(history.aggregate_cpu_utilization.iloc[-1],horizon))]:
            for h,(timestamp,y,yhat) in enumerate(zip(future.timestamp,future.aggregate_cpu_utilization,values),1):
                rows.append((fold,split.origin,h,model,int(timestamp),float(y),float(yhat)))
    return pd.DataFrame(rows,columns=['fold','origin','step','model','timestamp','y','yhat']),pd.DataFrame(diagnostics)


def regression_scores(frame):
    rows = []
    for model, group in frame.groupby("model", sort=True):
        e = group.yhat.to_numpy()-group.y.to_numpy()
        rows.append({"model": model, "forecast_cases": len(e), "mae": np.abs(e).mean(),
                     "rmse": np.sqrt(np.mean(e**2)), "bias": e.mean(),
                     "smape_pct": np.mean(200*np.abs(e)/np.maximum(np.abs(group.y)+np.abs(group.yhat), 1e-8)),
                     "underforecast_fraction": (e < 0).mean()})
    return pd.DataFrame(rows).sort_values(["mae", "rmse", "model"]).reset_index(drop=True)


def error_breakdowns(frame, high_threshold):
    data = frame.assign(error=frame.yhat-frame.y)
    data["regime"] = np.where(data.y > high_threshold, "high_training_Q80", "other")
    rows = []
    for dimension in ["step", "fold", "regime"]:
        for (value, model), group in data.groupby([dimension, "model"], sort=True):
            e = group.error.to_numpy()
            rows.append({"dimension": dimension, "value": str(value), "model": model,
                         "cases": len(e), "mae": np.abs(e).mean(), "rmse": np.sqrt(np.mean(e**2)),
                         "bias": e.mean(), "underforecast_fraction": (e < 0).mean()})
    return pd.DataFrame(rows)


def main():
    from pathlib import Path
    from cwfc.io import load_readings,load_metadata
    from cwfc.predict import load_config
    from cwfc.targets import build_targets,split_train_holdout
    config=load_config()
    readings,metadata=load_readings(),load_metadata()
    train,_=split_train_holdout(build_targets(readings,metadata),**config['split'])
    evidence,diagnostics=backtest_final(readings,metadata,train,params=config['parameters'],**config['validation'])
    output=Path('artifacts/final_model'); output.mkdir(parents=True,exist_ok=True)
    evidence.to_csv(output/'cpu_backtest.csv',index=False)
    scores=regression_scores(evidence)
    scores['skill_vs_vm_persistence']=1-scores.mae/float(scores.loc[scores.model.eq('vm_persistence'),'mae'].iloc[0])
    scores.to_csv(output/'cpu_scores.csv',index=False)
    error_breakdowns(evidence,train.aggregate_cpu_utilization.quantile(.8)).to_csv(output/'error_breakdowns.csv',index=False)
    diagnostics.to_csv(output/'fit_diagnostics.csv',index=False)
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
