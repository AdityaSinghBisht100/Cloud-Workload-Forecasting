"""Planned aggregate models, pooled segment learning and CV-only selection."""

import hashlib
import json
from pathlib import Path
import subprocess
import warnings

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.forecasting.theta import ThetaModel
from statsmodels.tsa.holtwinters import ExponentialSmoothing, SimpleExpSmoothing

from cwfc.features import MIN_HISTORY, TARGET, feature_frame
from cwfc.quality import observation_grid, prepare_readings

SEED = 42
RIDGE_ALPHA = 10.0
PANEL_STRIDE = 3
PANEL_COLUMNS = ["vm_category", "virtual_core_bucket", "memory_gb_bucket"]
LIGHTGBM_PARAMS = dict(objective="regression_l1", n_estimators=120, learning_rate=.05,
                      num_leaves=7, min_child_samples=20, random_state=SEED, n_jobs=1,
                      deterministic=True, force_col_wise=True, verbosity=-1)
MODEL_NAMES = ("ses", "ets_damped", "theta", "ridge_direct", "global_panel_lightgbm")


def build_model_inputs(readings, metadata, training_series):
    """Training timestamps only; preserve CPU labels, augment observation context.

    Segments are category/core/memory combinations. Only these metadata fields
    and vm_id are joined. Arrival/departure counts mean observed-set changes,
    not actual lifecycle events. The first sample has unknown prior activity.
    """
    series = training_series.copy().reset_index(drop=True)
    grid = observation_grid(series)
    if not np.array_equal(series.timestamp, grid):
        raise ValueError("Model inputs require a sorted complete five-minute grid")
    clean = prepare_readings(readings.loc[readings.timestamp.isin(grid)])
    labels = metadata[["vm_id", *PANEL_COLUMNS]].copy()
    if labels.isna().any().any() or labels.vm_id.duplicated().any():
        raise ValueError("Model metadata requires unique nonmissing VM IDs and labels")
    if set(clean.vm_id) - set(labels.vm_id):
        raise ValueError("Model metadata does not cover observed training VMs")
    merged = clean.merge(labels, on="vm_id", validate="many_to_one")
    # JSON tuples avoid ambiguous category strings containing separators.
    keys = labels[PANEL_COLUMNS].astype(str).apply(lambda row: json.dumps(row.tolist()), axis=1)
    id_map = dict(zip(labels.vm_id, keys))
    merged["segment_id"] = merged.vm_id.astype(object).map(id_map)
    panel = merged.groupby(["timestamp", "segment_id", *PANEL_COLUMNS], observed=True).agg(
        mean_cpu=("avg_cpu", "mean"), observed_vm_count=("vm_id", "nunique"),
        mean_max=("max_cpu", "mean"), mean_min=("min_cpu", "mean")
    ).reset_index()
    panel["burstiness"] = panel.mean_max - panel.mean_min
    actual = clean.groupby("timestamp", observed=True).agg(cpu=("avg_cpu", "mean"), count=("vm_id", "nunique")).reindex(grid)
    if not (np.allclose(series[TARGET], actual.cpu, equal_nan=True)
            and np.array_equal(series.active_vm_count, actual["count"].fillna(0))):
        raise ValueError("Training targets and observed readings do not match")
    groups = {int(t): set(frame.vm_id) for t, frame in clean.groupby("timestamp", observed=True)}
    previous = None
    arrivals, departures = [], []
    for timestamp in series.timestamp:
        current = groups.get(int(timestamp), set())
        arrivals.append(np.nan if previous is None else len(current - previous))
        departures.append(np.nan if previous is None else len(previous - current))
        previous = current
    series["observed_arrivals"], series["observed_departures"] = arrivals, departures
    category_count = merged.groupby(["timestamp", "vm_category"], observed=True).vm_id.nunique().unstack("vm_category").reindex(grid).fillna(0)
    total = actual["count"].replace(0, np.nan)
    for category in sorted(category_count.columns.astype(str)):
        series[f"segment_vm_share::{category}"] = (category_count[category] / total).to_numpy()
    return series, panel.sort_values(["segment_id", "timestamp"]).reset_index(drop=True)


def statistical_forecast(history, horizon, method):
    values = history[TARGET].to_numpy(dtype=float)
    if len(values) < 12 or not np.isfinite(values).all():
        raise ValueError("Statistical models require at least twelve finite target values; targets are not imputed")
    if np.ptp(values) < 1e-12:
        return np.repeat(values[-1], horizon)
    with warnings.catch_warnings():
        # Treat failed optimization as an explicit failure, not a hidden fallback.
        warnings.simplefilter("error", ConvergenceWarning)
        if method == "ses":
            fit = SimpleExpSmoothing(values, initialization_method="estimated").fit(optimized=True)
        elif method == "ets_damped":
            fit = ExponentialSmoothing(values, trend="add", damped_trend=True,
                                       seasonal=None, initialization_method="estimated").fit(optimized=True)
        elif method == "theta":
            fit = ThetaModel(values, deseasonalize=False).fit()
        else:
            raise ValueError(f"Unknown statistical model: {method}")
    return np.asarray(fit.forecast(horizon), dtype=float)


def ridge_forecast(history, horizon, alpha=RIDGE_ALPHA):
    """One standardized ridge per horizon, with only labels matured by cutoff."""
    features = feature_frame(history, len(history) - 1)
    query = features.iloc[[-1]]
    if not np.isfinite(query.to_numpy()).all():
        raise ValueError("Ridge forecast features lack sufficient past observations")
    output, samples = [], []
    for step in range(1, horizon + 1):
        labels = history[TARGET].shift(-step)
        valid = features.notna().all(axis=1) & labels.notna()
        if valid.sum() < 2:
            raise ValueError(f"Ridge step {step}: fewer than two matured training labels")
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
        model.fit(features.loc[valid], labels.loc[valid])
        output.append(float(model.predict(query)[0]))
        samples.append({"step": step, "training_samples": int(valid.sum()),
                        "latest_label_row": int(np.flatnonzero(valid.to_numpy())[-1] + step)})
    return np.asarray(output), pd.DataFrame(samples)


def _segment_history(panel, timestamps, aggregate):
    """Materialize each observed segment on the past grid; counts zero when absent."""
    histories = {}
    for segment, group in panel.groupby("segment_id", observed=True):
        group = group.set_index("timestamp").reindex(timestamps)
        frame = pd.DataFrame({"timestamp": timestamps.to_numpy(),
                              TARGET: group.mean_cpu.to_numpy(),
                              "active_vm_count": group.observed_vm_count.fillna(0).to_numpy(),
                              "burstiness": group.burstiness.to_numpy(),
                              "aggregate_cpu_context": aggregate[TARGET].to_numpy()})
        static = panel.loc[panel.segment_id.eq(segment), PANEL_COLUMNS].iloc[0].astype(str).to_dict()
        histories[segment] = (frame, static)
    return histories


def panel_forecast(history, panel_history, horizon, params=None, stride=PANEL_STRIDE):
    """Direct multi-horizon L1 LightGBM shared across observed segment series.

    Examples use every third historical anchor, a horizon-step feature and only
    labels <= the current cutoff. Predicted segment means are aggregated with
    the last observed segment VM counts (fixed over the horizon). Segment CPU
    targets stay missing when unobserved; only feature inputs are forward-filled.
    """
    if stride < 1:
        raise ValueError("Panel stride must be positive")
    cutoff = history.timestamp.iloc[-1]
    if panel_history.timestamp.gt(cutoff).any():
        raise ValueError("Panel history contains observations after the forecast origin")
    histories = _segment_history(panel_history, history.timestamp, history)
    training_x, training_y, query_rows, weights, ids = [], [], [], [], []
    latest_label_row = -1
    for segment, (frame, static) in histories.items():
        features = feature_frame(frame, len(frame) - 1)
        features["history_mean"] = frame[TARGET].expanding(min_periods=1).mean()
        for window in (3, 6, 12):
            values = frame[TARGET].ffill().rolling(window, min_periods=window)
            features[f"rolling_min_{window}"] = values.min()
            features[f"rolling_max_{window}"] = values.max()
        for name, value in {"segment_id": segment, **static}.items():
            features[name] = value
        anchors = np.arange(MIN_HISTORY - 1, len(frame) - 1, stride)
        for step in range(1, horizon + 1):
            eligible = anchors[anchors + step < len(frame)]
            x = features.iloc[eligible].copy()
            y = frame[TARGET].iloc[eligible + step].to_numpy(dtype=float)
            finite = x.select_dtypes(include="number").notna().all(axis=1).to_numpy() & np.isfinite(y)
            if finite.any():
                x = x.loc[finite].copy()
                x["horizon_step"] = step
                training_x.append(x)
                training_y.append(y[finite])
                latest_label_row = max(latest_label_row, int((eligible + step)[finite].max()))
        weight = float(frame.active_vm_count.iloc[-1])
        if weight > 0:
            query = features.iloc[[-1]].copy()
            if not query.select_dtypes(include="number").notna().all().all():
                raise ValueError("Panel forecast features lack historical coverage")
            query = pd.concat([query.assign(horizon_step=step) for step in range(1, horizon + 1)], ignore_index=True)
            query_rows.append(query)
            weights.append(weight)
            ids.append(segment)
    if not training_x or not query_rows:
        raise ValueError("No valid panel training examples or currently observed segment weights")
    x = pd.concat(training_x, ignore_index=True)
    y = np.concatenate(training_y)
    query = pd.concat(query_rows, ignore_index=True)
    for name in ["segment_id", *PANEL_COLUMNS]:
        dtype = pd.CategoricalDtype(categories=sorted(x[name].unique()))
        if not set(query[name]).issubset(dtype.categories):
            raise ValueError("A queried segment has no matured historical panel labels")
        x[name], query[name] = x[name].astype(dtype), query[name].astype(dtype)
    model = LGBMRegressor(**(LIGHTGBM_PARAMS if params is None else params))
    model.fit(x, y)
    segment_values = np.clip(model.predict(query).reshape(len(ids), horizon), 0, 100)
    weights = np.asarray(weights)
    aggregate = np.average(segment_values, axis=0, weights=weights)
    detail = pd.DataFrame({"segment_id": np.repeat(ids, horizon),
        "step": np.tile(np.arange(1, horizon + 1), len(ids)),
        "yhat": segment_values.ravel(), "observed_vm_weight": np.repeat(weights, horizon)})
    diagnostics = {"training_samples": len(y), "segments": len(histories),
                   "latest_label_row": latest_label_row, "features": x.shape[1]}
    return aggregate, detail, diagnostics


def count_churn_forecast(history, horizon, window=6):
    """Persistence for unchanged counts; otherwise extrapolate recent net change.

    This is a fallback for changing observed populations, not a lifecycle model.
    """
    values = history.active_vm_count.to_numpy(dtype=float)
    if not len(values) or not np.isfinite(values).all() or window < 1:
        raise ValueError("Count fallback requires finite history and a positive window")
    slope = float(np.diff(values[-(window + 1):]).mean()) if len(values) > 1 else 0.
    return np.maximum(values[-1] + np.arange(1, horizon + 1) * slope, 0)


def inverse_error_weights(evidence, models, top_k=3):
    """Use supplied matured CV rows only; equal weights before any evidence."""
    if evidence.empty:
        return {model: 1 / len(models) for model in sorted(models)}
    error = evidence.assign(error=(evidence.y - evidence.yhat).abs()).groupby("model").error.mean()
    error = error.reindex(models).dropna().sort_index().sort_values(kind="stable").head(top_k)
    if error.empty:
        raise ValueError("No matched model errors for weighting")
    if error.eq(0).any():
        winners = error.loc[error.eq(0)]
        return {model: 1 / len(winners) for model in winners.index}
    weights = (1 / error) / (1 / error).sum()
    return weights.to_dict()


