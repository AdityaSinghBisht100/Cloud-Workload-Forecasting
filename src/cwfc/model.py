"""Final recursive VM LightGBM; all features are origin-safe."""
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from cwfc.quality import observation_grid, prepare_readings

MODEL = "global_vm_lightgbm"
STATIC = ["vm_category", "virtual_core_bucket", "memory_gb_bucket"]
PARAMS = dict(objective="regression", n_estimators=120, learning_rate=0.05,
              num_leaves=31, min_child_samples=100, random_state=42,
              n_jobs=2, deterministic=True, force_col_wise=True, verbosity=-1)
FEATURES = ([f"lag_{k}" for k in range(1, 13)]
            + [name for w in (3, 6, 12) for name in (f"mean_{w}", f"std_{w}")]
            + ["slope_6", "slope_12", "vm_history_mean", "aggregate_cpu_context"] + STATIC)


def vm_history(readings, metadata, timestamps):
    """Filter before cleaning/joining; never inspect future CPU or lifetime fields."""
    timestamps = pd.Index(timestamps, name="timestamp")
    if (not timestamps.is_unique or not timestamps.is_monotonic_increasing
            or len(timestamps) < 13
            or not np.array_equal(timestamps, observation_grid(pd.DataFrame({"timestamp": timestamps})))):
        raise ValueError("VM training requires at least 13 sorted five-minute slots")
    raw = readings.loc[readings.timestamp.isin(timestamps)]
    clean = prepare_readings(raw)
    ids = sorted(clean.vm_id.astype(str).unique())
    labels = metadata[["vm_id", *STATIC]].copy()
    labels["vm_id"] = labels.vm_id.astype(str)
    if labels.vm_id.duplicated().any():
        raise ValueError("VM metadata IDs must be unique")
    labels = labels.set_index("vm_id").reindex(ids)
    if labels.isna().any().any():
        raise ValueError("Observed training VMs require category/core/memory labels")
    static = np.column_stack([pd.Categorical(labels[c].astype(str)).codes for c in STATIC]).astype("float32")
    clean["vm_id"] = clean.vm_id.astype(str)
    values = clean.pivot(index="vm_id", columns="timestamp", values="avg_cpu").reindex(
        index=ids, columns=timestamps).to_numpy(dtype="float64")
    info = {"raw_training_readings": len(raw), "valid_training_readings": len(clean),
            "training_timestamps": len(timestamps), "vm_series": len(ids),
            "first_training_timestamp": int(timestamps[0]),
            "last_training_timestamp": int(timestamps[-1]),
            "missing_vm_timestamp_cells": int(np.isnan(values).sum())}
    return values, static, ids, info


def _features(values, filled, anchor, static):
    """One feature row per VM, using values at or before the inclusive anchor."""
    block = [filled[:, anchor - k + 1] for k in range(1, 13)]
    for window in (3, 6, 12):
        recent = filled[:, anchor - window + 1:anchor + 1]
        block.extend([recent.mean(axis=1), recent.std(axis=1, ddof=1)])
    for window in (6, 12):
        coordinate = np.arange(window) - (window - 1) / 2
        block.append(filled[:, anchor - window + 1:anchor + 1] @ coordinate / (coordinate @ coordinate))
    observed = values[:, :anchor + 1]
    counts = np.isfinite(observed).sum(axis=1)
    baseline = np.divide(np.nansum(observed, axis=1), counts,
                         out=np.full(len(values), np.nan), where=counts > 0)
    current = values[:, anchor]
    aggregate = np.nanmean(current) if np.isfinite(current).any() else np.nan
    block.extend([baseline, np.full(len(values), aggregate)])
    return np.column_stack([*block, static]).astype("float32"), baseline


def vm_forecast(readings, metadata, timestamps, horizon=24, params=None):
    """Learn one-step VM residuals from every eligible anchor, then recurse.

    Target = next observed CPU minus that VM's past-only historical mean.
    Squared error targets the conditional mean needed for aggregate demand.
    Predict each VM, add its baseline, clip to [0,100], append predictions to
    history, and repeat. No actual future CPU enters recursive features.
    """
    if not isinstance(horizon, int) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    values, static, ids, info = vm_history(readings, metadata, timestamps)
    filled = pd.DataFrame(values).ffill(axis=1).to_numpy()
    x_blocks, y_blocks = [], []
    latest_label_row = -1
    for anchor in range(11, values.shape[1] - 1):
        x, baseline = _features(values, filled, anchor, static)
        label = values[:, anchor + 1]
        valid = np.isfinite(label) & np.isfinite(baseline)
        if valid.any():
            x_blocks.append(x[valid])
            y_blocks.append((label[valid] - baseline[valid]).astype("float32"))
            latest_label_row = anchor + 1
    if not x_blocks:
        raise ValueError("No observed one-step VM training labels")
    x, y = np.concatenate(x_blocks), np.concatenate(y_blocks)
    model = LGBMRegressor(**(PARAMS if params is None else params))
    model.fit(x, y, feature_name=FEATURES,
              categorical_feature=[FEATURES.index(c) for c in STATIC])
    training_samples = len(y)
    del x_blocks, y_blocks, x, y
    # Forecast only VMs observed at the cutoff; unknown future membership is not inferred.
    active = np.isfinite(values[:, -1])
    if not active.any():
        raise ValueError("No VMs observed at the forecast cutoff")
    values, filled, static = values[active], filled[active], static[active]
    active_ids = np.asarray(ids)[active]
    predictions = []
    for _ in range(horizon):
        query, baseline = _features(values, filled, values.shape[1] - 1, static)
        next_cpu = np.clip(baseline + model.predict(pd.DataFrame(query, columns=FEATURES)), 0, 100)
        if not np.isfinite(next_cpu).all():
            raise ValueError("VM model returned nonfinite predictions")
        predictions.append(next_cpu)
        values = np.column_stack([values, next_cpu])
        filled = np.column_stack([filled, next_cpu])
    matrix = np.column_stack(predictions)
    aggregate = matrix.mean(axis=0)
    detail = pd.DataFrame({"vm_id": np.repeat(active_ids, horizon),
                           "step": np.tile(np.arange(1, horizon + 1), len(active_ids)),
                           "yhat": matrix.ravel()})
    importance = pd.DataFrame({"feature": FEATURES,
                               "gain": model.booster_.feature_importance(importance_type="gain")})
    diagnostic = {**info, "training_samples": training_samples,
                  "latest_label_row": latest_label_row, "features": len(FEATURES),
                  "forecast_vm_count": len(active_ids),
                  "holdout_labels_used": False}
    return aggregate, detail, diagnostic, importance.sort_values("gain", ascending=False)


