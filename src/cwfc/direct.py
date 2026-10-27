"""Past-only panel features, direct LightGBM CPU and activity models."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.linear_model import LogisticRegression

from cwfc.quality import observation_grid, prepare_readings
from cwfc.model import FEATURES, STATIC, _features


PEAK_FEATURES = ["last_min_cpu", "last_max_cpu", "peak_minus_average", "max_minus_min",
                 "mean_max_6", "mean_max_12", "max_max_12", "low_fraction_12",
                 "hot_fraction_12", "steps_since_hot"]
SUMMARY_FEATURES = ["past_max_cpu", "past_p95_max_cpu"]
VARIANTS = ("direct_base", "direct_peaks", "direct_history", "direct_no_category")


@dataclass
class Panel:
    ids: np.ndarray
    timestamps: np.ndarray
    avg: np.ndarray
    minimum: np.ndarray
    maximum: np.ndarray
    filled: np.ndarray
    static: np.ndarray


def make_panel(readings, metadata, timestamps):
    """Bound raw observations before preparation; join only allowed static fields."""
    timestamps = np.asarray(timestamps, dtype="int64")
    grid = observation_grid(pd.DataFrame({"timestamp": timestamps}))
    if len(timestamps) < 13 or not np.array_equal(timestamps, grid):
        raise ValueError("Operational VM modeling needs at least 13 complete grid slots")
    clean = prepare_readings(readings.loc[readings.timestamp.isin(timestamps)])
    clean["vm_id"] = clean.vm_id.astype(str)
    ids = np.asarray(sorted(clean.vm_id.unique()))
    labels = metadata[["vm_id", *STATIC]].copy()
    labels["vm_id"] = labels.vm_id.astype(str)
    if labels.vm_id.duplicated().any():
        raise ValueError("VM metadata must have unique identifiers")
    labels = labels.set_index("vm_id").reindex(ids)
    if labels.isna().any().any():
        raise ValueError("Observed VMs require static labels")
    static = np.column_stack([pd.Categorical(labels[c].astype(str)).codes for c in STATIC])
    arrays = [clean.pivot(index="vm_id", columns="timestamp", values=c).reindex(
        index=ids, columns=timestamps).to_numpy(dtype=float) for c in ("avg_cpu", "min_cpu", "max_cpu")]
    return Panel(ids, timestamps, *arrays, pd.DataFrame(arrays[0]).ffill(axis=1).to_numpy(),
                 static.astype("float32"))


def anchor_features(panel, anchor, busy_threshold=1.):
    """Summaries use <= anchor even when the panel contains later observations."""
    if anchor < 11 or anchor >= panel.avg.shape[1]:
        raise ValueError("Feature anchor needs twelve preceding grid positions")
    base, mean = _features(panel.avg, panel.filled, anchor, panel.static)
    recent_avg = panel.avg[:, anchor-11:anchor+1]
    recent_max = panel.maximum[:, anchor-11:anchor+1]
    past_max = panel.maximum[:, :anchor+1]
    observed = np.isfinite(past_max)
    hot = observed & (past_max > 80)
    last_hot = np.where(hot, np.arange(anchor+1), -1).max(axis=1)
    with np.errstate(invalid="ignore"):
        extras = [panel.minimum[:, anchor], panel.maximum[:, anchor],
                  panel.maximum[:, anchor] - panel.avg[:, anchor],
                  panel.maximum[:, anchor] - panel.minimum[:, anchor],
                  np.nanmean(recent_max[:, -6:], axis=1), np.nanmean(recent_max, axis=1),
                  np.nanmax(recent_max, axis=1),
                  np.where(np.isfinite(recent_avg).all(axis=1), (recent_avg < busy_threshold).mean(axis=1), np.nan),
                  np.where(np.isfinite(recent_max).all(axis=1), (recent_max > 80).mean(axis=1), np.nan),
                  np.where(last_hot >= 0, anchor-last_hot, anchor+1),
                  np.nanmax(past_max, axis=1), np.nanquantile(past_max, .95, axis=1)]
    names = [*FEATURES, *PEAK_FEATURES, *SUMMARY_FEATURES]
    return np.column_stack([base, *extras]).astype("float32"), mean, names


def variant_columns(names, variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown direct variant: {variant}")
    allowed = set(FEATURES)
    if variant != "direct_base":
        allowed.update(PEAK_FEATURES)
    if variant in {"direct_history", "direct_no_category"}:
        allowed.update(SUMMARY_FEATURES)
    if variant == "direct_no_category":
        allowed.remove("vm_category")
    return [i for i, name in enumerate(names) if name in allowed or name == "horizon"]


def low_use_mask(panel, anchor, settings):
    window = settings["screen_window"]
    if anchor+1 < window:
        return np.zeros(len(panel.ids), dtype=bool)
    avg = panel.avg[:, anchor-window+1:anchor+1]
    maximum = panel.maximum[:, anchor-window+1:anchor+1]
    return ((avg < settings["busy_threshold"]).all(axis=1)
            & (maximum < settings["peak_threshold"]).all(axis=1))


def behavior_labels(panel, anchor, settings):
    """Mutually exclusive origin-time descriptions, not future-derived clusters."""
    history = panel.avg[:, :anchor+1]
    mean = np.nanmean(history, axis=1)
    sd = np.nanstd(history, axis=1)
    hot = np.nanmax(panel.maximum[:, :anchor+1], axis=1) > 80
    low = low_use_mask(panel, anchor, settings)
    result = np.full(len(panel.ids), "variable", dtype=object)
    result[(mean >= 5) & (sd / np.maximum(mean, 1e-8) <= .5)] = "steady"
    result[(mean < 5) & hot] = "low_average_bursty"
    result[low] = "sustained_low_recent"
    return result


def direct_examples(panel, horizon, stride, settings):
    """All VMs at fixed subsampled anchors; every label matures inside history."""
    if horizon < 1 or stride < 1:
        raise ValueError("Horizon and anchor stride must be positive")
    blocks, residuals, busy, horizons = [], [], [], []
    latest = -1
    for anchor in range(11, panel.avg.shape[1]-1, stride):
        features, baseline, names = anchor_features(panel, anchor, settings["busy_threshold"])
        for h in range(1, min(horizon, panel.avg.shape[1]-1-anchor)+1):
            target = panel.avg[:, anchor+h]
            valid = np.isfinite(target) & np.isfinite(baseline)
            if not valid.any():
                continue
            blocks.append(np.column_stack([features[valid], np.full(valid.sum(), h, dtype="float32")]))
            residuals.append((target[valid]-baseline[valid]).astype("float32"))
            busy.append((target[valid] >= settings["busy_threshold"]).astype("int8"))
            horizons.append(np.full(valid.sum(), h, dtype="int16"))
            latest = max(latest, anchor+h)
    if not blocks:
        raise ValueError("No matured direct training labels")
    x, y, z, hs = np.concatenate(blocks), np.concatenate(residuals), np.concatenate(busy), np.concatenate(horizons)
    counts = np.bincount(hs, minlength=horizon+1)[1:]
    if (counts == 0).any():
        raise ValueError("Every forecast horizon needs matured training labels")
    weights = len(hs)/(horizon*counts[hs-1])
    info = {"training_samples": len(y), "samples_by_horizon": counts.tolist(),
            "latest_label_row": latest, "anchor_stride": stride,
            "horizon_weighting": "equal total weight per horizon"}
    return x, y, z, weights, [*names, "horizon"], info


def fit_cpu(x, y, weights, names, variant, parameters):
    columns = variant_columns(names, variant)
    selected_names = [names[i] for i in columns]
    categorical = [i for i, name in enumerate(selected_names) if name in STATIC]
    model = LGBMRegressor(**parameters)
    model.fit(x[:, columns], y, sample_weight=weights,
              feature_name=selected_names, categorical_feature=categorical)
    return model, columns


def direct_predict(model, columns, panel, horizon, settings):
    features, baseline, _ = anchor_features(panel, panel.avg.shape[1]-1, settings["busy_threshold"])
    result = []
    for h in range(1, horizon+1):
        query = np.column_stack([features, np.full(len(features), h, dtype="float32")])[:, columns]
        # Native booster accepts the explicitly ordered feature matrix without pandas warnings.
        result.append(np.clip(baseline+model.booster_.predict(query), 0, 100))
    return np.column_stack(result)


@dataclass
class BinaryModel:
    model: object
    constant: float | None
    columns: list

    def predict(self, x):
        if self.constant is not None:
            return np.full(len(x), self.constant)
        return np.clip(self.model.booster_.predict(x[:, self.columns]), 0, 1)


def fit_binary(x, y, names, parameters, weights=None, columns=None):
    """A one-class history has a documented smoothed constant probability."""
    columns = list(range(x.shape[1])) if columns is None else columns
    if not len(y):
        raise ValueError("No matured binary labels")
    if np.unique(y).size < 2:
        return BinaryModel(None, float((y.sum()+1)/(len(y)+2)), columns)
    selected_names = [names[i] for i in columns]
    model = LGBMClassifier(**parameters)
    model.fit(x[:, columns], y, sample_weight=weights, feature_name=selected_names,
              categorical_feature=[i for i, name in enumerate(selected_names) if name in STATIC])
    return BinaryModel(model, None, columns)


def event_examples(panel, horizon, settings):
    """Train only historical screen candidates; complete event labels are required."""
    blocks, labels, ends = [], [], []
    for anchor in range(11, panel.avg.shape[1]-horizon):
        candidate = low_use_mask(panel, anchor, settings)
        avg = panel.avg[:, anchor+1:anchor+horizon+1]
        peak = panel.maximum[:, anchor+1:anchor+horizon+1]
        valid = candidate & np.isfinite(avg).all(axis=1) & np.isfinite(peak).all(axis=1)
        if valid.any():
            features, _, names = anchor_features(panel, anchor, settings["busy_threshold"])
            event = ((avg >= settings["busy_threshold"]) | (peak >= settings["peak_threshold"])).any(axis=1)
            blocks.append(features[valid])
            labels.append(event[valid].astype("int8"))
            ends.extend([anchor+horizon]*int(valid.sum()))
    if not blocks:
        return None, None, None, {"event_training_samples": 0, "latest_event_label_row": None}
    return np.concatenate(blocks), np.concatenate(labels), names, {
        "event_training_samples": sum(map(len, labels)), "latest_event_label_row": max(ends)}


@dataclass
class Calibration:
    model: object
    samples: int
    latest_timestamp: int | None

    def predict(self, probability):
        probability = np.asarray(probability)
        if self.model is None:
            return probability.copy()
        return self.model.predict_proba(_logit(probability))[:, 1]


def _logit(probability):
    p = np.clip(np.asarray(probability, dtype=float).reshape(-1), 1e-6, 1-1e-6)
    return np.log(p/(1-p)).reshape(-1, 1)


def fit_calibration(previous, cutoff_timestamp):
    """Only earlier-origin OOF predictions whose labels are now observable."""
    eligible = previous.loc[previous.label_timestamp <= cutoff_timestamp].dropna(subset=["raw_probability", "actual"])
    if len(eligible) < 30 or eligible.actual.nunique() < 2:
        return Calibration(None, len(eligible), None if eligible.empty else int(eligible.label_timestamp.max()))
    model = LogisticRegression(C=1., solver="lbfgs", max_iter=200, random_state=42)
    model.fit(_logit(eligible.raw_probability), eligible.actual.astype(int))
    return Calibration(model, len(eligible), int(eligible.label_timestamp.max()))
