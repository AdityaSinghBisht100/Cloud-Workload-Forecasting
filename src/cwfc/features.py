"""Causal features: origin is the inclusive positional index last observed.

The CV splitter's origin is a row count, so pass cv_origin - 1 here. Lag 1 is
the last observed value used to forecast the next sample. Targets are never
filled; forward-fill below applies only to historical feature inputs.
"""

from numbers import Integral

import numpy as np
import pandas as pd

TARGET = "aggregate_cpu_utilization"
MIN_HISTORY = 12
CONTEXT_PREFIXES = ("segment_cpu_utilization::", "segment_vm_share::")


def feature_frame(series, origin, target=TARGET):
    """Return feature rows through origin, with all operations on a bounded copy.

    Early rows without twelve observations retain NaN. Context columns not yet
    observed at the origin are omitted; no future category values determine
    features. Missing values can only be carried forward from an earlier row.
    """
    if isinstance(origin, bool) or not isinstance(origin, Integral) or not 0 <= origin < len(series):
        raise ValueError("origin must be an inclusive positional index inside the supplied history")
    history = series.iloc[:origin + 1].copy().reset_index(drop=True)
    if target not in history:
        raise ValueError(f"Missing feature target: {target}")
    values = history[target].astype(float).ffill()
    columns = {f"lag_{lag}": values.shift(lag - 1) for lag in range(1, 13)}
    for window in (3, 6, 12):
        rolling = values.rolling(window, min_periods=window)
        columns[f"rolling_mean_{window}"] = rolling.mean()
        columns[f"rolling_std_{window}"] = rolling.std(ddof=1)
    for window in (6, 12):
        x = np.arange(window, dtype=float) - (window - 1) / 2
        columns[f"slope_{window}"] = values.rolling(window, min_periods=window).apply(
            lambda y: float(np.dot(y, x) / np.dot(x, x)), raw=True)
    for name in history:
        if name.startswith(CONTEXT_PREFIXES) and history[name].notna().any():
            context = history[name].astype(float).ffill()
            columns[name] = context
            columns[f"{name}::mean_6"] = context.rolling(6, min_periods=6).mean()
    if "active_vm_count" in history:
        count = history.active_vm_count.astype(float).ffill()
        columns["active_vm_count"] = count
        columns["observed_count_change"] = count.diff()
        columns["observed_net_change_mean_6"] = count.diff().rolling(6, min_periods=6).mean()
    for name in ("observed_arrivals", "observed_departures"):
        if name in history:
            columns[name] = history[name].astype(float).ffill()
            columns[f"{name}_sum_6"] = columns[name].rolling(6, min_periods=6).sum()
    for name in ("burstiness", "aggregate_cpu_context"):
        if name in history:
            columns[name] = history[name].astype(float).ffill()
    frame = pd.DataFrame(columns)
    frame.iloc[:MIN_HISTORY - 1] = np.nan
    return frame


def make_features(series, origin, target=TARGET):
    """One feature vector computed exclusively from rows <= inclusive origin."""
    return feature_frame(series, origin, target).iloc[-1].copy()


def feature_availability(series, origin):
    """Method metadata for the report; not a feature-selection score."""
    features = make_features(series, origin)
    return pd.DataFrame({"feature": features.index,
                         "latest_row_allowed": origin,
                         "latest_timestamp_allowed": int(series.iloc[origin].timestamp),
                         "available_at_origin": features.notna().to_numpy()})
