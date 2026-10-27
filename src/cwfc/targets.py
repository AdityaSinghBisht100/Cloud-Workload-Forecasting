"""Full-grid operational targets and chronological train/holdout separation."""

from numbers import Integral

import numpy as np
import pandas as pd

from cwfc.quality import observation_grid, prepare_readings

REQUIRED_TARGETS = ("active_vm_count", "aggregate_cpu_utilization")
CPU_TARGETS = ("aggregate_cpu_utilization", "p95_vm_cpu", "aggregate_peak_cpu")
SEGMENT_PREFIX = "segment_cpu_utilization::"


def build_targets(readings: pd.DataFrame, metadata: pd.DataFrame | None = None) -> pd.DataFrame:
    """Build the full 300-second series before any temporal split.

    Apply the documented preparation policy once: average duplicate VM/time keys,
    exclude incomplete observations, reject invalid values, and never impute.
    P95 is the linearly interpolated 95th percentile of observed VM avg_cpu;
    hot_vm_share is a fraction in [0,1], using max_cpu strictly greater than 80.
    Optional category means use only vm_id and vm_category from metadata.
    Empty slots retain zero observed VMs and missing CPU/share/segment values.
    """
    grid = observation_grid(readings)
    clean = prepare_readings(readings)
    clean["is_hot"] = clean.max_cpu.gt(80)
    groups = clean.groupby("timestamp", observed=True)
    result = groups.agg(
        active_vm_count=("vm_id", "nunique"),
        aggregate_cpu_utilization=("avg_cpu", "mean"),
        hot_vm_share=("is_hot", "mean"),
        aggregate_peak_cpu=("max_cpu", "mean"),
    )
    result["p95_vm_cpu"] = groups.avg_cpu.quantile(0.95, interpolation="linear")
    result = result.reindex(grid)
    result["active_vm_count"] = result.active_vm_count.fillna(0).astype("int64")
    result["no_observations"] = result.active_vm_count.eq(0)
    result["missing_grid_timestamp"] = ~result.index.isin(readings.timestamp)

    if metadata is not None:
        missing = {"vm_id", "vm_category"} - set(metadata.columns)
        if missing:
            raise ValueError(f"Segment metadata missing columns: {sorted(missing)}")
        labels = metadata[["vm_id", "vm_category"]].copy()
        if labels.isna().any().any() or labels.vm_id.duplicated().any():
            raise ValueError("Segment metadata requires unique, nonmissing VM IDs and categories")
        if set(labels.vm_id) != set(readings.vm_id):
            raise ValueError("Segment metadata and readings VM populations must match")
        panel = clean.merge(labels, on="vm_id", how="left", validate="many_to_one")
        segments = panel.groupby(["timestamp", "vm_category"], observed=True).avg_cpu.mean().unstack("vm_category")
        categories = sorted(labels.vm_category.astype(str).unique())
        segments.columns = segments.columns.astype(str)
        segments = segments.reindex(index=grid, columns=categories)
        segments.columns = [f"{SEGMENT_PREFIX}{category}" for category in categories]
        result = result.join(segments)

    percent_columns = [*CPU_TARGETS, *[c for c in result if c.startswith(SEGMENT_PREFIX)]]
    result[percent_columns] = result[percent_columns].clip(0, 100)
    values = result[percent_columns].to_numpy(dtype=float)
    present = values[~np.isnan(values)]
    if not (np.isfinite(present).all() and ((present >= 0) & (present <= 100)).all()):
        raise AssertionError("CPU targets must be finite percentages in [0,100] where observed")
    columns = [*REQUIRED_TARGETS, "p95_vm_cpu", "hot_vm_share", "aggregate_peak_cpu",
               *[c for c in result if c.startswith(SEGMENT_PREFIX)],
               "no_observations", "missing_grid_timestamp"]
    return result[columns].reset_index()


def split_train_holdout(
    series: pd.DataFrame,
    n_train: int | None = 110,
    horizon: int = 24,
    min_train: int = 1,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reserve the final horizon rows; return independent chronological copies.

    n_train caps the contiguous training window immediately before the holdout;
    shorter datasets use all available training rows (at least min_train).
    Use None for an expanding window on longer datasets. On data-v1 this is
    exactly the first 110 rows and final 24. No rows inside either window are
    dropped, including missing targets. All calculations remain index-based;
    timestamps are checked as relative integer seconds, not calendar dates.
    """
    for name, value in (("horizon", horizon), ("min_train", min_train), ("n_train", n_train)):
        if name == "n_train" and value is None:
            continue
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise ValueError(f"{name} must be a positive integer" + (" or None" if name == "n_train" else ""))
    if n_train is not None and n_train < min_train:
        raise ValueError("n_train must be at least min_train")
    if len(series) < horizon + min_train:
        raise ValueError(
            f"Need at least {horizon + min_train} timestamps for horizon={horizon} "
            f"and min_train={min_train}; received {len(series)}"
        )
    if "timestamp" not in series or series.timestamp.isna().any():
        raise ValueError("Series requires nonmissing timestamp values")
    grid = observation_grid(series)
    if not np.array_equal(series.timestamp.to_numpy(), grid.to_numpy()):
        raise ValueError("Series must be sorted, unique and on the complete 300-second grid; build targets first")
    boundary = len(series) - horizon
    start = 0 if n_train is None else max(0, boundary - n_train)
    train = series.iloc[start:boundary].copy().reset_index(drop=True)
    holdout = series.iloc[boundary:].copy().reset_index(drop=True)
    return train, holdout
