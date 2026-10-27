"""Diagnostics and explicit, conservative reading preparation."""

import logging

import numpy as np
import pandas as pd

CPU = ["min_cpu", "avg_cpu", "max_cpu"]
LOGGER = logging.getLogger(__name__)


def observation_grid(readings: pd.DataFrame, cadence: int = 300) -> pd.Index:
    """Grid from first to last observed timestamp, including wholly missing slots."""
    if cadence <= 0 or not isinstance(cadence, int):
        raise ValueError("cadence must be a positive integer number of seconds")
    timestamps = readings["timestamp"].dropna().to_numpy(dtype=float)
    if not len(timestamps):
        raise ValueError("No timestamps available")
    if not np.isfinite(timestamps).all() or (timestamps % 1 != 0).any():
        raise ValueError("Timestamps must be finite integer relative seconds")
    start, stop = int(timestamps.min()), int(timestamps.max())
    if ((timestamps - start) % cadence != 0).any():
        raise ValueError(f"Timestamps are not aligned to a {cadence}-second grid")
    return pd.Index(range(start, stop + cadence, cadence), name="timestamp")


def _invalid_cpu(frame):
    return ((frame[CPU] < 0) | (frame[CPU] > 100)).any(axis=1) | (
        (frame.min_cpu > frame.avg_cpu) | (frame.avg_cpu > frame.max_cpu)
    )


def prepare_readings(readings: pd.DataFrame, cadence: int = 300) -> pd.DataFrame:
    """Exclude incomplete readings, reject invalid CPU, average duplicate keys.

    No VM-level values are imputed. Diagnostics should be run on raw input first.
    """
    observation_grid(readings, cadence)
    if readings[["vm_id", "timestamp"]].isna().any().any():
        raise ValueError("Missing vm_id or timestamp; repair source keys before preprocessing")
    if _invalid_cpu(readings).any():
        raise ValueError("CPU outside [0,100] or min <= avg <= max violated; inspect quality report")
    complete = readings[CPU].notna().all(axis=1)
    LOGGER.info("Excluded %d incomplete CPU readings (no imputation)", (~complete).sum())
    clean = readings.loc[complete, ["vm_id", "timestamp", *CPU]].copy()
    duplicates = clean.duplicated(["vm_id", "timestamp"]).sum()
    LOGGER.info("Averaging %d duplicate VM/timestamp rows", duplicates)
    clean = clean.groupby(["vm_id", "timestamp"], observed=True, as_index=False)[CPU].mean()
    return clean.sort_values(["timestamp", "vm_id"]).reset_index(drop=True)


def observation_summary(readings: pd.DataFrame, cadence: int = 300) -> pd.DataFrame:
    """Audit observed populations on a full grid; no training-target imputation.

    The grid uses raw timestamps, so slots with only invalid/missing readings
    remain visible after preparation. For operational targets and a temporal
    split use cwfc.targets; this function remains the observation audit view.
    """
    grid = observation_grid(readings, cadence)
    clean = prepare_readings(readings, cadence)
    summary = clean.groupby("timestamp", observed=True).agg(
        active_vm_count=("vm_id", "nunique"), aggregate_cpu_utilization=("avg_cpu", "mean")
    ).reindex(grid)
    summary["active_vm_count"] = summary.active_vm_count.fillna(0).astype("int64")
    summary["no_observations"] = summary.active_vm_count.eq(0)
    summary["missing_grid_timestamp"] = ~summary.index.isin(readings.timestamp.dropna())
    return summary.reset_index()


def run_checks(readings: pd.DataFrame, metadata: pd.DataFrame, cadence: int = 300) -> dict:
    """Return diagnostic tables, including per-VM coverage and censoring flags.

    Checks do not silently fix data and include the entire release for data
    integrity only. They must not be used for holdout-based model selection.
    """
    timestamps = np.sort(readings.timestamp.dropna().unique().astype("int64"))
    if not len(timestamps):
        raise ValueError("No timestamps available")
    diffs = np.diff(timestamps)
    try:
        grid = observation_grid(readings, cadence)
        aligned = True
        missing_grid = len(grid.difference(timestamps))
    except ValueError:
        aligned, missing_grid = False, None
        grid = None
    valid = readings[CPU].notna().all(axis=1) & ~_invalid_cpu(readings)
    valid &= readings[["vm_id", "timestamp"]].notna().all(axis=1)
    coverage = readings.groupby("vm_id", observed=True).timestamp.nunique().rename("observed_timestamps").to_frame()
    coverage["valid_timestamps"] = readings.loc[valid].groupby("vm_id", observed=True).timestamp.nunique().reindex(coverage.index, fill_value=0)
    coverage["expected_timestamps"] = len(grid) if grid is not None else np.nan
    coverage["coverage_fraction"] = coverage.valid_timestamps / coverage.expected_timestamps
    reading_ids = set(readings.vm_id.dropna())
    metadata_ids = set(metadata.vm_id.dropna())
    lifetimes = metadata[["vm_id", "timestamp_vm_created", "timestamp_vm_deleted"]].copy()
    lifetimes["duration_seconds"] = lifetimes.timestamp_vm_deleted - lifetimes.timestamp_vm_created
    lifetimes["at_or_before_window_start"] = lifetimes.timestamp_vm_created.le(timestamps[0])
    lifetimes["creation_at_trace_zero"] = lifetimes.timestamp_vm_created.eq(0)
    lifetimes["deletion_after_window"] = lifetimes.timestamp_vm_deleted.gt(timestamps[-1])
    lifetimes["deletion_within_6h_after_window"] = lifetimes.timestamp_vm_deleted.between(int(timestamps[-1]) + 1, int(timestamps[-1]) + 21600)
    facts = {
        "reading_rows": len(readings), "distinct_timestamps": len(timestamps),
        "first_timestamp_seconds": int(timestamps[0]), "last_timestamp_seconds": int(timestamps[-1]),
        "cadence_seconds": cadence, "timestamp_grid_aligned": aligned,
        "all_timestamp_diffs_equal_cadence": bool(len(diffs) and (diffs == cadence).all()),
        "missing_grid_timestamps": missing_grid,
        "duplicate_vm_timestamp_rows": int(readings.duplicated(["vm_id", "timestamp"]).sum()),
        "missing_cells": int(readings.isna().sum().sum()),
        "cpu_out_of_range_rows": int(((readings[CPU] < 0) | (readings[CPU] > 100)).any(axis=1).sum()),
        "cpu_order_violations": int(((readings.min_cpu > readings.avg_cpu) | (readings.avg_cpu > readings.max_cpu)).sum()),
        "readings_vm_count": len(reading_ids), "metadata_vm_count": len(metadata_ids),
        "vm_populations_match": reading_ids == metadata_ids,
        "metadata_duplicate_vm_rows": int(metadata.duplicated("vm_id").sum()),
        "metadata_missing_cells": int(metadata.isna().sum().sum()),
        "partial_coverage_vms": int(coverage.coverage_fraction.lt(1).sum()),
        "invalid_lifetime_order_rows": int(lifetimes.duration_seconds.lt(0).sum()),
    }
    return {
        "checks": pd.DataFrame(facts.items(), columns=["check", "value"]),
        "missingness": pd.concat([readings.isna().sum().rename("readings"), metadata.isna().sum().rename("metadata")], axis=1),
        "cadence": pd.Series(diffs).value_counts().rename_axis("difference_seconds").rename("count").to_frame(),
        "coverage": coverage.reset_index(), "lifetimes": lifetimes,
        "lifetime_distribution": lifetimes[["timestamp_vm_created", "timestamp_vm_deleted", "duration_seconds"]].describe(),
        "population_mismatches": {"readings_only": sorted(reading_ids - metadata_ids), "metadata_only": sorted(metadata_ids - reading_ids)},
    }
