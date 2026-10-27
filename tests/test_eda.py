import numpy as np
import pandas as pd
import pytest

from cwfc.eda import analyze_workload, change_scan, load_windows, peak_drivers
from cwfc.targets import build_targets, split_train_holdout


def sample():
    rows = []
    for i in range(20):
        for vm in ("a", "b"):
            if i == 4 or (i == 8 and vm == "b"):
                continue
            average = (30 if 10 <= i < 15 else 10) + (2 if vm == "a" else 0)
            rows.append([i * 300, vm, 0., 90., float(average)])
    readings = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    metadata = pd.DataFrame({"vm_id": ["a", "b"], "vm_category": ["Interactive", "Unknown"],
        "virtual_core_bucket": ["2", "4"], "memory_gb_bucket": ["4", "8"],
        "subscription_id": ["s1", "s1"], "deployment_id": ["d1", "d2"],
        "avg_cpu": [999., 999.], "timestamp_vm_deleted": [999999, 999999]})
    return readings, metadata


def test_training_only_eda_and_metadata_leakage_exclusion():
    readings, meta = sample()
    train, _ = split_train_holdout(build_targets(readings), horizon=4)
    first = analyze_workload(readings, meta, train)
    readings.loc[readings.timestamp.ge(4800), "avg_cpu"] = 89.
    meta["avg_cpu"] = -1000.
    meta["timestamp_vm_deleted"] = 0
    second = analyze_workload(readings, meta, train)
    for key in ("indicators", "segments", "categories", "peaks", "lows", "vm_drivers", "concentration"):
        pd.testing.assert_frame_equal(first[key], second[key])
    assert first["metrics"]["high_threshold"] == train.aggregate_cpu_utilization.quantile(.8)
    assert first["series"].timestamp.max() == 4500
    assert first["vm_statistics"].observations.max() == 15


def test_driver_reconciliation_with_changing_population():
    panel = pd.DataFrame([
        [0, "a", "X", 20.], [0, "b", "Y", 40.],
        [300, "a", "X", 40.],
        [600, "a", "X", 60.], [600, "b", "Y", 80.],
    ], columns=["timestamp", "vm_id", "vm_category", "avg_cpu"])
    series = pd.DataFrame({"timestamp": [0, 300, 600], "aggregate_cpu_utilization": [30., 40., 70.]})
    windows = load_windows(series, 60)
    result, vms, categories = peak_drivers(panel, series, windows)
    assert result.increase_pp.iloc[0] == 35
    assert vms.contribution_pp.sum() == pytest.approx(35)
    assert categories.contribution_pp.sum() == pytest.approx(35)
    assert result.baseline_samples.iloc[0] == 2


def test_windows_are_strict_contiguous_and_missing_values_break_them():
    series = pd.DataFrame({"timestamp": np.arange(7) * 300,
                           "aggregate_cpu_utilization": [1., 2., 3., np.nan, 3., 2., 1.]})
    peaks = load_windows(series, 2)
    assert peaks.start_index.tolist() == [2, 4]
    assert peaks.samples.tolist() == [1, 1]
    lows = load_windows(series, 2, above=False)
    assert lows.start_index.tolist() == [0, 6]


def test_change_scan_locates_known_step_and_skips_missing_windows():
    series = pd.DataFrame({"timestamp": np.arange(18) * 300,
                           "aggregate_cpu_utilization": [0.] * 9 + [10.] * 9})
    scan, candidates = change_scan(series, window=3)
    strongest = candidates.loc[candidates.absolute_change_pp.idxmax()]
    assert strongest.boundary_index == 9
    assert strongest.change_pp == 10
    series.loc[8, "aggregate_cpu_utilization"] = np.nan
    scan, _ = change_scan(series, window=3)
    assert not scan.boundary_index.between(6, 11).any()


def test_inconsistent_targets_fail_instead_of_mixing_data():
    readings, meta = sample()
    train, _ = split_train_holdout(build_targets(readings), horizon=4)
    train.loc[0, "aggregate_cpu_utilization"] = 99
    with pytest.raises(ValueError, match="do not match"):
        analyze_workload(readings, meta, train)
