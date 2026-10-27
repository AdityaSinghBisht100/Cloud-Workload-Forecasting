import numpy as np
import pandas as pd
import pytest

from cwfc.targets import build_targets
from cwfc.model import PARAMS, _features, vm_forecast, vm_history
from cwfc.validation import backtest_final


def fixture(n=64):
    rows = []
    for t in range(n):
        for vm in range(5):
            cpu = 10 + vm * 4 + np.sin(t / 5 + vm)
            rows.append((t * 300, f"vm{vm}", cpu - 1, cpu, cpu + 1))
    readings = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "avg_cpu", "max_cpu"])
    metadata = pd.DataFrame({"vm_id": [f"vm{i}" for i in range(5)],
                             "vm_category": ["A", "A", "A", "B", "B"],
                             "virtual_core_bucket": ["2"] * 5, "memory_gb_bucket": ["4"] * 5})
    return readings, metadata


SMALL = {**PARAMS, "n_estimators": 8, "min_child_samples": 5, "n_jobs": 1}


def test_all_vm_readings_used_and_bottom_up_reconciles():
    readings, metadata = fixture()
    timestamps = pd.Series(np.arange(48) * 300)
    first, detail, info, importance = vm_forecast(readings, metadata, timestamps, 24, SMALL)
    again, _, _, _ = vm_forecast(readings, metadata, timestamps, 24, SMALL)
    assert info["raw_training_readings"] == 48 * 5
    assert info["training_samples"] == (48 - 12) * 5
    assert info["latest_label_row"] == 47
    assert info["last_training_timestamp"] == 47 * 300
    assert len(detail) == 24 * 5
    assert np.array_equal(first, again)
    assert np.allclose(first, detail.groupby("step").yhat.mean())
    assert detail.yhat.between(0, 100).all()
    assert set(importance.feature) >= {"lag_1", "vm_history_mean", "vm_category"}


def test_future_cpu_and_lifetime_metadata_cannot_change_forecast():
    readings, metadata = fixture()
    timestamps = pd.Series(np.arange(48) * 300)
    expected, _, _, _ = vm_forecast(readings, metadata, timestamps, 6, SMALL)
    readings.loc[readings.timestamp >= 48 * 300, ["min_cpu", "avg_cpu", "max_cpu"]] = -1e6
    metadata["timestamp_vm_deleted"] = -123
    metadata["avg_cpu"] = 99.
    actual, _, _, _ = vm_forecast(readings, metadata, timestamps, 6, SMALL)
    assert np.array_equal(expected, actual)


def test_feature_row_does_not_read_future_and_vm_histories_are_separate():
    readings, metadata = fixture()
    values, static, _, _ = vm_history(readings, metadata, pd.Series(np.arange(48) * 300))
    expected, means = _features(values, values.copy(), 15, static)
    assert np.allclose(expected[:, 0], values[:, 15])
    assert np.allclose(means, values[:, :16].mean(axis=1))
    values = values.copy()
    values[:, 16:] = 99
    actual, _ = _features(values, values.copy(), 15, static)
    assert np.array_equal(expected, actual)


def test_missing_labels_not_imputed_and_absent_cutoff_vm_not_forecast():
    readings, metadata = fixture()
    readings = readings.loc[~((readings.vm_id == "vm0") & readings.timestamp.isin([30 * 300, 47 * 300]))]
    _, detail, info, _ = vm_forecast(readings, metadata, pd.Series(np.arange(48) * 300), 3, SMALL)
    assert info["training_samples"] == 36 * 5 - 2
    assert info["missing_vm_timestamp_cells"] == 2
    assert info["forecast_vm_count"] == 4
    assert "vm0" not in set(detail.vm_id)


def test_backtest_labels_mature_and_same_cases_as_references():
    readings, metadata = fixture()
    series = build_targets(readings, metadata)
    evidence, info = backtest_final(readings, metadata, series, horizon=6, min_train=48, step=6, params=SMALL)
    assert (info.latest_label_row < info.origin).all()
    assert np.array_equal(info.raw_training_readings, info.origin * 5)
    vm = evidence.loc[evidence.model.eq("global_vm_lightgbm")]
    reference = evidence.loc[evidence.model.eq("vm_persistence")]
    assert np.array_equal(vm[["fold", "step", "timestamp", "y"]], reference[["fold", "step", "timestamp", "y"]])


def test_insufficient_history_rejected():
    readings, metadata = fixture()
    with pytest.raises(ValueError, match="13"):
        vm_forecast(readings, metadata, pd.Series(np.arange(12) * 300), 6, SMALL)
