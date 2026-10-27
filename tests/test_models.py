from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cwfc.models import (LIGHTGBM_PARAMS, build_model_inputs, count_churn_forecast,
                         inverse_error_weights, panel_forecast,
                         ridge_forecast, statistical_forecast)
from cwfc.targets import build_targets
from cwfc.validation import backtest_models, score_backtest


def fixture(n=78):
    rows = []
    for t in range(n):
        for vm in range(4):
            avg = 10 + vm * 3 + np.sin(t / 7) + (2 if 50 <= t < 60 else 0)
            rows.append([t * 300, str(vm), 0., 90., avg])
    raw = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    meta = pd.DataFrame({"vm_id": [str(i) for i in range(4)],
        "vm_category": ["A", "A", "A", "B"], "virtual_core_bucket": ["2"] * 4,
        "memory_gb_bucket": ["4"] * 4, "avg_cpu": [999.] * 4,
        "timestamp_vm_deleted": [999999] * 4})
    targets = build_targets(raw, meta)
    return raw, meta, targets


def inputs(n=78):
    raw, meta, targets = fixture(n)
    return build_model_inputs(raw, meta, targets)


def test_training_inputs_ignore_holdout_readings_and_lifetime_metadata():
    raw, meta, targets = fixture()
    train = targets.iloc[:60].copy()
    expected_series, expected_panel = build_model_inputs(raw, meta, train)
    raw.loc[raw.timestamp.ge(18000), "avg_cpu"] = 1e6
    meta["avg_cpu"], meta["timestamp_vm_deleted"] = -999., 0
    changed_series, changed_panel = build_model_inputs(raw, meta, train)
    pd.testing.assert_frame_equal(expected_series, changed_series)
    pd.testing.assert_frame_equal(expected_panel, changed_panel)
    assert expected_panel.timestamp.max() == 17700
    assert expected_series.observed_arrivals.iloc[1:].eq(0).all()


@pytest.mark.parametrize("name", ["ses", "ets_damped", "theta"])
def test_statistical_candidates_constant_history_and_finite_nonconstant_forecast(name):
    history, _ = inputs(60)
    values = statistical_forecast(history, 24, name)
    assert len(values) == 24 and np.isfinite(values).all()
    history["aggregate_cpu_utilization"] = 8.
    assert np.array_equal(statistical_forecast(history, 24, name), np.repeat(8., 24))


def test_direct_ridge_uses_only_matured_labels_and_skips_missing_labels():
    history, _ = inputs(60)
    forecast, info = ridge_forecast(history, 24)
    assert forecast.shape == (24,)
    assert info.latest_label_row.le(59).all()
    assert info.training_samples.iloc[0] > info.training_samples.iloc[-1]
    history.loc[40, "aggregate_cpu_utilization"] = np.nan
    _, changed = ridge_forecast(history, 24)
    assert (changed.training_samples <= info.training_samples).all()
    assert changed.training_samples.iloc[0] == info.training_samples.iloc[0] - 1
    assert pd.isna(history.loc[40, "aggregate_cpu_utilization"])


def test_panel_model_bottom_up_weights_and_deterministic_training():
    history, panel = inputs(60)
    params = {**LIGHTGBM_PARAMS, "n_estimators": 8, "min_child_samples": 5}
    first, detail, diagnostic = panel_forecast(history, panel, 24, params=params)
    repeated, _, _ = panel_forecast(history, panel, 24, params=params)
    assert np.array_equal(first, repeated)
    assert diagnostic["latest_label_row"] <= 59
    assert diagnostic["training_samples"] > 60
    assert sorted(detail.observed_vm_weight.unique()) == [1., 3.]
    recomputed = detail.assign(weighted=detail.yhat * detail.observed_vm_weight).groupby("step").agg(
        cpu=("weighted", "sum"), count=("observed_vm_weight", "sum"))
    assert np.allclose(first, recomputed.cpu / recomputed["count"])
    assert (detail.yhat.between(0, 100)).all()
    with pytest.raises(ValueError, match="after the forecast origin"):
        panel_forecast(history.iloc[:48], panel, 24, params=params)


def test_inverse_weights_and_zero_error_rule():
    evidence = pd.DataFrame({"model": ["a", "b", "c"], "y": [0., 0., 0.], "yhat": [1., 2., 4.]})
    weights = inverse_error_weights(evidence, ["a", "b", "c"], top_k=2)
    assert weights == pytest.approx({"a": 2 / 3, "b": 1 / 3})
    evidence.loc[0, "yhat"] = 0
    assert inverse_error_weights(evidence, ["a", "b", "c"]) == {"a": 1.}


def test_count_fallback_on_constant_and_changing_populations():
    assert count_churn_forecast(pd.DataFrame({"active_vm_count": [10, 10, 10]}), 2).tolist() == [10., 10.]
    assert count_churn_forecast(pd.DataFrame({"active_vm_count": [10, 12, 14]}), 2).tolist() == [16., 18.]
    assert count_churn_forecast(pd.DataFrame({"active_vm_count": [4, 2, 0]}), 2).tolist() == [0., 0.]


def test_backtest_boundaries_matured_ensemble_errors_and_future_poisoning(monkeypatch):
    # Keep model fitting out of this harness test; real fits are checked above.
    import cwfc.models as models
    monkeypatch.setattr(models, "statistical_forecast", lambda history, horizon, name:
                        np.repeat(history.aggregate_cpu_utilization.iloc[-1], horizon))
    monkeypatch.setattr(models, "ridge_forecast", lambda history, horizon:
                        (np.repeat(history.aggregate_cpu_utilization.mean(), horizon),
                         pd.DataFrame({"step": [1], "latest_label_row": [len(history) - 1]})))
    def fake_panel(history, panel, horizon):
        assert panel.timestamp.max() <= history.timestamp.max()
        return (np.repeat(history.aggregate_cpu_utilization.mean(), horizon),
                pd.DataFrame({"step": [1]}), {"latest_label_row": len(history) - 1})
    monkeypatch.setattr(models, "panel_forecast", fake_panel)
    history, panel = inputs()
    result = backtest_models(history, panel)
    weights = result["ensemble_weights"]
    observed = weights.latest_error_timestamp.notna()
    assert (weights.loc[observed, "latest_error_timestamp"] <= weights.loc[observed, "cutoff_timestamp"]).all()
    assert weights.loc[weights.origin.eq(54), "available_error_cases"].eq(6).all()
    assert weights.groupby("origin").weight.sum().round(12).eq(1).all()
    first = result["backtest"].query("origin == 48").reset_index(drop=True)
    history.loc[48:, "aggregate_cpu_utilization"] = 90.
    panel.loc[panel.timestamp.ge(14400), "mean_cpu"] = 90.
    changed = backtest_models(history, panel)["backtest"].query("origin == 48").reset_index(drop=True)
    pd.testing.assert_series_equal(first.yhat, changed.yhat)
    assert not first.y.equals(changed.y)


