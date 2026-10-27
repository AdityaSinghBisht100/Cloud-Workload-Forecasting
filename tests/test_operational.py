from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from cwfc.direct import (anchor_features, behavior_labels, direct_examples, event_examples,
    fit_calibration, low_use_mask, make_panel)
from cwfc.operational import fit_forecasts, load_recipe, run
from cwfc.operational_evaluation import candidate_policies, headroom_backtest

ROOT = Path(__file__).resolve().parents[1]


def fixture(n=84, vms=8):
    rows = []
    for t in range(n):
        for vm in range(vms):
            if vm == 0:
                cpu, peak = .03, .08
            elif vm < 4:
                cpu = .2 + .05*np.sin(t/3+vm)
                peak = 8. if (t+vm*3) % 19 == 0 else cpu+.1
            else:
                cpu = vm*3+np.sin(t/5+vm)
                peak = cpu+2
            rows.append((t*300, f"vm{vm}", cpu/2, peak, cpu))
    readings = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    metadata = pd.DataFrame({"vm_id": [f"vm{i}" for i in range(vms)],
        "subscription_id": ["s"]*vms, "deployment_id": ["d"]*vms,
        "timestamp_vm_created": [0]*vms, "timestamp_vm_deleted": [999999]*vms,
        "max_cpu": [99.]*vms, "avg_cpu": [99.]*vms, "p95_max_cpu": [99.]*vms,
        "vm_category": ["A" if i % 2 else "B" for i in range(vms)],
        "virtual_core_bucket": ["2"]*vms, "memory_gb_bucket": ["4"]*vms})
    return readings, metadata


def recipe():
    cfg = load_recipe(ROOT / "configs/operational.yaml")
    for key in ["cpu_parameters", "classifier_parameters"]:
        cfg[key].update(n_estimators=5, min_child_samples=5, n_jobs=1, num_leaves=7)
    cfg["split"].update(n_train=60, min_train=36)
    cfg["validation"].update(min_train=36)
    return cfg


def test_anchor_features_ignore_future_and_reconstruct_metadata_summaries():
    readings, metadata = fixture()
    cfg = recipe()
    panel = make_panel(readings, metadata, np.arange(48)*300)
    x, baseline, names = anchor_features(panel, 15)
    assert np.allclose(x[:, names.index("past_max_cpu")], panel.maximum[:, :16].max(axis=1))
    assert np.allclose(x[:, names.index("past_p95_max_cpu")], np.quantile(panel.maximum[:, :16], .95, axis=1))
    assert np.allclose(baseline, panel.avg[:, :16].mean(axis=1))
    for field in ["avg", "minimum", "maximum", "filled"]:
        changed = getattr(panel, field).copy()
        changed[:, 16:] = 999
        setattr(panel, field, changed)
    actual, actual_mean, _ = anchor_features(panel, 15)
    # Copying changes array strides and floating-point dot-product summation order.
    np.testing.assert_allclose(x, actual, rtol=1e-6, atol=1e-7)
    assert np.array_equal(baseline, actual_mean)
    assert low_use_mask(panel, 15, cfg["activity"]).shape == (8,)
    assert len(behavior_labels(panel, 15, cfg["activity"])) == 8


def test_direct_and_event_labels_mature_and_horizons_have_equal_total_weight():
    readings, metadata = fixture()
    cfg = recipe()
    panel = make_panel(readings, metadata, np.arange(48)*300)
    x, y, busy, weights, names, info = direct_examples(panel, 24, 6, cfg["activity"])
    assert info["latest_label_row"] < 48
    assert len(y) == sum(info["samples_by_horizon"])
    totals = [weights[x[:, -1] == h].sum() for h in range(1, 25)]
    assert np.allclose(totals, totals[0])
    assert np.allclose(y[:8], panel.avg[:, 12]-panel.avg[:, :12].mean(axis=1))
    assert set(busy) <= {0, 1}
    ex, ey, en, ei = event_examples(panel, 24, cfg["activity"])
    assert ei["latest_event_label_row"] < 48
    assert len(ex) == len(ey)
    assert set(ey) == {0, 1}
    assert "past_p95_max_cpu" in en


def test_missing_future_event_labels_are_not_filled():
    readings, metadata = fixture()
    cfg = recipe()
    complete = make_panel(readings, metadata, np.arange(48)*300)
    _, _, _, expected = event_examples(complete, 24, cfg["activity"])
    readings = readings.loc[~((readings.vm_id == "vm0") & (readings.timestamp == 30*300))]
    partial = make_panel(readings, metadata, np.arange(48)*300)
    _, _, _, actual = event_examples(partial, 24, cfg["activity"])
    assert actual["event_training_samples"] < expected["event_training_samples"]


def test_forecasts_ignore_future_readings_lifetime_metadata_and_are_deterministic():
    readings, metadata = fixture()
    cfg = recipe()
    timestamps = pd.Series(np.arange(48)*300)
    expected = fit_forecasts(readings, metadata, timestamps, cfg)
    readings.loc[readings.timestamp >= 48*300, ["min_cpu", "avg_cpu", "max_cpu"]] = -1e6
    metadata.loc[:, ["max_cpu", "avg_cpu", "p95_max_cpu", "timestamp_vm_deleted"]] = -1e6
    actual = fit_forecasts(readings, metadata, timestamps, cfg)
    for model in expected[1]:
        assert np.array_equal(expected[1][model], actual[1][model])
    assert np.array_equal(expected[2], actual[2])
    assert np.allclose(expected[3], actual[3], equal_nan=True)
    assert expected[5]["latest_label_row"] < 48
    assert expected[5]["latest_event_label_row"] < 48


def test_calibration_excludes_unmatured_labels():
    pool = pd.DataFrame({"label_timestamp": np.arange(60)*300,
        "raw_probability": np.tile([.1, .8], 30), "actual": np.tile([0, 1], 30)})
    first = fit_calibration(pool, 40*300)
    assert first.samples == 41 and first.latest_timestamp == 40*300
    pool.loc[pool.label_timestamp > 40*300, ["raw_probability", "actual"]] = 0
    second = fit_calibration(pool, 40*300)
    assert np.array_equal(first.model.coef_, second.model.coef_)
    assert np.array_equal(first.predict([.2, .9]), second.predict([.2, .9]))


def test_matched_policy_counts_and_hysteresis():
    cfg = recipe()["activity"]
    ids = np.array(["a", "b", "c", "d"])
    risk = np.array([.04, .15, .3, np.nan])
    candidate = np.array([True, True, True, False])
    masks = candidate_policies(risk, candidate, np.arange(4), np.arange(4), ids, cfg,
                               {"risk_hysteresis": {"b"}})
    assert masks["risk_hysteresis"].tolist() == [True, True, False, False]
    for p in cfg["risk_budgets"]:
        assert masks[f"risk_le_{p:g}"].sum() == masks[f"lowest_peak_matched_{p:g}"].sum()
        assert masks[f"risk_le_{p:g}"].sum() == masks[f"lowest_average_matched_{p:g}"].sum()


def test_headroom_calibration_does_not_use_current_or_unmatured_errors():
    rows = [{"origin": origin, "step": h, "timestamp": (origin+h-1)*300,
             "model": "test", "y": 10., "yhat": 9.}
            for origin in [48, 54, 60] for h in range(1, 25)]
    frame = pd.DataFrame(rows)
    settings = recipe()["policy"]
    expected = headroom_backtest(frame, settings)
    frame.loc[frame.timestamp >= 54*300, "y"] = 99.
    actual = headroom_backtest(frame, settings)
    columns = ["capacity_proxy_pct", "calibration_samples"]
    pd.testing.assert_frame_equal(expected.loc[expected.origin <= 54, columns].reset_index(drop=True),
                                  actual.loc[actual.origin <= 54, columns].reset_index(drop=True))


def test_end_to_end_separate_outputs_and_holdout_poisoning(tmp_path):
    readings, metadata = fixture()
    data = tmp_path / "data"
    data.mkdir()
    readings.to_csv(data / "vm_cpu_readings.csv.gz", index=False)
    metadata.to_csv(data / "vm_metadata.csv.gz", index=False)
    config = tmp_path / "recipe.yaml"
    config.write_text(yaml.safe_dump(recipe()), encoding="utf-8")
    output, activity = tmp_path / "cpu.csv", tmp_path / "activity.csv"
    record = run(data, config, tmp_path / "run", output, activity)
    first = output.read_bytes(), activity.read_bytes()
    cpu, operational = pd.read_csv(output), pd.read_csv(activity)
    assert cpu.shape == (24, 3)
    assert operational.shape[0] == 24
    assert np.isfinite(operational.to_numpy(dtype=float)).all()
    assert cpu.active_vm_count.eq(8).all()
    assert operational.expected_cpu_busy_count.between(0, 8).all()
    assert not record["holdout_scored"]
    assert record["fit"]["latest_label_row"] < 60
    detail = pd.read_csv(tmp_path / "run/vm_predictions.csv.gz")
    assert np.allclose(detail.groupby("step").predicted_cpu.mean(), cpu.aggregate_cpu_utilization)
    assert (tmp_path / "run/folds/fold_01_vm_predictions.csv.gz").exists()
    assert (tmp_path / "run/low_use_review_candidates.csv").exists()
    readings.loc[readings.timestamp >= 60*300, ["min_cpu", "avg_cpu", "max_cpu"]] = 50.
    readings.to_csv(data / "vm_cpu_readings.csv.gz", index=False)
    run(data, config, tmp_path / "poisoned", output, activity)
    assert first == (output.read_bytes(), activity.read_bytes())
    with pytest.raises(ValueError, match="different paths"):
        run(data, config, tmp_path / "blocked", activity, activity)

