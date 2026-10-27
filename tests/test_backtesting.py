import numpy as np
import pandas as pd
import pytest

import cwfc.validation as validation
from cwfc.targets import split_train_holdout
from cwfc.validation import (backtest_baselines, rolling_origin_index,
                             rolling_origin_splits, score_backtest, select_moving_average)


def series(n=110):
    return pd.DataFrame({"timestamp": np.arange(n) * 300,
        "active_vm_count": np.full(n, 100),
        "aggregate_cpu_utilization": 10 + np.sin(np.arange(n) / 6)})


def test_default_origins_and_half_open_boundaries():
    folds = rolling_origin_index()
    assert [s.origin for s in folds] == [48, 54, 60, 66, 72, 78, 84]
    for s in folds:
        assert s.train_start == 0
        assert s.train_end == s.validation_start
        assert s.validation_end - s.validation_start == 24
        assert s.validation_end <= 110
    assert folds[-1].validation_end == 108


@pytest.mark.parametrize("kwargs", [{"step": 0}, {"step": -1}, {"n": 71},
                                   {"horizon": True}, {"min_train": 1.5}])
def test_invalid_index_split_parameters(kwargs):
    with pytest.raises(ValueError):
        rolling_origin_index(**kwargs)


def test_relative_seconds_are_preserved_and_legacy_step_cannot_loop():
    folds = rolling_origin_splits(series(), step=6)
    assert folds[0].train_end == 14100
    assert not isinstance(folds[0].train_end, pd.Timestamp)
    assert folds[0].validation_start == 14400
    assert folds[-1].validation_end == 32100
    with pytest.raises(ValueError):
        rolling_origin_splits(series(), step=0)


def test_one_long_table_matches_all_fold_model_target_forecasts(monkeypatch):
    original = validation.baseline_forecasts
    seen = []
    def inspect_inputs(history, template, target):
        assert template.columns.tolist() == ["timestamp"]
        assert history.columns.tolist() == ["timestamp", target]
        assert history.timestamp.max() < template.timestamp.min()
        seen.append((len(history), target))
        return original(history, template, target)
    monkeypatch.setattr(validation, "baseline_forecasts", inspect_inputs)
    result = backtest_baselines(series())
    assert len(seen) == 14
    assert len(result) == 7 * 24 * 7 * 2
    assert result.groupby(["fold", "model", "target"]).size().eq(24).all()
    assert not result.duplicated(["fold", "step", "model", "target"]).any()
    assert result.step.min() == 1 and result.step.max() == 24
    assert result.timestamp.max() == 32100
    assert np.isfinite(result[["y", "yhat"]]).all().all()


def test_future_validation_values_cannot_change_earlier_origin_forecasts():
    raw = series()
    first = backtest_baselines(raw)
    raw.loc[48:, "aggregate_cpu_utilization"] = 95
    changed = backtest_baselines(raw)
    a = first.query("origin == 48").reset_index(drop=True)
    b = changed.query("origin == 48").reset_index(drop=True)
    pd.testing.assert_series_equal(a.yhat, b.yhat)
    assert not a.y.equals(b.y)


def test_holdout_changes_cannot_affect_training_cv():
    raw = series(134)
    train, _ = split_train_holdout(raw)
    first = backtest_baselines(train)
    raw.loc[110:, "aggregate_cpu_utilization"] = 99
    changed_train, _ = split_train_holdout(raw)
    pd.testing.assert_frame_equal(first, backtest_baselines(changed_train))


def test_scores_zero_reference_and_deterministic_ma_ties():
    raw = series()
    raw["aggregate_cpu_utilization"] = 0.
    table = backtest_baselines(raw)
    scores = score_backtest(table)
    assert scores.mae.eq(0).all() and scores.rmse.eq(0).all()
    assert scores.smape_pct.eq(0).all()
    assert scores.skill_vs_persistence.isna().all()
    selected = select_moving_average(scores)
    assert selected.window.eq(3).all()
    assert scores.forecast_cases.eq(168).all()


def test_scoring_uses_long_table_and_selection_uses_mae_only():
    scores = score_backtest(backtest_baselines(series()))
    for row in select_moving_average(scores).itertuples():
        candidates = scores.loc[scores.target.eq(row.target) & scores.model.str.startswith("moving_average_")]
        assert row.mae == candidates.mae.min()
    cpu = scores.loc[scores.target.eq("aggregate_cpu_utilization")]
    reference = cpu.loc[cpu.model.eq("persistence"), "mae"].iloc[0]
    assert np.allclose(cpu.skill_vs_persistence, 1 - cpu.mae / reference)


def test_missing_targets_and_broken_grids_require_source_review():
    raw = series()
    raw.loc[2, "aggregate_cpu_utilization"] = np.nan
    with pytest.raises(ValueError, match="not imputed"):
        backtest_baselines(raw)
    with pytest.raises(ValueError, match="complete five-minute"):
        backtest_baselines(series().drop(index=3))
    with pytest.raises(ValueError, match="min_train >= 24"):
        backtest_baselines(series(), min_train=12)


def test_domain_clipping_applies_to_extrapolated_drift():
    raw = series(72)
    raw["aggregate_cpu_utilization"] = np.linspace(20, 0, 72)
    raw["active_vm_count"] = np.linspace(100, 0, 72)
    table = backtest_baselines(raw)
    assert table.yhat.ge(0).all()
    assert table.loc[table.target.eq("aggregate_cpu_utilization"), "yhat"].le(100).all()
