import numpy as np
import pandas as pd
import pytest

from cwfc.targets import build_targets, split_train_holdout


def readings():
    return pd.DataFrame([
        [0, "a", 0., 80., 20.], [0, "a", 0., 100., 40.],
        [0, "b", 0., 80., 10.], [600, "a", 0., 100., 50.],
        [900, "a", 0., 100., np.nan],
    ], columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


def metadata():
    return pd.DataFrame({"vm_id": ["a", "b"], "vm_category": ["Interactive", "Unknown"],
                         "avg_cpu": [99., 99.], "timestamp_vm_deleted": [99999, 99999]})


def test_full_grid_targets_deduplicate_before_equal_vm_aggregation():
    result = build_targets(readings(), metadata())
    assert result.timestamp.tolist() == [0, 300, 600, 900]
    assert result.active_vm_count.tolist() == [2, 0, 1, 0]
    first = result.iloc[0]
    assert first.aggregate_cpu_utilization == 20  # VM a=30, VM b=10
    assert first.p95_vm_cpu == pytest.approx(29)
    assert first.hot_vm_share == .5  # max=80 is not hot; a's deduplicated max=90 is
    assert first.aggregate_peak_cpu == 85
    assert first["segment_cpu_utilization::Interactive"] == 30
    assert first["segment_cpu_utilization::Unknown"] == 10
    assert np.isnan(result.loc[2, "segment_cpu_utilization::Unknown"])
    values = ["aggregate_cpu_utilization", "p95_vm_cpu", "hot_vm_share", "aggregate_peak_cpu",
              "segment_cpu_utilization::Interactive", "segment_cpu_utilization::Unknown"]
    assert result.loc[[1, 3], values].isna().all().all()
    assert result.no_observations.tolist() == [False, True, False, True]
    assert result.missing_grid_timestamp.tolist() == [False, True, False, False]


@pytest.mark.parametrize("meta", [None, metadata()])
def test_no_valid_observations(meta):
    frame = readings()
    frame["avg_cpu"] = np.nan
    result = build_targets(frame, meta)
    assert len(result) == 4
    assert result.active_vm_count.eq(0).all()
    assert result.aggregate_cpu_utilization.isna().all()
    if meta is None:
        assert not any(c.startswith("segment_") for c in result)
    else:
        assert result.filter(like="segment_cpu_utilization::").isna().all().all()


def test_lifetime_metadata_cannot_change_targets_and_inputs_are_preserved():
    raw = readings()
    before = raw.copy(deep=True)
    meta = metadata()
    result = build_targets(raw, meta)
    meta["avg_cpu"] = -999
    meta["timestamp_vm_deleted"] = 0
    pd.testing.assert_frame_equal(result, build_targets(raw, meta))
    pd.testing.assert_frame_equal(raw, before)


@pytest.mark.parametrize("kind", ["duplicate", "missing_category", "population"])
def test_segment_metadata_errors_do_not_silently_change_weights(kind):
    meta = metadata()
    if kind == "duplicate":
        meta = pd.concat([meta, meta.iloc[[0]]])
    elif kind == "missing_category":
        meta.loc[0, "vm_category"] = None
    else:
        meta.loc[0, "vm_id"] = "not_in_readings"
    with pytest.raises(ValueError, match="metadata"):
        build_targets(readings(), meta)


def series(n):
    return pd.DataFrame({"timestamp": np.arange(n) * 300, "aggregate_cpu_utilization": np.arange(n, dtype=float)})


def test_release_split_is_110_then_24_without_overlap_and_returns_copies():
    full = series(134)
    train, holdout = split_train_holdout(full)
    assert len(train) == 110 and len(holdout) == 24
    assert train.timestamp.tolist() == list(range(0, 33000, 300))
    assert holdout.timestamp.tolist() == list(range(33000, 40200, 300))
    assert train.timestamp.max() < holdout.timestamp.min()
    train.loc[0, "aggregate_cpu_utilization"] = -1
    assert full.loc[0, "aggregate_cpu_utilization"] == 0


def test_smoke_split_preserves_gaps_and_final_horizon():
    full = build_targets(readings())
    train, holdout = split_train_holdout(full, horizon=2, min_train=2)
    assert train.timestamp.tolist() == [0, 300]
    assert holdout.timestamp.tolist() == [600, 900]
    assert np.isnan(train.aggregate_cpu_utilization.iloc[-1])
    assert np.isnan(holdout.aggregate_cpu_utilization.iloc[-1])
    with pytest.raises(ValueError, match="Need at least 5 timestamps"):
        split_train_holdout(full, horizon=2, min_train=3)


def test_longer_series_uses_latest_training_window_or_all_history():
    train, holdout = split_train_holdout(series(150))
    assert train.timestamp.iloc[0] == 16 * 300
    assert len(train) == 110
    assert holdout.timestamp.iloc[0] == 126 * 300
    expanding, same_holdout = split_train_holdout(series(150), n_train=None)
    assert len(expanding) == 126
    pd.testing.assert_frame_equal(holdout, same_holdout)


@pytest.mark.parametrize("kwargs", [{"horizon": 0}, {"horizon": True}, {"n_train": 0},
                                   {"min_train": 0}, {"n_train": 2, "min_train": 3}])
def test_invalid_split_sizes(kwargs):
    with pytest.raises(ValueError):
        split_train_holdout(series(134), **kwargs)


@pytest.mark.parametrize("timestamps", [[300, 0, 600], [0, 0, 300], [0, 300, 900]])
def test_split_requires_prebuilt_chronological_full_grid(timestamps):
    with pytest.raises(ValueError, match="sorted, unique"):
        split_train_holdout(pd.DataFrame({"timestamp": timestamps}), horizon=1)


def test_future_readings_do_not_change_training_targets():
    raw = readings().fillna({"avg_cpu": 50.})
    train, _ = split_train_holdout(build_targets(raw, metadata()), horizon=2)
    raw.loc[raw.timestamp.ge(600), "avg_cpu"] = 99.
    changed_train, _ = split_train_holdout(build_targets(raw, metadata()), horizon=2)
    pd.testing.assert_frame_equal(train, changed_train)
