import numpy as np
import pandas as pd
import pytest

from cwfc.features import feature_frame, make_features


def series(n=60):
    return pd.DataFrame({"timestamp": np.arange(n) * 300,
        "aggregate_cpu_utilization": np.arange(n, dtype=float),
        "active_vm_count": np.full(n, 20), "observed_arrivals": np.zeros(n),
        "observed_departures": np.zeros(n), "segment_cpu_utilization::A": np.arange(n, dtype=float) * 2,
        "segment_vm_share::A": np.ones(n)})


def test_lags_rolling_statistics_slopes_and_activity_are_origin_aligned():
    row = make_features(series(), 20)
    assert row.lag_1 == 20
    assert row.lag_12 == 9
    assert row.rolling_mean_3 == 19
    assert row.rolling_std_3 == 1
    assert row.slope_6 == pytest.approx(1)
    assert row.slope_12 == pytest.approx(1)
    assert row.observed_count_change == 0
    assert row.observed_arrivals_sum_6 == 0
    assert row["segment_cpu_utilization::A"] == 40


@pytest.mark.parametrize("origin", [11, 23, 47])
def test_future_poisoning_cannot_change_features_or_their_schema(origin):
    raw = series()
    raw["segment_cpu_utilization::future_only"] = np.nan
    first = make_features(raw, origin)
    for column in raw.columns.difference(["timestamp"]):
        raw.loc[origin + 1:, column] = 1e6
    pd.testing.assert_series_equal(first, make_features(raw, origin))
    assert "segment_cpu_utilization::future_only" not in first.index


def test_forward_fill_is_past_only_and_does_not_modify_target_labels():
    raw = series()
    raw.loc[14, "aggregate_cpu_utilization"] = np.nan
    raw.loc[15, "aggregate_cpu_utilization"] = 1e6
    row = make_features(raw, 14)
    assert row.lag_1 == 13
    assert row.lag_2 == 13
    assert pd.isna(raw.loc[14, "aggregate_cpu_utilization"])


def test_forbidden_lifetime_columns_never_enter_features():
    raw = series()
    expected = make_features(raw, 20)
    for name in ["avg_cpu", "max_cpu", "p95_max_cpu", "timestamp_vm_created", "timestamp_vm_deleted"]:
        raw[name] = 1e9
    pd.testing.assert_series_equal(expected, make_features(raw, 20))
    assert feature_frame(raw, 20).iloc[:11].isna().all().all()


@pytest.mark.parametrize("origin", [-1, 60, True, 1.5])
def test_invalid_feature_origins(origin):
    with pytest.raises(ValueError, match="origin"):
        make_features(series(), origin)
