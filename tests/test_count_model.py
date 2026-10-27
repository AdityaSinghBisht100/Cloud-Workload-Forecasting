from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.holtwinters import Holt

from cwfc.count_model import COUNT_MODEL, count_forecast


@pytest.mark.parametrize("count", [0, 8500, 10000])
def test_constant_counts_still_fit_holt_and_are_not_hardcoded(count):
    with patch("cwfc.count_model.Holt", wraps=Holt) as constructor:
        forecast, info = count_forecast(pd.DataFrame({"active_vm_count": [count] * 110}), 24)
    constructor.assert_called_once()
    assert np.array_equal(forecast, np.repeat(count, 24))
    assert info["model"] == COUNT_MODEL
    assert info["fitted_last_level"] == count
    assert info["fitted_last_trend"] == 0


def test_increasing_counts_learn_a_damped_trend():
    forecast, info = count_forecast(pd.DataFrame({"active_vm_count": np.arange(100, 210)}), 24)
    assert info["fitted_last_trend"] > 0
    assert np.all(np.diff(forecast) > 0)
    assert np.all(np.diff(np.diff(forecast)) < 0)
    assert forecast[-1] > 209


def test_future_fields_not_read_and_decreasing_counts_stay_nonnegative():
    history = pd.DataFrame({"active_vm_count": np.arange(30, 0, -1)})
    expected, _ = count_forecast(history, 24)
    history["future_count"] = np.inf
    history["timestamp_vm_deleted"] = -999
    actual, _ = count_forecast(history, 24)
    assert np.array_equal(expected, actual)
    assert np.all(actual >= 0)
    assert actual[0] > actual[-1]


@pytest.mark.parametrize("values", [[1, 2], [1, np.nan, 3], [1, -1, 3], [1, np.inf, 3]])
def test_invalid_history_not_imputed(values):
    with pytest.raises(ValueError, match="three finite nonnegative"):
        count_forecast(pd.DataFrame({"active_vm_count": values}), 24)
