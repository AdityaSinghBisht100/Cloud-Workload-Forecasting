import numpy as np
import pandas as pd
import pytest

from cwfc.baselines import (drift_forecast, moving_average_forecast,
                            persistence_forecast, seasonal_naive_forecast)


def test_forecast_values_and_template_index():
    train = pd.DataFrame({"cpu": [1., 3., 5., 7.]})
    template = pd.DataFrame(index=[20, 30, 40])
    assert persistence_forecast(train, template, "cpu").tolist() == [7., 7., 7.]
    assert moving_average_forecast(train, template, "cpu", 2).tolist() == [6., 6., 6.]
    assert drift_forecast(train, template, "cpu").tolist() == [9., 11., 13.]
    seasonal = seasonal_naive_forecast(train, template, "cpu", 2)
    assert seasonal.tolist() == [5., 7., 5.]
    assert seasonal.index.tolist() == template.index.tolist()


@pytest.mark.parametrize("window", [0, -1, 1.5, True, 5])
def test_invalid_or_insufficient_moving_average_history(window):
    with pytest.raises(ValueError):
        moving_average_forecast(pd.DataFrame({"cpu": [1., 2.]}), pd.DataFrame(index=[0]), "cpu", window)


def test_missing_targets_are_not_silently_filled_or_averaged_away():
    train = pd.DataFrame({"cpu": [1., np.nan, 2.]})
    with pytest.raises(ValueError, match="not imputed"):
        moving_average_forecast(train, pd.DataFrame(index=[0]), "cpu",3)
    with pytest.raises(ValueError):
        drift_forecast(train.iloc[:1], pd.DataFrame(index=[0]), "cpu")
