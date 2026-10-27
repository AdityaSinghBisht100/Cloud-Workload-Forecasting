from __future__ import annotations

from numbers import Integral

import numpy as np
import pandas as pd


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


def _history(train, target, minimum=1):
    values = train[target].to_numpy(dtype=float)
    if len(values) < minimum:
        raise ValueError(f"{target}: need at least {minimum} training observations")
    if not np.isfinite(values).all():
        raise ValueError(f"{target}: training values must be finite; missing targets are not imputed")
    return values


def persistence_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str) -> pd.Series:
    last_value = float(_history(train, target)[-1])
    return pd.Series(last_value, index=template.index, dtype=float)


def seasonal_naive_forecast(train: pd.DataFrame, template: pd.DataFrame, target: str, season_length: int) -> pd.Series:
    _positive_integer(season_length, "season_length")
    values = _history(train, target, minimum=season_length)
    forecast = [float(values[-season_length + (i % season_length)]) for i in range(len(template))]
    return pd.Series(forecast, index=template.index, dtype=float)


def moving_average_forecast(train, template, target, window):
    """Repeat the last k observed values' mean across the forecast horizon."""
    _positive_integer(window, "window")
    values = _history(train, target, minimum=window)
    return pd.Series(float(values[-window:].mean()), index=template.index, dtype=float)


def drift_forecast(train, template, target):
    """Last value + h * (last - first)/(n-1), the mean first-difference slope."""
    values = _history(train, target, minimum=2)
    slope = (values[-1] - values[0]) / (len(values) - 1)
    return pd.Series(values[-1] + np.arange(1, len(template) + 1) * slope,
                     index=template.index, dtype=float)




def baseline_forecasts(train, template, target):
    """Fixed candidate set; seasonal length 24 is a two-hour negative control."""
    forecasts = {"persistence": persistence_forecast(train, template, target)}
    for window in (3, 6, 12, 24):
        forecasts[f"moving_average_{window}"] = moving_average_forecast(train, template, target, window)
    forecasts["drift"] = drift_forecast(train, template, target)
    forecasts["seasonal_naive_24"] = seasonal_naive_forecast(train, template, target, 24)
    return forecasts
