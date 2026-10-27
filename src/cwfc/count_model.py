"""Holt damped-trend forecasting of the observed VM-count time series."""

from numbers import Integral
import warnings

import numpy as np
from statsmodels.tsa.holtwinters import Holt

COUNT_MODEL = "holt_damped_count"
COUNT_SETTINGS = {"smoothing_level": 0.3, "smoothing_trend": 0.1, "damping_trend": 0.98}


def count_forecast(history, horizon, settings=None):
    """Fit level/trend states on past counts and extrapolate a damped trend.

    Smoothing coefficients are fixed in the experiment recipe; the level and
    trend states are fitted again at each origin. Normalize for numerical
    stability, then transform back to VM units. There is no constant-history
    shortcut, persistence substitution, future-lifetime input or hardcoded count.
    """
    if isinstance(horizon, bool) or not isinstance(horizon, Integral) or horizon < 1:
        raise ValueError("Count forecast horizon must be a positive integer")
    settings = COUNT_SETTINGS.copy() if settings is None else dict(settings)
    if set(settings) != set(COUNT_SETTINGS):
        raise ValueError("Expected level/trend smoothing and trend damping settings")
    if not all(np.isfinite(v) and 0 < v < 1 for v in settings.values()):
        raise ValueError("Count smoothing/damping settings must lie strictly between zero and one")
    values = history.active_vm_count.to_numpy(dtype=float)
    if len(values) < 3 or not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("Holt count forecasting needs at least three finite nonnegative counts")
    offset, scale = float(values[0]), max(float(values.std()), 1.)
    normalized = (values - offset) / scale
    model = Holt(normalized, damped_trend=True, initialization_method="known",
                 initial_level=0., initial_trend=0.)
    with warnings.catch_warnings():
        # Zero residuals make unused AIC/BIC logarithms undefined. The fitted
        # states and forecasts remain well-defined; no optimization is run.
        warnings.filterwarnings("ignore", message="divide by zero encountered in log", category=RuntimeWarning)
        warnings.filterwarnings("ignore", message="invalid value encountered in scalar add", category=RuntimeWarning)
        fitted = model.fit(**settings, optimized=False, remove_bias=False)
        predictions = np.maximum(offset + scale * np.asarray(fitted.forecast(horizon)), 0.)
    if predictions.shape != (horizon,) or not np.isfinite(predictions).all():
        raise ValueError("Holt count model returned invalid predictions")
    info = {"model": COUNT_MODEL, "training_timestamps": len(values),
            "smoothing_settings": settings, "coefficients_optimized": False,
            "initial_level": offset, "initial_trend": 0.,
            "fitted_last_level": float(offset + scale * fitted.level[-1]),
            "fitted_last_trend": float(scale * fitted.trend[-1]),
            "training_sse": float(fitted.sse * scale ** 2),
            "constant_training_count": bool(np.ptp(values) == 0),
            "holdout_labels_used": False}
    return predictions, info
