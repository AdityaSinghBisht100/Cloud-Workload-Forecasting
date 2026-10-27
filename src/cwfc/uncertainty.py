"""Exact-step residual calibration with explicit finite-sample limitations.

These are split-conformal order statistics. Time dependence, overlapping CV
windows and reuse of model-selection evidence prevent an exchangeability-based
coverage guarantee here. Callers must supply only residuals observed by origin.
"""

import math

import numpy as np


def conformal_quantile(scores, level):
    """Return order ceil((n+1)*level), or infinity if unsupported by n.

    No interpolation, pooling across steps, or silent maximum-score fallback.
    """
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Calibration scores must be one-dimensional and finite")
    if not np.isfinite(level) or not 0 < level < 1:
        raise ValueError("level must be strictly between zero and one")
    rank = math.ceil((len(values)+1)*level)
    return float(np.sort(values)[rank-1]) if rank <= len(values) else float("inf")


def cpu_bounds(point, residuals, level):
    """One-sided signed upper bound and symmetric interval at the given level.

    residuals = observed - forecast at this exact horizon step. CPU's physical
    [0,100] support makes unsupported bounds [0,100] / upper=100 explicit.
    The symmetric interval's level differs from its upper-tail probability.
    """
    if not np.isfinite(point) or not 0 <= point <= 100:
        raise ValueError("Point forecast must be in [0,100]")
    scores = np.asarray(residuals, dtype=float)
    correction = conformal_quantile(scores, level)
    radius = conformal_quantile(np.abs(scores), level)
    return {"upper_cpu": float(np.clip(point+correction, 0, 100)),
            "lower_cpu": float(np.clip(point-radius, 0, 100)),
            "interval_upper_cpu": float(np.clip(point+radius, 0, 100)),
            "calibration_n": len(scores), "upper_supported": bool(np.isfinite(correction)),
            "interval_supported": bool(np.isfinite(radius))}
