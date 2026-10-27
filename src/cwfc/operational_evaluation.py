"""Training-only error summaries and explicitly assumed operational policies."""

import numpy as np
import pandas as pd


def regression_scores(frame):
    rows = []
    for model, group in frame.groupby("model", sort=True):
        e = group.yhat.to_numpy()-group.y.to_numpy()
        rows.append({"model": model, "forecast_cases": len(e), "mae": np.abs(e).mean(),
                     "rmse": np.sqrt(np.mean(e**2)), "bias": e.mean(),
                     "smape_pct": np.mean(200*np.abs(e)/np.maximum(np.abs(group.y)+np.abs(group.yhat), 1e-8)),
                     "underforecast_fraction": (e < 0).mean()})
    return pd.DataFrame(rows).sort_values(["mae", "rmse", "model"]).reset_index(drop=True)


def error_breakdowns(frame, high_threshold):
    data = frame.assign(error=frame.yhat-frame.y)
    data["regime"] = np.where(data.y > high_threshold, "high_training_Q80", "other")
    rows = []
    for dimension in ["step", "fold", "regime"]:
        for (value, model), group in data.groupby([dimension, "model"], sort=True):
            e = group.error.to_numpy()
            rows.append({"dimension": dimension, "value": str(value), "model": model,
                         "cases": len(e), "mae": np.abs(e).mean(), "rmse": np.sqrt(np.mean(e**2)),
                         "bias": e.mean(), "underforecast_fraction": (e < 0).mean()})
    return pd.DataFrame(rows)


def vm_error_tables(panel, actual, predictions, groups, fold, origin):
    """Individual-VM and behavior-group errors; missing labels stay unscored."""
    vm_rows, group_rows = [], []
    for model, prediction in predictions.items():
        error = prediction-actual
        finite = np.isfinite(error)
        counts = finite.sum(axis=1)
        absolute = np.where(finite, np.abs(error), 0)
        square = np.where(finite, error**2, 0)
        signed = np.where(finite, error, 0)
        vm_rows.append(pd.DataFrame({"fold": fold, "origin": origin, "model": model,
            "vm_id": panel.ids, "behavior": groups, "observed_labels": counts,
            "mae": np.divide(absolute.sum(axis=1), counts, out=np.full(len(counts), np.nan), where=counts > 0),
            "rmse": np.sqrt(np.divide(square.sum(axis=1), counts, out=np.full(len(counts), np.nan), where=counts > 0)),
            "bias": np.divide(signed.sum(axis=1), counts, out=np.full(len(counts), np.nan), where=counts > 0)}))
        for label in sorted(set(groups)):
            values = error[groups == label]
            values = values[np.isfinite(values)]
            if len(values):
                group_rows.append({"fold": fold, "origin": origin, "model": model, "behavior": label,
                    "cases": len(values), "mae": np.abs(values).mean(), "rmse": np.sqrt(np.mean(values**2)),
                    "bias": values.mean(), "underforecast_fraction": (values < 0).mean()})
    return pd.concat(vm_rows, ignore_index=True), pd.DataFrame(group_rows)


def probability_scores(actual, probability, origin, target, calibrated):
    valid = np.isfinite(actual) & np.isfinite(probability)
    y, p = actual[valid], probability[valid]
    if not len(y):
        return [], []
    p = np.clip(p, 1e-8, 1-1e-8)
    score = {"origin": origin, "target": target, "calibrated": calibrated, "cases": len(y),
             "brier_score": np.mean((p-y)**2),
             "log_loss": -np.mean(y*np.log(p)+(1-y)*np.log(1-p)),
             "observed_event_fraction": y.mean(), "mean_probability": p.mean()}
    bins = np.minimum((p*10).astype(int), 9)
    reliability = [{"origin": origin, "target": target, "calibrated": calibrated,
        "probability_bin": int(b), "cases": int((bins == b).sum()),
        "mean_probability": p[bins == b].mean(), "observed_event_fraction": y[bins == b].mean()}
        for b in np.unique(bins)]
    return [score], reliability


def candidate_policies(risk, candidate, last_avg, last_peak, ids, settings, previous=None):
    """Fixed entry/exit thresholds; matched-count baselines prevent trivial safety wins."""
    previous = {} if previous is None else previous
    masks = {"keep_all": np.zeros(len(ids), dtype=bool), "screen_only": candidate.copy()}
    for budget in settings["risk_budgets"]:
        name = f"risk_le_{budget:g}"
        chosen = candidate & (risk <= budget)
        masks[name] = chosen
        k = int(chosen.sum())
        eligible = np.flatnonzero(candidate)
        for score, prefix in [(last_avg, "lowest_average"), (last_peak, "lowest_peak")]:
            rank = eligible[np.lexsort((ids[eligible], score[eligible]))]
            matched = np.zeros(len(ids), dtype=bool)
            matched[rank[:k]] = True
            masks[f"{prefix}_matched_{budget:g}"] = matched
    old = previous.get("risk_hysteresis", set())
    already = np.isin(ids, list(old))
    limit = np.where(already, settings["hysteresis_exit_risk"], settings["final_risk_budget"])
    masks["risk_hysteresis"] = candidate & (risk <= limit)
    return masks


def evaluate_candidates(masks, event, ids, origin, horizon, previous):
    rows = []
    valid = np.isfinite(event)
    for name, mask in masks.items():
        selected = mask & valid
        safe = valid & (event == 0)
        false = int((selected & (event == 1)).sum())
        count = int(selected.sum())
        current = set(ids[mask])
        switches = None if name not in previous else len(current.symmetric_difference(previous[name]))
        rows.append({"origin": origin, "policy": name, "candidate_decisions": count,
                     "future_activity_events": false, "stable_candidate_decisions": count-false,
                     "candidate_precision": None if not count else (count-false)/count,
                     "safe_opportunity_captured": None if not safe.sum() else (selected & safe).sum()/safe.sum(),
                     "candidate_vm_hours": count*horizon/12,
                     "switches_since_previous_origin": switches,
                     "scope": "CPU-threshold review opportunities, not verified shutdown savings"})
        previous[name] = current
    return rows


def empirical_margin(history, model, step, cutoff_timestamp, settings):
    """One-sided empirical upper margin; no exchangeability/conformal claim."""
    lo = ((step-1)//6)*6+1
    known = history.loc[(history.model == model) & (history.timestamp <= cutoff_timestamp)
                        & history.step.between(lo, lo+5)]
    if len(known) < settings["minimum_calibration_cases"]:
        return settings["initial_margin_pp"], len(known), "fixed_warmup_margin"
    residual = known.y-known.yhat
    margin = max(0., float(np.quantile(residual, settings["empirical_upper_quantile"], method="higher")))
    return margin, len(known), "empirical_matured_residuals"


def headroom_backtest(frame, settings):
    rows = []
    for origin in sorted(frame.origin.unique()):
        current = frame.loc[frame.origin == origin]
        # Restrict origins as well as labels: current-fold residuals are never calibration.
        previous = frame.loc[frame.origin < origin]
        cutoff = int(current.timestamp.min()-300)
        for row in current.itertuples(index=False):
            choices = [(f"fixed_{margin:g}pp", margin, 0, "fixed") for margin in settings["fixed_margins_pp"]]
            margin, samples, source = empirical_margin(previous, row.model, row.step, cutoff, settings)
            choices.append(("empirical_upper", margin, samples, source))
            for policy, extra, n, source in choices:
                capacity = min(100., row.yhat+extra)
                under, over = max(0., row.y-capacity), max(0., capacity-row.y)
                rows.append({"origin": origin, "step": row.step, "timestamp": row.timestamp,
                    "model": row.model, "policy": policy, "y": row.y, "capacity_proxy_pct": capacity,
                    "underforecast": under, "excess_headroom_pp": over,
                    "cost_proxy": settings["under_cost"]*under+settings["over_cost"]*over,
                    "calibration_samples": n, "margin_source": source})
    return pd.DataFrame(rows)
