"""LightGBM VM CPU, busy-count and two-hour activity-risk experiment.

python -m cwfc.operational --config configs/operational.yaml
All selection and calibration use the public training period only.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from cwfc.io import load_metadata, load_readings
from cwfc.predict import atomic_write, dataset_fingerprints, prediction_bytes, sha256_file, validate_predictions
from cwfc.targets import build_targets, split_train_holdout
from cwfc.validation import rolling_origin_index
from cwfc.model import vm_forecast
from cwfc.direct import (BinaryModel, Panel, VARIANTS, anchor_features, behavior_labels,
    direct_examples, direct_predict, event_examples, fit_binary, fit_calibration, fit_cpu,
    low_use_mask, make_panel, variant_columns)
from cwfc.operational_evaluation import (candidate_policies, empirical_margin, error_breakdowns,
    evaluate_candidates, headroom_backtest, probability_scores, regression_scores, vm_error_tables)

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[2]
POOL_COLUMNS = ["label_timestamp", "raw_probability", "actual"]


def load_recipe(path):
    recipe = yaml.safe_load(Path(path).read_text(encoding="utf-8-sig"))
    if recipe.get("schema_version") != 1 or recipe["split"]["horizon"] != 24:
        raise ValueError("Operational recipe requires schema version 1 and horizon 24")
    if recipe["validation"]["horizon"] != 24 or recipe["validation"]["min_train"] < 36:
        raise ValueError("Validation requires horizon 24 and at least 36 training slots")
    if set(recipe["direct"]["variants"]) != set(VARIANTS):
        raise ValueError("Run all four specified direct ablations")
    if recipe["direct"]["anchor_stride"] < 1:
        raise ValueError("Anchor stride must be positive")
    if recipe["cpu_parameters"]["objective"] != "regression" or recipe["classifier_parameters"]["objective"] != "binary":
        raise ValueError("Expected squared-error CPU regression and binary activity classification")
    activity = recipe["activity"]
    if activity["category_feature"] or activity["screen_window"] != 12:
        raise ValueError("Activity models exclude category and use the specified 12-slot screen")
    if not 0 < activity["busy_threshold"] < activity["peak_threshold"] <= 100:
        raise ValueError("Activity thresholds must satisfy 0 < average < peak <= 100")
    if not 0 <= activity["final_risk_budget"] <= activity["hysteresis_exit_risk"] <= 1:
        raise ValueError("Risk entry/exit thresholds must be ordered probabilities")
    if any(not 0 <= p <= 1 for p in activity["risk_budgets"]):
        raise ValueError("Risk budgets must be probabilities")
    if recipe["selection"]["count_model"] != "persistence":
        raise ValueError("Observed count uses persistence in this experiment")
    if recipe["selection"]["metric"] != "pooled_cv_mae":
        raise ValueError("Select CPU by training-only pooled CV MAE")
    return recipe


def _live_panel(panel):
    keep = np.isfinite(panel.avg[:, -1])
    if not keep.any():
        raise ValueError("No VMs observed at forecast origin")
    return Panel(panel.ids[keep], panel.timestamps, panel.avg[keep], panel.minimum[keep],
                 panel.maximum[keep], panel.filled[keep], panel.static[keep]), keep


def fit_forecasts(readings, metadata, timestamps, recipe, model_directory=None):
    """Fit on bounded readings; returns VM predictions without accessing future labels."""
    settings, horizon = recipe["activity"], recipe["split"]["horizon"]
    panel = make_panel(readings, metadata, timestamps)
    live, keep = _live_panel(panel)
    x, residual, busy, weights, names, diagnostic = direct_examples(
        panel, horizon, recipe["direct"]["anchor_stride"], settings)
    predictions, importances = {}, []
    if model_directory is not None:
        model_directory.mkdir(parents=True, exist_ok=True)
    for variant in recipe["direct"]["variants"]:
        LOGGER.info("Origin %d: fit %s (%s matured examples)", len(timestamps), variant, len(residual))
        model, columns = fit_cpu(x, residual, weights, names, variant, recipe["cpu_parameters"])
        predictions[variant] = direct_predict(model, columns, live, horizon, settings)
        importances.append(pd.DataFrame({"model": variant, "feature": [names[i] for i in columns],
            "gain": model.booster_.feature_importance(importance_type="gain")}))
        if model_directory is not None:
            model.booster_.save_model(str(model_directory / f"{variant}.txt"))
    activity_columns = variant_columns(names, "direct_no_category")
    classifier = fit_binary(x, busy, names, recipe["classifier_parameters"], weights, activity_columns)
    query, _, _ = anchor_features(live, len(timestamps)-1, settings["busy_threshold"])
    raw_busy = np.column_stack([classifier.predict(np.column_stack(
        [query, np.full(len(query), h, dtype="float32")])) for h in range(1, horizon+1)])
    if model_directory is not None and classifier.model is not None:
        classifier.model.booster_.save_model(str(model_directory / "busy_classifier.txt"))
    diagnostic["busy_constant_probability"] = classifier.constant
    del x, residual, busy, weights, model, classifier

    event_x, event_y, event_names, event_info = event_examples(panel, horizon, settings)
    if event_x is None:
        event_model = BinaryModel(None, 1., [])
        diagnostic["event_fallback"] = "no matured candidate events: risk=1"
    else:
        event_columns = variant_columns(event_names, "direct_no_category")
        event_model = fit_binary(event_x, event_y, event_names, recipe["classifier_parameters"], columns=event_columns)
        diagnostic["event_fallback"] = "none" if event_model.constant is None else "smoothed one-class rate"
    candidate = low_use_mask(live, len(timestamps)-1, settings)
    raw_event = np.full(len(live.ids), np.nan)
    raw_event[candidate] = event_model.predict(query[candidate]) if candidate.any() else []
    if model_directory is not None and event_model.model is not None:
        event_model.model.booster_.save_model(str(model_directory / "event_classifier.txt"))
    diagnostic["event_constant_probability"] = event_model.constant
    del event_x, event_y, event_model

    n = len(live.ids)
    for name, values in [("vm_persistence", live.avg[:, -1]),
                         ("vm_moving_average_3", np.nanmean(live.avg[:, -3:], axis=1))]:
        predictions[name] = np.repeat(values[:, None], horizon, axis=1)
    _, recursive, recursive_info, _ = vm_forecast(readings, metadata, timestamps, horizon, recipe["cpu_parameters"])
    predictions["global_vm_lightgbm"] = recursive.pivot(index="vm_id", columns="step", values="yhat").reindex(live.ids).to_numpy()
    for values in predictions.values():
        if values.shape != (n, horizon) or not np.isfinite(values).all() or ((values < 0) | (values > 100)).any():
            raise ValueError("CPU model returned invalid per-VM forecasts")
    diagnostic.update(event_info)
    diagnostic.update({"training_timestamps": len(timestamps), "vm_histories": len(panel.ids),
        "forecast_vm_count": n, "last_observed_timestamp": int(panel.timestamps[-1]),
        "recursive_training_samples": recursive_info["training_samples"],
        "holdout_labels_used": False})
    return live, predictions, raw_busy, raw_event, candidate, diagnostic, pd.concat(importances, ignore_index=True)


def observed_validation(readings, ids, timestamps):
    """Attach actuals only after issuing forecasts; never fill missing labels."""
    from cwfc.quality import prepare_readings
    future = prepare_readings(readings.loc[readings.timestamp.isin(timestamps)])
    future["vm_id"] = future.vm_id.astype(str)
    arrays = [future.pivot(index="vm_id", columns="timestamp", values=name).reindex(
        index=ids, columns=timestamps).to_numpy(dtype=float) for name in ("avg_cpu", "max_cpu")]
    return arrays


def _event_actual(avg, peak, settings):
    complete = np.isfinite(avg).all(axis=1) & np.isfinite(peak).all(axis=1)
    event = np.full(len(avg), np.nan)
    event[complete] = ((avg[complete] >= settings["busy_threshold"])
                       | (peak[complete] >= settings["peak_threshold"])).any(axis=1)
    return event


def _pool_frame(timestamps, probability, actual):
    return pd.DataFrame({"label_timestamp": np.broadcast_to(timestamps, probability.shape).ravel(),
                         "raw_probability": probability.ravel(), "actual": actual.ravel()})


def _write_gzip(frame, path):
    frame.to_csv(path, index=False, compression={"method": "gzip", "mtime": 0})


def _json(path, record):
    # pandas serialization maps missing diagnostics to JSON null, not invalid NaN tokens.
    atomic_write(path, pd.Series(record).to_json(indent=2, default_handler=str).encode("utf-8"))


def _calibration_record(calibrator):
    return {"samples": calibrator.samples, "latest_label_timestamp": calibrator.latest_timestamp,
            "method": "raw_warmup" if calibrator.model is None else "Platt_on_matured_earlier_origin_predictions",
            "coefficient": None if calibrator.model is None else calibrator.model.coef_.tolist(),
            "intercept": None if calibrator.model is None else calibrator.model.intercept_.tolist()}


def backtest(readings, metadata, train, recipe, directory):
    directory = Path(directory)
    (directory / "folds").mkdir(parents=True, exist_ok=True)
    horizon, settings = recipe["split"]["horizon"], recipe["activity"]
    cpu_rows, count_rows, vm_tables, behavior_tables, fit_rows = [], [], [], [], []
    probability_rows, reliability_rows, policy_rows, event_tables = [], [], [], []
    busy_pool = pd.DataFrame({c: pd.Series(dtype="float64") for c in POOL_COLUMNS})
    event_pool = busy_pool.copy()
    previous_policies = {}
    for fold, split in enumerate(rolling_origin_index(len(train), **recipe["validation"]), 1):
        history, validation = train.iloc[:split.origin], train.iloc[split.origin:split.validation_end]
        panel, forecasts, raw_busy, raw_event, candidate, info, _ = fit_forecasts(
            readings, metadata, history.timestamp, recipe)
        cutoff = int(history.timestamp.iloc[-1])
        busy_cal = fit_calibration(busy_pool, cutoff)
        event_cal = fit_calibration(event_pool, cutoff)
        calibrated_busy = busy_cal.predict(raw_busy).reshape(raw_busy.shape)
        calibrated_event = np.full(len(panel.ids), np.nan)
        calibrated_event[candidate] = event_cal.predict(raw_event[candidate])
        actual, peak = observed_validation(readings, panel.ids, validation.timestamp)
        event = _event_actual(actual, peak, settings)
        actual_busy = np.where(np.isfinite(actual), actual >= settings["busy_threshold"], np.nan)
        groups = behavior_labels(panel, split.origin-1, settings)
        info.update({"fold": fold, "origin": split.origin,
                     "busy_calibration": _calibration_record(busy_cal), "event_calibration": _calibration_record(event_cal)})
        fit_rows.append(info)
        vm, behavior = vm_error_tables(panel, actual, forecasts, groups, fold, split.origin)
        vm_tables.append(vm)
        behavior_tables.append(behavior)
        detail = pd.DataFrame({"fold": fold, "origin": split.origin,
            "vm_id": np.repeat(panel.ids, horizon), "step": np.tile(np.arange(1, horizon+1), len(panel.ids)),
            "timestamp": np.tile(validation.timestamp.to_numpy(), len(panel.ids)),
            "behavior": np.repeat(groups, horizon), "y": actual.ravel(), "max_cpu_actual": peak.ravel(),
            **{name: values.ravel() for name, values in forecasts.items()},
            "busy_probability_raw": raw_busy.ravel(), "busy_probability": calibrated_busy.ravel()})
        _write_gzip(detail, directory / "folds" / f"fold_{fold:02d}_vm_predictions.csv.gz")
        del detail
        for model, values in forecasts.items():
            for step, (timestamp, y, yhat) in enumerate(zip(validation.timestamp, validation.aggregate_cpu_utilization, values.mean(axis=0)), 1):
                cpu_rows.append({"fold": fold, "origin": split.origin, "step": step, "model": model,
                                 "timestamp": int(timestamp), "y": float(y), "yhat": float(yhat)})
        from cwfc.quality import prepare_readings
        valid_future = prepare_readings(readings.loc[readings.timestamp.isin(validation.timestamp)])
        for threshold in settings["sensitivity_thresholds"]:
            # All observed VMs, not just the modeled cutoff cohort, define the target.
            actual_counts = valid_future.assign(busy=valid_future.avg_cpu >= threshold).groupby("timestamp").busy.sum().reindex(validation.timestamp, fill_value=0).to_numpy()
            recent_counts = (panel.avg[:, -3:] >= threshold).sum(axis=0)
            candidates = {"busy_count_persistence": np.full(horizon, recent_counts[-1]),
                          "busy_count_ma3": np.full(horizon, recent_counts.mean()),
                          "cpu_threshold_count": (forecasts["direct_history"] >= threshold).sum(axis=0)}
            if threshold == settings["busy_threshold"]:
                candidates["busy_probability_model"] = calibrated_busy.sum(axis=0)
            for model, values in candidates.items():
                for step, (timestamp, y, yhat) in enumerate(zip(validation.timestamp, actual_counts, values), 1):
                    count_rows.append({"fold": fold, "origin": split.origin, "step": step,
                        "model": model, "threshold": threshold, "timestamp": int(timestamp), "y": y, "yhat": yhat})
        for target, y, raw, calibrated, calibrator in [
                ("busy_at_horizon", actual_busy.ravel(), raw_busy.ravel(), calibrated_busy.ravel(), busy_cal),
                ("two_hour_activity_given_screen", event[candidate], raw_event[candidate], calibrated_event[candidate], event_cal)]:
            for p, status in [(raw, False), (calibrated, calibrator.model is not None)]:
                scores, reliability = probability_scores(y, p, split.origin, target, status)
                # Preserve raw and served streams separately, including raw warmup served predictions.
                stream = "raw" if p is raw else "served"
                probability_rows.extend([{**row, "stream": stream} for row in scores])
                reliability_rows.extend([{**row, "stream": stream} for row in reliability])
        masks = candidate_policies(calibrated_event, candidate, panel.avg[:, -12:].mean(axis=1),
            panel.maximum[:, -12:].max(axis=1), panel.ids, settings, previous_policies)
        policy_rows.extend(evaluate_candidates(masks, event, panel.ids, split.origin, horizon, previous_policies))
        event_table = pd.DataFrame({"fold": fold, "origin": split.origin, "vm_id": panel.ids,
            "behavior": groups, "screen_candidate": candidate, "raw_probability": raw_event,
            "future_activity_risk": calibrated_event, "actual": event,
            "label_timestamp": int(validation.timestamp.iloc[-1]),
            **{name: mask for name, mask in masks.items()}})
        event_tables.append(event_table)
        busy_pool = pd.concat([busy_pool, _pool_frame(validation.timestamp.to_numpy(), raw_busy, actual_busy)], ignore_index=True)
        event_pool = pd.concat([event_pool, event_table.loc[candidate, POOL_COLUMNS]], ignore_index=True)
        LOGGER.info("Completed fold %d: CPU models=%d, screen candidates=%d", fold, len(forecasts), candidate.sum())
    cpu, counts = pd.DataFrame(cpu_rows), pd.DataFrame(count_rows)
    pd.concat(vm_tables, ignore_index=True).to_csv(directory / "vm_errors.csv.gz", index=False, compression="gzip")
    pd.concat(behavior_tables, ignore_index=True).to_csv(directory / "behavior_errors.csv", index=False)
    pd.concat(event_tables, ignore_index=True).to_csv(directory / "candidate_decisions.csv.gz", index=False, compression="gzip")
    for name, frame in [("cpu_backtest", cpu), ("busy_count_backtest", counts),
                        ("probability_scores", pd.DataFrame(probability_rows)),
                        ("probability_reliability", pd.DataFrame(reliability_rows)),
                        ("candidate_policy", pd.DataFrame(policy_rows))]:
        frame.to_csv(directory / f"{name}.csv", index=False)
    _json(directory / "fit_diagnostics.json", {"folds": fit_rows})
    return cpu, counts, busy_pool, event_pool


def operational_predictions(panel, forecasts, selected, busy_probability, risk, candidate,
                            future_timestamps, cpu_backtest, recipe):
    horizon = len(future_timestamps)
    settings = recipe["activity"]
    selected_cpu = forecasts[selected]
    cutoff = int(panel.timestamps[-1])
    margins = [empirical_margin(cpu_backtest, selected, h, cutoff, recipe["policy"]) for h in range(1, horizon+1)]
    low_risk = candidate & (risk <= settings["final_risk_budget"])
    result = pd.DataFrame({"timestamp": future_timestamps, "active_vm_count": len(panel.ids),
        "aggregate_cpu_utilization": selected_cpu.mean(axis=0),
        "direct_history_cpu_utilization": forecasts["direct_history"].mean(axis=0),
        "expected_cpu_busy_count": busy_probability.sum(axis=0),
        "predicted_cpu_threshold_count": (selected_cpu >= settings["busy_threshold"]).sum(axis=0),
        "busy_cpu_threshold_pct": settings["busy_threshold"],
        "low_use_candidate_count_at_origin": int(candidate.sum()),
        "low_risk_review_candidate_count_at_origin": int(low_risk.sum()),
        "two_hour_mean_candidate_activity_risk": float(risk[candidate].mean()) if candidate.any() else 0.,
        "two_hour_expected_candidate_activity_events": float(risk[candidate].sum()),
        "risk_budget": settings["final_risk_budget"],
        "cpu_upper_bound_proxy_pct": np.minimum(100., selected_cpu.mean(axis=0)+[m[0] for m in margins]),
        "upper_margin_calibration_cases": [m[1] for m in margins]})
    for threshold in settings["sensitivity_thresholds"]:
        result[f"predicted_cpu_count_ge_{threshold:g}pct"] = (selected_cpu >= threshold).sum(axis=0)
    if len(result) != 24 or not np.isfinite(result.to_numpy(dtype=float)).all():
        raise ValueError("Operational CSV must contain 24 finite numeric forecast rows")
    if not ((busy_probability >= 0) & (busy_probability <= 1)).all():
        raise ValueError("Busy probabilities must be in [0,1]")
    if not ((risk[candidate] >= 0) & (risk[candidate] <= 1)).all():
        raise ValueError("Candidate event probabilities must be in [0,1]")
    return result


def run(data="challenge_data", config="configs/operational.yaml", directory="artifacts/operational",
        output="artifacts/operational/cpu_forecast.csv", activity_output="artifacts/operational/activity_forecast.csv"):
    recipe = load_recipe(config)
    if Path(output).resolve() == Path(activity_output).resolve():
        raise ValueError("CPU and activity outputs require different paths")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fingerprints = dataset_fingerprints(data)
    readings, metadata = load_readings(data), load_metadata(data)
    full = build_targets(readings, metadata)
    train, holdout = split_train_holdout(full, **recipe["split"])
    # Holdout values never enter fitting, selection, calibration, or scoring below.
    past = readings.loc[readings.timestamp.isin(train.timestamp)].copy()
    del readings, full
    cpu, counts, busy_pool, event_pool = backtest(past, metadata, train, recipe, directory)
    scores = regression_scores(cpu)
    scores["skill_vs_vm_persistence"] = 1-scores.mae/float(scores.loc[scores.model == "vm_persistence", "mae"].iloc[0])
    scores.to_csv(directory / "cpu_scores.csv", index=False)
    error_breakdowns(cpu, train.aggregate_cpu_utilization.quantile(.8)).to_csv(directory / "error_breakdowns.csv", index=False)
    count_scores = pd.concat([regression_scores(group).assign(threshold=threshold)
        for threshold, group in counts.groupby("threshold")], ignore_index=True)
    count_scores.to_csv(directory / "busy_count_scores.csv", index=False)
    selected = scores.loc[scores.model.isin(recipe["selection"]["eligible_models"])].iloc[0].model
    frozen = {"selected_cpu_model": selected, "selection_metric": "pooled_cv_mae",
              "tie_break": ["rmse", "model"], "configuration_sha256": sha256_file(config),
              "training_end_timestamp": int(train.timestamp.iloc[-1]),
              "cpu_scores": json.loads(scores.to_json(orient="records")),
              "holdout_used_for_selection": False, "original_holdout_previously_inspected": True}
    _json(directory / "selection.json", frozen)
    headroom = headroom_backtest(cpu, recipe["policy"])
    headroom.to_csv(directory / "headroom_backtest.csv", index=False)
    headroom.assign(under=lambda f: f.underforecast.gt(0)).groupby(["model", "policy"]).agg(
        cases=("under", "size"), underforecast_fraction=("under", "mean"),
        mean_shortfall_pp=("underforecast", "mean"), mean_excess_headroom_pp=("excess_headroom_pp", "mean"),
        mean_cost_proxy=("cost_proxy", "mean")).reset_index().to_csv(directory / "headroom_scores.csv", index=False)
    panel, forecasts, raw_busy, raw_event, candidate, fit, importance = fit_forecasts(
        past, metadata, train.timestamp, recipe, directory / "models")
    busy_cal = fit_calibration(busy_pool, int(train.timestamp.iloc[-1]))
    event_cal = fit_calibration(event_pool, int(train.timestamp.iloc[-1]))
    busy = busy_cal.predict(raw_busy).reshape(raw_busy.shape)
    event = np.full(len(panel.ids), np.nan)
    event[candidate] = event_cal.predict(raw_event[candidate])
    operational = operational_predictions(panel, forecasts, selected, busy, event, candidate,
                                         holdout.timestamp.to_numpy(dtype="int64"), cpu, recipe)
    prediction = operational[["timestamp", "active_vm_count", "aggregate_cpu_utilization"]]
    validate_predictions(prediction, holdout.timestamp)
    atomic_write(output, prediction_bytes(prediction))
    atomic_write(activity_output, prediction_bytes(operational))
    validate_predictions(pd.read_csv(output), holdout.timestamp)
    pd.testing.assert_frame_equal(pd.read_csv(activity_output), operational, check_dtype=False, check_exact=False)
    importance.to_csv(directory / "feature_importance.csv", index=False)
    shortlist = pd.DataFrame({"vm_id": panel.ids[candidate], "forecast_origin_timestamp": int(train.timestamp.iloc[-1]),
        "event_window_end_timestamp": int(holdout.timestamp.iloc[-1]),
        "last_hour_mean_cpu": panel.avg[candidate, -12:].mean(axis=1),
        "last_hour_max_cpu": panel.maximum[candidate, -12:].max(axis=1),
        "future_activity_risk_raw": raw_event[candidate], "future_activity_risk": event[candidate],
        "review_candidate": event[candidate] <= recipe["activity"]["final_risk_budget"]})
    shortlist.sort_values(["future_activity_risk", "vm_id"]).to_csv(directory / "low_use_review_candidates.csv", index=False)
    detail = pd.DataFrame({"vm_id": np.repeat(panel.ids, 24), "step": np.tile(np.arange(1, 25), len(panel.ids)),
        "timestamp": np.tile(holdout.timestamp.to_numpy(), len(panel.ids)),
        "predicted_cpu": forecasts[selected].ravel(), "busy_probability": busy.ravel()})
    _write_gzip(detail, directory / "vm_predictions.csv.gz")
    threshold_history = pd.DataFrame({"timestamp": train.timestamp})
    prepared_panel = make_panel(past, metadata, train.timestamp)
    for threshold in recipe["activity"]["sensitivity_thresholds"]:
        threshold_history[f"cpu_busy_count_ge_{threshold:g}pct"] = (prepared_panel.avg >= threshold).sum(axis=0)
    threshold_history.to_csv(directory / "activity_history.csv", index=False)
    atomic_write(directory / "recipe.yaml", Path(config).read_bytes())
    record = {**frozen, "data_sha256": fingerprints, "fit": fit,
        "busy_calibration": _calibration_record(busy_cal), "event_calibration": _calibration_record(event_cal),
        "forecast_rows": len(prediction), "forecast_vm_count": len(panel.ids),
        "screen_candidate_vms": int(candidate.sum()), "review_candidate_vms": int(shortlist.review_candidate.sum()),
        "cpu_output_sha256": sha256_file(output), "activity_output_sha256": sha256_file(activity_output),
        "count_model": "observed_count_persistence", "holdout_scored": False,
        "uncertainty": "one-sided empirical residual margin; no conformal coverage guarantee",
        "source_sha256": {f"src/cwfc/{name}.py": sha256_file(Path(__file__).with_name(f"{name}.py"))
                          for name in ["direct", "operational", "operational_evaluation", "model"]}}
    _json(directory / "manifest.json", record)
    LOGGER.info("Selected %s; exported %s and %s", selected, output, activity_output)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="challenge_data")
    parser.add_argument("--config", default="configs/operational.yaml")
    parser.add_argument("--directory", default="artifacts/operational")
    parser.add_argument("--output", default="artifacts/operational/cpu_forecast.csv")
    parser.add_argument("--activity-output", default="artifacts/operational/activity_forecast.csv")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    record = run(**vars(args))
    print(json.dumps({k: record[k] for k in ["selected_cpu_model", "forecast_rows", "screen_candidate_vms", "review_candidate_vms"]}, indent=2))


if __name__ == "__main__":
    main()
