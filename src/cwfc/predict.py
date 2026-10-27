"""Fit the final recursive model and export 24 forecasts; never score holdout labels."""
import argparse
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import yaml
from cwfc.io import load_metadata, load_readings, verify_checksums
from cwfc.model import MODEL, FEATURES, vm_forecast
from cwfc.targets import build_targets, split_train_holdout

LOGGER = logging.getLogger(__name__)
OUTPUT_COLUMNS = ["timestamp", "active_vm_count", "aggregate_cpu_utilization"]

def sha256_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def dataset_fingerprints(data_dir):
    root = Path(data_dir)
    if (root / "SHA256SUMS").exists():
        hashes = verify_checksums(root)
    else:
        LOGGER.warning("No SHA256SUMS supplied; validating replacement-data schema and quality without release checksums")
        hashes = {}
    for name in ("vm_cpu_readings.csv.gz", "vm_metadata.csv.gz"):
        if name not in hashes:
            hashes[name] = sha256_file(root / name)
    if (root / "SHA256SUMS").exists():
        hashes["SHA256SUMS"] = sha256_file(root / "SHA256SUMS")
    return hashes


def validate_predictions(predictions, expected_timestamps):
    if list(predictions.columns) != OUTPUT_COLUMNS:
        raise ValueError(f"Prediction columns must be exactly {OUTPUT_COLUMNS}, without an index column")
    expected = np.asarray(expected_timestamps)
    if len(predictions) != 24 or len(expected) != 24:
        raise ValueError("Predictions and holdout timestamps must each have exactly 24 rows")
    for column in OUTPUT_COLUMNS:
        if not pd.api.types.is_numeric_dtype(predictions[column]) or pd.api.types.is_bool_dtype(predictions[column]):
            raise ValueError(f"Prediction column must be numeric: {column}")
    if not np.isfinite(predictions.to_numpy(dtype=float)).all():
        raise ValueError("Predictions must contain finite values without missing entries")
    if (predictions.timestamp % 1 != 0).any() or not predictions.timestamp.is_unique or not predictions.timestamp.is_monotonic_increasing:
        raise ValueError("Prediction timestamps must be sorted unique integer seconds")
    if not np.array_equal(predictions.timestamp.to_numpy(), expected):
        raise ValueError("Prediction timestamps must match the holdout exactly")
    if predictions.active_vm_count.lt(0).any() or not predictions.aggregate_cpu_utilization.between(0, 100).all():
        raise ValueError("Predicted count must be >=0 and CPU must be within [0,100]")


def prediction_bytes(predictions):
    return predictions.to_csv(index=False, float_format="%.17g", lineterminator="\n").encode("utf-8")


def atomic_write(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def load_config(path="configs/final_model.yaml"):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if config.get("model") != MODEL or config.get("features") != FEATURES:
        raise ValueError("Configuration must describe the final recursive model and its 25 features")
    if config.get("count_model") != "observed_count_persistence":
        raise ValueError("Expected observed-count persistence")
    if config["split"]["horizon"] != 24:
        raise ValueError("The final export requires horizon=24")
    return config

def run_prediction(data_dir="challenge_data", output="predictions/prediction.csv",
                   config_path="configs/final_model.yaml", directory="artifacts/final_model"):
    config = load_config(config_path)
    hashes = dataset_fingerprints(data_dir)
    readings, metadata = load_readings(data_dir), load_metadata(data_dir)
    # Select timestamps first: future CPU values do not enter preparation or fitting.
    from cwfc.quality import observation_grid
    grid = observation_grid(readings)
    available = min(config["split"]["n_train"], len(grid)-24)
    if available < 13:
        raise ValueError("Need at least 13 history slots plus 24 forecast timestamps")
    timestamps = grid[-(available+24):-24]
    future = grid[-24:]
    past = readings.loc[readings.timestamp.isin(timestamps)].copy()
    targets = build_targets(past, metadata)
    cpu, detail, fit, importance = vm_forecast(past, metadata, timestamps, 24, config["parameters"])
    predictions = pd.DataFrame({"timestamp": np.asarray(future,dtype="int64"),
        "active_vm_count": float(targets.active_vm_count.iloc[-1]), "aggregate_cpu_utilization": cpu})
    validate_predictions(predictions, future)
    payload = prediction_bytes(predictions)
    atomic_write(output, payload)
    directory = Path(directory)
    directory.mkdir(parents=True,exist_ok=True)
    detail = detail.rename(columns={"yhat":"predicted_cpu"})
    detail["timestamp"] = np.asarray(future)[detail.step.to_numpy()-1]
    detail.to_csv(directory/"individual_predictions.csv.gz",index=False,compression={"method":"gzip","mtime":0})
    importance.to_csv(directory/"feature_importance.csv",index=False)
    record = {"model":MODEL,"count_model":config["count_model"],"fit":fit,
        "predictions_sha256":hashlib.sha256(payload).hexdigest(),"data_sha256":hashes,
        "configuration_sha256":sha256_file(config_path),"holdout_scored":False,
        "holdout_previously_inspected":True,
        "source_sha256":{p.name:sha256_file(p) for p in Path(__file__).parent.glob("*.py")}}
    atomic_write(directory/"forecast_manifest.json",json.dumps(record,indent=2).encode())
    return predictions,record

def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',default='challenge_data')
    parser.add_argument('--output',default='predictions/prediction.csv')
    parser.add_argument('--config',default='configs/final_model.yaml')
    parser.add_argument('--directory',default='artifacts/final_model')
    args = parser.parse_args(argv)
    predictions,_ = run_prediction(args.data,args.output,args.config,args.directory)
    print(f'Wrote {len(predictions)} final-model forecasts to {args.output}')
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
