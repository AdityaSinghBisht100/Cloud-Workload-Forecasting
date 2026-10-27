import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from cwfc.predict import run_prediction, validate_predictions
ROOT=Path(__file__).resolve().parents[1]

def smoke(tmp_path, n=40, changing=False, gap=False):
    data = tmp_path / "data"
    data.mkdir(exist_ok=True)
    rows = []
    for t in range(n):
        if gap and t == 4:
            continue
        for vm in range(3):
            if changing and vm == 2 and t < n - 24 - 3:
                continue
            rows.append([t * 300, str(vm), 0., 90., 10 + vm * 3 + .1 * t])
    # A duplicate must not overweight a VM in the aggregate target.
    rows.append(rows[0].copy())
    raw = pd.DataFrame(rows, columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])
    raw.to_csv(data / "vm_cpu_readings.csv.gz", index=False)
    metadata = pd.DataFrame({"vm_id": [str(i) for i in range(3)],
        "subscription_id": ["s"] * 3, "deployment_id": ["d"] * 3,
        "timestamp_vm_created": [0] * 3, "timestamp_vm_deleted": [n * 300 + 600] * 3,
        "max_cpu": [100.] * 3, "avg_cpu": [99.] * 3, "p95_max_cpu": [99.] * 3,
        "vm_category": ["A", "A", "B"], "virtual_core_bucket": ["2"] * 3,
        "memory_gb_bucket": ["4"] * 3})
    metadata.to_csv(data / "vm_metadata.csv.gz", index=False)
    config = tmp_path / "config.yaml"
    config.write_bytes((ROOT / "configs/final_model.yaml").read_bytes())
    return data, config, raw, metadata


def test_final_export_reconciles_and_ignores_future_labels(tmp_path):
    data,config,raw,metadata=smoke(tmp_path)
    settings=yaml.safe_load(config.read_text())
    settings['parameters'].update(n_estimators=8,min_child_samples=2,n_jobs=1)
    config.write_text(yaml.safe_dump(settings))
    output=tmp_path/'prediction.csv'
    directory=tmp_path/'evidence'
    first,record=run_prediction(data,output,config,directory)
    payload=output.read_bytes()
    assert first.shape==(24,3)
    assert record['fit']['latest_label_row']==15
    assert record['holdout_scored'] is False
    assert first.active_vm_count.eq(3).all()
    detail=pd.read_csv(directory/'individual_predictions.csv.gz')
    assert np.allclose(detail.groupby('step').predicted_cpu.mean(),first.aggregate_cpu_utilization)
    raw.loc[raw.timestamp>=16*300,['min_cpu','avg_cpu','max_cpu']]=-999
    raw.to_csv(data/'vm_cpu_readings.csv.gz',index=False)
    run_prediction(data,output,config,directory)
    assert output.read_bytes()==payload
    assert hashlib.sha256(payload).hexdigest()==record['predictions_sha256']

def test_short_history_rejected(tmp_path):
    data,config,_,_=smoke(tmp_path,n=36)
    with pytest.raises(ValueError,match='13 history'):
        run_prediction(data,tmp_path/'out.csv',config,tmp_path/'artifacts')

def test_export_schema_rejects_extra_columns():
    frame=pd.DataFrame({'timestamp':np.arange(24)*300,'active_vm_count':3.,'aggregate_cpu_utilization':10.,'extra':0})
    with pytest.raises(ValueError,match='columns'):
        validate_predictions(frame,frame.timestamp)
