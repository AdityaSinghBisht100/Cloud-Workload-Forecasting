import hashlib
import logging

import numpy as np
import pandas as pd
import pytest

from cwfc.io import load_metadata, load_readings, verify_checksums
from cwfc.quality import observation_summary, prepare_readings, run_checks


def sample():
    return pd.DataFrame([
        [0, "a", 0., 40., 20.], [0, "a", 10., 50., 30.],
        [0, "b", 0., 20., 10.], [600, "a", 0., 30., 15.],
    ], columns=["timestamp", "vm_id", "min_cpu", "max_cpu", "avg_cpu"])


def test_duplicate_gap_and_partial_coverage(caplog):
    with caplog.at_level(logging.INFO):
        clean = prepare_readings(sample())
    assert "1 duplicate" in caplog.text
    assert len(clean) == 3
    summary = observation_summary(sample())
    assert summary.timestamp.tolist() == [0, 300, 600]
    assert summary.active_vm_count.tolist() == [2, 0, 1]
    assert summary.aggregate_cpu_utilization.iloc[0] == 17.5  # Equal VM weighting
    assert np.isnan(summary.aggregate_cpu_utilization.iloc[1])
    assert summary.missing_grid_timestamp.tolist() == [False, True, False]
    meta = pd.DataFrame({"vm_id": ["a", "c"], "timestamp_vm_created": [0, 0], "timestamp_vm_deleted": [900, 1200]})
    report = run_checks(sample(), meta)
    facts = report["checks"].set_index("check").value
    assert facts["duplicate_vm_timestamp_rows"] == 1
    assert facts["missing_grid_timestamps"] == 1
    assert facts["partial_coverage_vms"] == 2
    assert report["population_mismatches"] == {"readings_only": ["b"], "metadata_only": ["c"]}


def test_missing_cpu_is_not_imputed():
    frame = sample()
    frame.loc[3, "avg_cpu"] = np.nan
    summary = observation_summary(frame)
    assert summary.active_vm_count.tolist() == [2, 0, 0]
    assert summary.no_observations.iloc[-1]
    assert not summary.missing_grid_timestamp.iloc[-1]
    assert np.isnan(summary.aggregate_cpu_utilization.iloc[-1])


@pytest.mark.parametrize("column,value", [("avg_cpu", 101), ("min_cpu", 35), ("max_cpu", -1), ("avg_cpu", np.inf)])
def test_invalid_cpu_is_rejected(column, value):
    frame = sample()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match="CPU"):
        prepare_readings(frame)


def test_bad_keys_and_off_grid_are_rejected():
    frame = sample()
    frame.loc[0, "vm_id"] = None
    with pytest.raises(ValueError, match="Missing"):
        prepare_readings(frame)
    frame = sample()
    frame.loc[3, "timestamp"] = 601
    with pytest.raises(ValueError, match="aligned"):
        prepare_readings(frame)


def test_checksum_mismatch_and_path_escape(tmp_path):
    (tmp_path / "data").write_bytes(b"example")
    digest = hashlib.sha256(b"example").hexdigest()
    manifest = tmp_path / "SHA256SUMS"
    manifest.write_text(f"{digest}  data\n")
    assert verify_checksums(tmp_path) == {"data": digest}
    (tmp_path / "data").write_bytes(b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        verify_checksums(tmp_path)
    manifest.write_text(f"{digest}  ../escape\n")
    with pytest.raises(ValueError, match="escapes"):
        verify_checksums(tmp_path)


def test_typed_loading_and_required_schema(tmp_path):
    sample().to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    frame = load_readings(tmp_path)
    assert str(frame.vm_id.dtype) == "category"
    assert str(frame.timestamp.dtype) == "Int64"
    sample().drop(columns="avg_cpu").to_csv(tmp_path / "vm_cpu_readings.csv.gz", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_readings(tmp_path)
    pd.DataFrame({"vm_id": ["a"]}).to_csv(tmp_path / "vm_metadata.csv.gz", index=False)
    with pytest.raises(ValueError, match="missing required columns"):
        load_metadata(tmp_path)
