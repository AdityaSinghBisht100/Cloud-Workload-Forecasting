"""Typed release-data loading and streaming SHA-256 verification."""

import hashlib
from pathlib import Path

import pandas as pd

READING_DTYPES = {
    "timestamp": "Int64", "vm_id": "category",
    "min_cpu": "float64", "max_cpu": "float64", "avg_cpu": "float64",
}
METADATA_DTYPES = {
    "vm_id": "category", "subscription_id": "category", "deployment_id": "category",
    "timestamp_vm_created": "Int64", "timestamp_vm_deleted": "Int64",
    "max_cpu": "float64", "avg_cpu": "float64", "p95_max_cpu": "float64",
    "vm_category": "category", "virtual_core_bucket": "category",
    "memory_gb_bucket": "category",
}


def verify_checksums(data_dir="challenge_data") -> dict[str, str]:
    """Verify every manifest entry; raise on missing files or mismatches.

    Hash compressed files as distributed, without decompressing them. Manifest
    paths must stay within data_dir. No platform-specific sha256sum is needed.
    """
    root = Path(data_dir).resolve()
    verified = {}
    for line in (root / "SHA256SUMS").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        expected, name = line.split(maxsplit=1)
        name = name.lstrip("*")
        path = (root / name).resolve()
        if not path.is_relative_to(root):
            raise ValueError(f"Checksum path escapes data directory: {name}")
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected.lower():
            raise ValueError(f"SHA-256 mismatch: {name}")
        verified[name] = actual
    if not verified:
        raise ValueError("SHA256SUMS contains no entries")
    return verified


def _load(path, dtypes):
    frame = pd.read_csv(path, dtype=dtypes)
    missing = set(dtypes) - set(frame.columns)
    if missing:
        raise ValueError(f"{Path(path).name}: missing required columns {sorted(missing)}")
    return frame[list(dtypes)]


def load_readings(data_dir="challenge_data") -> pd.DataFrame:
    """Load relative integer seconds (never convert them to calendar dates)."""
    return _load(Path(data_dir) / "vm_cpu_readings.csv.gz", READING_DTYPES)


def load_metadata(data_dir="challenge_data") -> pd.DataFrame:
    return _load(Path(data_dir) / "vm_metadata.csv.gz", METADATA_DTYPES)
