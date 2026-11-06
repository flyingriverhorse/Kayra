"""Monitoring references rebuilt from saved training rows reproduce the old table results.

The golden file holds results measured on the frames the removed table path wrote.
The Delta round trip itself (lossless for these types) is checked on Databricks.
"""

import hashlib
import json
import math
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_config import MonitorConfig
from skyulf.integrations.databricks.observability.monitoring.monitoring_source_evidence import (
    source_evidence,
)
from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_reference as reference,
)
from skyulf.integrations.databricks.training.fitting.candidate import TrainingSpec
from skyulf.integrations.databricks.training.shared.holdout_evidence import save_frame_artifact
from skyulf.integrations.mlflow.shared import _client

FEATURE_DDL = (
    "x_float double, x_big long, x_decimal decimal(10,2), x_cat string, x_bool boolean, "
    "x_special double, x_when timestamp, x_late string, x_empty double"
)
SOURCE_DDL = f"id long, {FEATURE_DDL}, event timestamp, y double"
FEATURES = tuple(field.split()[0] for field in FEATURE_DDL.split(", "))
GOLDEN = Path(__file__).with_name("reference_parity_golden.json")
POLICY = {
    "mode": "report",
    "metric": "f1_weighted",
    "direction": "higher",
    "baseline": {"kind": "training_holdout", "model_version": "1"},
    "tolerance": 0.05,
    "tolerance_mode": "absolute",
    "window_hours": 24,
    "label_delay_hours": 6,
    "minimum_labeled_rows": 20,
    "minimum_label_coverage": 0.8,
    "consecutive_windows": 3,
}


class _Artifacts:
    """Keep run artifacts in a directory, logged and downloaded the way MLflow's client does."""

    def __init__(self, root: Path):
        """Store every run's files below one test directory."""
        self.root = root

    def log_artifact(self, run_id, path, artifact_path):
        """Copy one local file to the run's artifact folder."""
        target = self.root / run_id / artifact_path
        target.mkdir(parents=True, exist_ok=True)
        shutil.copy(path, target)

    def download_artifacts(self, run_id, path, directory):
        """Copy one saved file into the caller's directory and return its local path."""
        return shutil.copy(self.root / run_id / path, directory)


def wide_source(rows: int = 40) -> pd.DataFrame:
    """Saved rows with ties, big ints, decimals, many categories, nulls, NaN/inf and duplicates."""
    start = pd.Timestamp("2026-01-01", tz="UTC")
    frame = pd.DataFrame(
        {
            "id": range(rows),
            "x_float": [0.1 * (i % 5) for i in range(rows)],
            "x_big": [2**53 + i for i in range(rows)],
            "x_decimal": [Decimal(f"{i % 7}.25") if i % 9 else None for i in range(rows)],
            "x_cat": [f"c{i % 12}" for i in range(rows)],
            "x_bool": [bool(i % 3) if i % 10 else None for i in range(rows)],
            "x_special": [
                float("nan") if i % 11 == 0 else float("inf") if i % 13 == 0 else float(i % 4)
                for i in range(rows)
            ],
            "x_when": [start + pd.Timedelta(hours=i % 6) for i in range(rows)],
            # Non-null only from day 32: inside the temporal holdout, so null in its training rows.
            "x_late": [None if i < 32 else f"late{i % 2}" for i in range(rows)],
            "x_empty": [None] * rows,
            "event": [start + pd.Timedelta(days=i) for i in range(rows)],
            "y": [float(i % 5) for i in range(rows)],
        }
    )
    for name in FEATURES:
        frame.loc[20:23, name] = frame.loc[0:3, name].to_numpy()
    frame.attrs["source_table_id"] = "table-id"
    return frame


def wide_spec(features: tuple[str, ...], temporal: bool) -> TrainingSpec:
    """A random or temporal split over the given wide-source features."""
    split: dict[str, Any] = {}
    if temporal:
        split = {
            "split_strategy": "temporal",
            "event_column": "event",
            "start": datetime(2026, 1, 1, tzinfo=UTC),
            "holdout_start": datetime(2026, 2, 2, tzinfo=UTC),
            "cutoff": datetime(2026, 2, 10, tzinfo=UTC),
            "test_size": None,
            "random_state": None,
            "stratify": None,
        }
    return TrainingSpec(
        table="main.models.source",
        version=0,
        record_key_columns=("id",),
        input_columns=features,
        target_column="y",
        max_rows=1000,
        max_bytes=10_000_000,
        **split,
    )


def current_rows(spark, features):
    """Shifted production rows so every drift statistic is computed, not trivially zero."""
    when = datetime(2026, 1, 1, tzinfo=UTC)
    rows = [
        (
            0.1 * (i % 5) + 0.05,
            2**53 + i + 3,
            Decimal(f"{i % 5}.25"),
            f"c{i % 14}",
            i % 2 == 0,
            float(i % 6),
            when + timedelta(hours=i % 4),
            f"late{i % 3}",
            None,
        )
        for i in range(30)
    ]
    return spark.createDataFrame(rows, FEATURE_DDL).select(*features)


def saved_reference(spark, tmp_path, monkeypatch, features, *, temporal=False):
    """Build one model's reference from rows saved the way training saves them."""
    source = wide_source().loc[:, ["id", *features, "event", "y"]]
    source.attrs["source_table_id"] = "table-id"
    spec = wide_spec(features, temporal)
    client = _Artifacts(tmp_path / "mlruns")
    monkeypatch.setattr(_client, "make_tracking_client", lambda uri: client)
    receipt = save_frame_artifact(client, "run-1", "source", source)
    locator = {
        "run_id": "run-1",
        "path": "lifecycle/data/source.parquet",
        "tracking_uri": "databricks",
        "receipt": receipt,
    }
    located = replace(spec, source_snapshot_json=json.dumps(locator, sort_keys=True))
    config = MonitorConfig(
        environment="test",
        project="parity",
        model_name="main.models.model",
        model_version="1",
        source_table=spec.table,
        prediction_table="main.models.predictions",
        label_table="main.models.labels",
        result_available_at_column="label_at",
        execution_engine="spark",
        reference_namespace="main.monitoring",
        performance_policy=POLICY,
    )
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(fitted_engine="pandas", project_source_sha256=None)
    )
    evidence = {"model_name": config.model_name, "model_version": "1", "training_run_id": "run-1"}

    def measure(artifact, spec, holdout):
        """Derive a metric from the holdout rows so a changed holdout changes the baseline."""
        values = {"f1_weighted": float(holdout["y"].sum())}
        return {"values": values, "labeled_rows": len(holdout), "label_coverage": 1.0}

    patches = {
        "load_monitoring_artifact": lambda *a, **k: (artifact, located, {}, dict(evidence)),
        "make_registry_client": Mock(),
        "require_mlflow": Mock(),
        "reference_document": lambda *a: source_evidence(
            source, spec.source_columns, spec.dataset_id
        ),
        "validate_training_evidence": Mock(),
        "measure_holdout_values": measure,
        "performance_contract": lambda *a: "contract",
    }
    for name, value in patches.items():
        monkeypatch.setattr(reference, name, value)
    _, _, readers, loaded = reference.load_spark_reference_populations(
        spark, config, tracking_uri=None, registry_uri=None
    )
    return {kind: read() for kind, read in readers.items()}, loaded


def _result(spark, frames, features) -> dict:
    """The drift and freshness outputs the old reference tables were measured to give."""
    from skyulf.integrations.databricks.data.training.spark_retraining_data import _counts
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_drift import (
        drift_evidence,
    )

    drift = drift_evidence(frames["train"], current_rows(spark, features), features, None)
    counts = sorted(tuple(row) for row in _counts(frames["seen"], [*features, "y"]).collect())
    return {
        "drift": json.loads(json.dumps(drift)),
        "seen_counts_sha256": hashlib.sha256(json.dumps(counts).encode()).hexdigest(),
        "train_rows": frames["train"].count(),
        "seen_rows": frames["seen"].count(),
    }


def _assert_same_report(actual, expected, path="drift"):
    """Match every key, string and count exactly; floats only up to summation-order noise.

    NumPy and Arrow releases may add the same values in another order, which moves the
    last digit or two of a float. Anything larger is a real change.
    """
    if isinstance(expected, float) and isinstance(actual, float):
        assert math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15), path
    elif isinstance(expected, dict):
        assert isinstance(actual, dict) and actual.keys() == expected.keys(), path
        for key in expected:
            _assert_same_report(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, list):
        assert isinstance(actual, list) and len(actual) == len(expected), path
        for index, (left, right) in enumerate(zip(actual, expected, strict=True)):
            _assert_same_report(left, right, f"{path}[{index}]")
    else:
        assert actual == expected, path


CASES = {
    "random": (tuple(name for name in FEATURES if name != "x_empty"), False),
    "temporal": (tuple(name for name in FEATURES if name != "x_empty"), True),
    "all_null": (FEATURES, False),
}


@pytest.mark.parametrize("case", list(CASES))
def test_saved_rows_reference_reproduces_the_table_reference(spark, tmp_path, monkeypatch, case):
    """Drift, freshness counts and holdout baseline equal what the reference tables gave.

    reference_parity_golden.json was recorded from the frames the removed table path
    wrote (pyspark 4.0.3, local[2]); both paths agreed exactly then. Everything but
    float rounding noise must still match exactly.
    """
    features, temporal = CASES[case]
    frames, evidence = saved_reference(spark, tmp_path, monkeypatch, features, temporal=temporal)
    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))[case]
    result = _result(spark, frames, features)
    _assert_same_report(json.loads(json.dumps(result["drift"])), golden["drift"])
    assert result["seen_counts_sha256"] == golden["seen_counts_sha256"]
    assert (result["train_rows"], result["seen_rows"]) == (
        golden["train_rows"],
        golden["seen_rows"],
    )
    assert {"performance_baseline": evidence["performance_baseline"]} == golden["baseline"]
    assert evidence["prepared_reference_source_table_id"] == "table-id"


def test_all_null_feature_is_typed_string_from_saved_rows(spark, tmp_path, monkeypatch):
    """An always-null feature has no stored type; it reads as string and drift is unavailable."""
    frames, _ = saved_reference(spark, tmp_path, monkeypatch, FEATURES)
    assert frames["train"].schema["x_empty"].dataType.simpleString() == "string"
    assert frames["source"].schema["x_empty"].dataType.simpleString() == "string"
