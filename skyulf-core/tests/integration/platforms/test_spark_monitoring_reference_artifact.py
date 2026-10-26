"""Monitoring reads a model's saved training rows exactly, or says which grant is missing."""

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_reference as reference,
)
from skyulf.integrations.databricks.training.fitting.candidate import TrainingSpec
from skyulf.integrations.databricks.training.shared.holdout_evidence import save_frame_artifact
from skyulf.integrations.mlflow.shared import _client


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


def _saved_spec(tmp_path, monkeypatch, source: pd.DataFrame, spec: TrainingSpec) -> TrainingSpec:
    """Save rows as training does and point the spec at them through a fake tracking server."""
    client = _Artifacts(tmp_path / "mlruns")
    monkeypatch.setattr(_client, "make_tracking_client", lambda uri: client)
    receipt = save_frame_artifact(client, "run-1", "source", source)
    locator = {
        "run_id": "run-1",
        "path": "lifecycle/data/source.parquet",
        "tracking_uri": "databricks",
        "receipt": receipt,
    }
    return replace(spec, source_snapshot_json=json.dumps(locator, sort_keys=True))


def _small_spec() -> TrainingSpec:
    """Describe a four-row random split with generous budgets."""
    return TrainingSpec(
        table="a.b.training",
        version=1,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="y",
        max_rows=100,
        max_bytes=100000,
    )


def _small_source() -> pd.DataFrame:
    """Rows as training saved them, with the source id the live read recorded."""
    frame = pd.DataFrame({"id": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0], "y": [0.0, 1.0, 0.0, 1.0]})
    frame.attrs["source_table_id"] = "table-id"
    return frame


def test_saved_rows_round_trip_with_their_source_id(tmp_path, monkeypatch):
    """The rows monitoring replays are the exact rows training saved, with their source id."""
    spec = _saved_spec(tmp_path, monkeypatch, _small_source(), _small_spec())
    rows = reference._read_saved_training_rows(None, spec)
    pd.testing.assert_frame_equal(rows, _small_source())
    assert rows.attrs["source_table_id"] == "table-id"


def test_tampered_saved_rows_keep_the_digest_error(tmp_path, monkeypatch):
    """Changed bytes must fail as a digest mismatch, not as a missing permission."""
    spec = _saved_spec(tmp_path, monkeypatch, _small_source(), _small_spec())
    saved = tmp_path / "mlruns" / "run-1" / "lifecycle" / "data" / "source.parquet"
    data = bytearray(saved.read_bytes())
    data[len(data) // 2] ^= 0xFF
    saved.write_bytes(bytes(data))
    with pytest.raises(ValueError, match="digest differs"):
        reference._read_saved_training_rows(None, spec)


def test_unreadable_saved_rows_name_the_missing_grant(tmp_path, monkeypatch):
    """A denied artifact read must tell the operator which run and which grant is missing."""
    spec = _saved_spec(tmp_path, monkeypatch, _small_source(), _small_spec())

    def denied(*args):
        """Fail the way MLflow does when the identity cannot read the experiment."""
        raise PermissionError("403 PERMISSION_DENIED")

    monkeypatch.setattr(_Artifacts, "download_artifacts", denied)
    with pytest.raises(RuntimeError) as error:
        reference._read_saved_training_rows(None, spec)
    message = str(error.value)
    assert "runs:/run-1/lifecycle/data/source.parquet" in message
    assert "PermissionError" in message
    assert "CAN_READ on the training experiment" in message
    assert "READ VOLUME" in message
    assert isinstance(error.value.__cause__, PermissionError)
