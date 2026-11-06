"""Monitoring references come only from the training rows saved in the model's run."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring.monitoring_source_evidence import (
    source_evidence,
)
from skyulf.integrations.databricks.observability.monitoring.spark import (
    spark_monitoring_reference as reference,
)
from skyulf.integrations.databricks.training.fitting.candidate import TrainingSpec

LOCATOR = {
    "run_id": "run-1",
    "path": "lifecycle/data/source.parquet",
    "tracking_uri": "databricks",
    "receipt": {"sha256": "f" * 64},
}


def test_model_without_saved_rows_must_be_retrained(monkeypatch):
    """Models trained before 0.9.7 have no saved rows; monitoring must say to retrain them."""
    spec = SimpleNamespace(source_snapshot_json=None)
    monkeypatch.setattr(
        reference,
        "load_monitoring_artifact",
        lambda *a, **k: (None, spec, {}, {"model_version": "1"}),
    )
    for call in (
        lambda: reference.load_spark_monitoring_reference(
            None, _config(), tracking_uri=None, registry_uri=None
        ),
        lambda: reference.prepare_spark_monitoring_reference(None, _config()),
    ):
        with pytest.raises(ValueError, match="retrain it with the current project"):
            call()


def _config(**changes):
    """Name one Spark-monitored model; only the fields the reference code reads matter."""
    values = {
        "model_name": "a.b.model",
        "model_version": "1",
        "model_alias": None,
        "reference_namespace": "a.monitoring",
        "performance_policy": None,
    }
    return SimpleNamespace(**(values | changes))


@pytest.fixture
def saved_rows_model(monkeypatch):
    """Stub MLflow, the model and Spark around the real split and evidence code."""
    source = pd.DataFrame(
        {"id": [1, 2, 3, 4], "x": [1.0, 1.0, 3.0, 4.0], "y": [2.0, 2.0, 6.0, 8.0]}
    )
    source.attrs["source_table_id"] = "table-id"
    spec = TrainingSpec(
        table="a.b.training",
        version=1,
        record_key_columns=("id",),
        input_columns=("x",),
        target_column="y",
        max_rows=100,
        max_bytes=10000,
        source_snapshot_json=json.dumps(LOCATOR),
    )
    artifact = SimpleNamespace(
        manifest=SimpleNamespace(fitted_engine="pandas", project_source_sha256=None)
    )
    evidence = {"training_run_id": "run-1", "model_version": "1", "model_name": "a.b.model"}
    monkeypatch.setattr(
        reference, "load_monitoring_artifact", lambda *a, **k: (artifact, spec, {}, dict(evidence))
    )
    read = Mock(return_value=source)
    monkeypatch.setattr(reference, "read_training_snapshot", read)
    monkeypatch.setattr(reference, "make_registry_client", Mock())
    monkeypatch.setattr(reference, "require_mlflow", Mock())
    monkeypatch.setattr(
        reference,
        "reference_document",
        lambda *a: source_evidence(source, spec.source_columns, spec.dataset_id),
    )
    monkeypatch.setattr(reference, "validate_training_evidence", Mock())
    monkeypatch.setattr(
        reference, "split_labeled_snapshot", lambda *a, **k: (source.head(3), source.tail(1), None)
    )
    monkeypatch.setattr(reference, "_replayed_schema", Mock(return_value="schema"))
    monkeypatch.setattr(reference, "_prepared_frame", Mock(side_effect=lambda *a: Mock()))
    measure = Mock(return_value={"values": {"rmse": 0.5}, "labeled_rows": 1, "label_coverage": 1.0})
    monkeypatch.setattr(reference, "measure_holdout_values", measure)
    monkeypatch.setattr(reference, "performance_contract", lambda *a: "contract")
    return SimpleNamespace(source=source, measure=measure, read=read)


def test_prepare_from_saved_rows_creates_no_reference_tables(saved_rows_model):
    """Models with saved training rows must not create monitor_ref_ or monitor_src_ tables."""
    record = reference.prepare_spark_monitoring_reference(None, _config())
    assert record == {
        "format": 3,
        "run_id": "run-1",
        "path": "lifecycle/data/source.parquet",
        "sha256": "f" * 64,
        "membership_column": "is_train",
    }
    saved_rows_model.read.assert_called_once()


def test_saved_rows_reference_uses_recorded_source_id_and_policy_gated_holdout(saved_rows_model):
    """Observations read no table; holdout inference runs only when a performance policy is on."""
    _, _, readers, evidence = reference.load_spark_reference_populations(
        None, _config(), tracking_uri=None, registry_uri=None
    )
    assert set(readers) == {"train", "seen", "source"}
    assert evidence["prepared_reference"]["format"] == 3
    assert evidence["prepared_reference_source_table_id"] == "table-id"
    assert "performance_baseline" not in evidence
    saved_rows_model.measure.assert_not_called()
    policy = {"mode": "warn", "metric": "rmse"}
    _, _, _, evidence = reference.load_spark_reference_populations(
        None, _config(performance_policy=policy), tracking_uri=None, registry_uri=None
    )
    baseline = evidence["performance_baseline"]
    assert (baseline["value"], baseline["labeled_rows"], baseline["kind"]) == (
        0.5,
        1,
        "training_holdout",
    )
    saved_rows_model.measure.assert_called_once()


def test_saved_rows_without_source_identity_fail_clearly(saved_rows_model):
    """Retraining compares the recorded source id; rows without one must not pass silently."""
    saved_rows_model.source.attrs.clear()
    with pytest.raises(ValueError, match="source table identity"):
        reference.load_spark_reference_populations(
            None, _config(), tracking_uri=None, registry_uri=None
        )
