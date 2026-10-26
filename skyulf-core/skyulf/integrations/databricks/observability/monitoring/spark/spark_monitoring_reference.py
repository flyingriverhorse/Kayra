"""Rebuild a model's drift and freshness reference from its training run's saved rows."""

import importlib
import json
from collections.abc import Callable
from functools import partial
from typing import Any

import pandas as pd

from .....mlflow.shared._client import make_registry_client, require_mlflow
from ....jobs.shared.notebook_diagnostics import notebook_task
from ....shared._contracts import column_name
from ....training.fitting.candidate import read_training_snapshot, split_labeled_snapshot
from ....training.shared.training_evidence import validate_training_evidence
from ..monitoring_config import MonitorConfig
from ..monitoring_performance import measure_holdout_values, performance_contract
from ..monitoring_reference import load_monitoring_artifact, reference_document
from ..monitoring_source_evidence import validate_source_evidence
from ..reference_layout import combine_reference_population

RETRAIN_MESSAGE = (
    "Model {model} has no training rows saved in its MLflow run: this model was trained "
    "before 0.9.7 or outside the generated job graph; retrain it with the current project."
)


def _membership_rows(frame: Any, membership: str, *, train_only: bool) -> Any:
    """Keep training members when asked, then drop the boolean marker before any metric use."""
    if frame.schema[membership].dataType.simpleString() != "boolean":
        raise ValueError("Monitoring reference membership is not boolean.")
    if train_only:
        frame = frame.where(f"{column_name(membership)} = true")
    return frame.drop(membership)


def _prepared_holdout(artifact: Any, spec: Any, config: MonitorConfig, receipt: dict) -> dict:
    """Adapt exact holdout metrics to the selected policy."""
    policy = config.performance_policy
    if not policy or policy["mode"] == "off":
        return {}
    values = receipt["holdout"]
    evidence = receipt["evidence"]
    return {
        "performance_baseline": {
            "model_version": evidence["model_version"],
            "metric": policy["metric"],
            "contract_digest": performance_contract(artifact, spec, config),
            "value": values["values"].get(policy["metric"]),
            "labeled_rows": values["labeled_rows"],
            "label_coverage": values["label_coverage"],
            "reference": f"runs:/{evidence['training_run_id']}/training_filter_evidence.json",
            "kind": "training_holdout",
        }
    }


def load_spark_monitoring_reference(
    spark: Any, config: MonitorConfig, *, tracking_uri: str | None, registry_uri: str | None
) -> tuple[Any, Any, Any, dict]:
    """Return the model's training population and evidence for one observation."""
    artifact, spec, readers, evidence = load_spark_reference_populations(
        spark, config, tracking_uri=tracking_uri, registry_uri=registry_uri
    )
    return artifact, spec, readers["train"](), evidence


def load_spark_reference_populations(
    spark: Any, config: MonitorConfig, *, tracking_uri: str | None, registry_uri: str | None
) -> tuple[Any, Any, dict[str, Callable[[], Any]], dict]:
    """Return train/seen/source population readers; each builds its frame only when called.

    The populations come from the source rows the training run saved as an MLflow
    artifact (``source_snapshot_json``), verified by digest and training evidence.
    No table is read or written.
    """
    loaded = load_monitoring_artifact(config, tracking_uri=tracking_uri, registry_uri=registry_uri)
    readers, evidence = _artifact_reference(spark, config, loaded, tracking_uri, registry_uri)
    return loaded[0], loaded[1], readers, evidence


def _artifact_reference(
    spark: Any,
    config: MonitorConfig,
    loaded: tuple[Any, Any, dict, dict],
    tracking_uri: str | None,
    registry_uri: str | None,
) -> tuple[dict[str, Callable[[], Any]], dict]:
    """Rebuild the reference from the model's saved training rows."""
    artifact, spec, _, evidence = loaded
    if spec.source_snapshot_json is None:
        raise ValueError(RETRAIN_MESSAGE.format(model=config.model_name))
    identity = f"{config.model_name}/{config.model_version or config.model_alias}"
    rows = _verified_rows(spark, loaded, identity, tracking_uri, registry_uri)
    population = _prepared_frame(spark, rows["population"], rows["schema"], spec)
    membership = rows["membership"]
    readers: dict[str, Callable[[], Any]] = {
        "train": partial(_membership_rows, population, membership, train_only=True),
        "seen": partial(_membership_rows, population, membership, train_only=False),
        "source": partial(_prepared_frame, spark, rows["source"], rows["schema"], spec),
    }
    policy = config.performance_policy
    holdout = None
    if policy and policy["mode"] != "off":
        with notebook_task(f"reference.measure_holdout {identity}", None):
            holdout = measure_holdout_values(artifact, spec, rows["holdout"])
    baseline = _prepared_holdout(artifact, spec, config, {"evidence": evidence, "holdout": holdout})
    locator = json.loads(spec.source_snapshot_json)
    record = {
        "format": 3,
        "run_id": locator["run_id"],
        "path": locator["path"],
        "sha256": locator["receipt"]["sha256"],
        "membership_column": membership,
    }
    return readers, evidence | baseline | {
        "prepared_reference": record,
        "prepared_reference_source_table_id": rows["source_id"],
    }


def _prepared_frame(spark: Any, frame: pd.DataFrame, source_schema: Any, spec: Any) -> Any:
    """Retain transport types and recover typed null columns from the saved rows."""
    arrow = importlib.import_module("pyarrow")
    conversion = importlib.import_module("pyspark.sql.pandas.types")
    types = importlib.import_module("pyspark.sql.types")
    schema = conversion.from_arrow_schema(arrow.Schema.from_pandas(frame, preserve_index=False))
    dates = {spec.event_column, spec.result_available_at_column}
    for field in schema.fields:
        if isinstance(field.dataType, types.NullType):
            field.dataType = (
                types.TimestampType() if field.name in dates else source_schema[field.name].dataType
            )
    return spark.createDataFrame(frame, schema=schema)


def _replayed_schema(frame: pd.DataFrame) -> Any:
    """Spark types of saved training rows; a column that is null in every row reads as string."""
    arrow = importlib.import_module("pyarrow")
    conversion = importlib.import_module("pyspark.sql.pandas.types")
    types = importlib.import_module("pyspark.sql.types")
    schema = conversion.from_arrow_schema(arrow.Schema.from_pandas(frame, preserve_index=False))
    for field in schema.fields:
        if isinstance(field.dataType, types.NullType):
            field.dataType = types.StringType()
    return schema


def prepare_spark_monitoring_reference(
    spark: Any,
    config: MonitorConfig,
    *,
    tracking_uri: str | None = "databricks",
    registry_uri: str | None = "databricks-uc",
) -> dict:
    """Verify at activation that monitoring can rebuild this model's reference.

    Reads and checks the saved training rows exactly as every observation will, so a
    missing grant or an unsupported model fails before enrollment. Creates no table.
    """
    identity = f"{config.model_name}/{config.model_version or config.model_alias}"
    with notebook_task(f"reference.load_model {identity}", None):
        loaded = load_monitoring_artifact(
            config, tracking_uri=tracking_uri, registry_uri=registry_uri
        )
    _, evidence = _artifact_reference(spark, config, loaded, tracking_uri, registry_uri)
    return evidence["prepared_reference"]


def _verified_rows(
    spark: Any,
    loaded: tuple[Any, Any, dict, dict],
    identity: str,
    tracking_uri: str | None,
    registry_uri: str | None,
) -> dict[str, Any]:
    """Replay the exact training split and prove it matches the saved training evidence."""
    artifact, spec, filters, evidence = loaded
    with notebook_task(f"reference.read_training {identity}", None):
        source = _read_saved_training_rows(spark, spec)
        source_id = source.attrs.get("source_table_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Saved training rows do not record their source table identity.")
        client = make_registry_client(require_mlflow(), tracking_uri, registry_uri)
        source_receipt = reference_document(
            client, evidence["training_run_id"], "monitoring_source_evidence.json"
        )
        validate_source_evidence(source_receipt, source, spec.source_columns, spec.dataset_id)
        train, holdout, _ = split_labeled_snapshot(
            source, spec, engine=artifact.manifest.fitted_engine
        )
        validate_training_evidence(
            filters,
            spec,
            project_source_sha256=artifact.manifest.project_source_sha256,
            heldout=holdout,
        )
    columns = [*spec.input_columns, spec.target_column]
    population, membership = combine_reference_population(
        train, holdout, columns, spec.split_strategy
    )
    source_columns: list[str] = source_receipt["columns"]
    return {
        "source_id": source_id,
        "source": source.loc[:, source_columns],
        "population": population,
        "membership": membership,
        "holdout": holdout,
        # Types of always-null columns come from the saved rows, never a live table.
        "schema": _replayed_schema(source),
    }


def _read_saved_training_rows(spark: Any, spec: Any) -> pd.DataFrame:
    """Replay the model's saved training rows, naming the grant monitoring needs to read them."""
    try:
        return read_training_snapshot(spark, spec)
    except ValueError:
        raise
    except Exception as exc:  # noqa: BLE001 - every other read failure gets one actionable text
        # read_training_snapshot raises ValueError for a malformed locator, so this parse holds.
        locator = json.loads(spec.source_snapshot_json)
        raise RuntimeError(
            "Monitoring cannot read training rows saved at "
            f"runs:/{locator['run_id']}/{locator['path']} ({type(exc).__name__}); grant the "
            "monitoring job identity CAN_READ on the training experiment (and READ VOLUME if "
            "artifacts are in a UC volume)."
        ) from exc
