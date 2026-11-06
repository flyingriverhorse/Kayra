"""Read pinned monitoring populations without collecting their rows on the driver."""

import importlib
from datetime import UTC, datetime, timedelta
from functools import reduce
from typing import Any

from ....data.delta_io.delta import (
    UNVERSIONED_KINDS,
    source_table_identity,
    table_identity,
    table_kind,
    unversioned_kind,
)
from ....feature_store.monitoring import enrich_observation, validate_observation_snapshot
from ..monitoring_config import MonitorConfig
from ..monitoring_sources import (
    model_predictions,
    observation_window,
    prediction_columns,
    prediction_source_version,
    read_snapshot,
    snapshot_at,
    window_receipts,
)


def require_unique_keys(frame: Any, keys: tuple[str, ...], name: str) -> None:
    """Reject null/nonfinite or duplicate keys through distributed scalar probes."""
    functions = importlib.import_module("pyspark.sql.functions")
    if not keys or set(keys) - set(frame.columns):
        raise ValueError(f"{name} is missing record key columns.")
    invalid = []
    for key in keys:
        column = functions.col(key)
        condition = column.isNull()
        if frame.schema[key].dataType.typeName() in {"float", "double"}:
            condition = (
                condition | functions.isnan(column) | (functions.abs(column) == float("inf"))
            )
        invalid.append(condition)
    if frame.where(reduce(lambda left, right: left | right, invalid)).limit(1).count():
        raise ValueError(f"{name} has a null or nonfinite record key.")
    if frame.groupBy(*keys).count().where("count > 1").limit(1).count():
        raise ValueError(f"{name} has duplicate record keys.")


def _unchanged_identity(spark: Any, table: str, expected: str) -> None:
    """Reject same-name table replacement across planning and execution."""
    if table_identity(spark, table) != expected:
        raise ValueError(f"Monitoring table was replaced during observation: {table}.")


def _matching_features(
    spark: Any,
    config: MonitorConfig,
    version: str,
    selected: Any,
    receipts: dict,
    keys: tuple[str, ...],
    features: tuple[str, ...],
    source_id: str,
    target_id: str,
    feature_binding: dict | None = None,
) -> tuple[Any, list[dict]]:
    """Join each saved batch against its own source version using distributed keys."""
    functions = importlib.import_module("pyspark.sql.functions")
    frames, used = [], []
    for run_id, item in receipts.items():
        batch_keys = selected.where(functions.col("run_id") == run_id).select(*keys)
        if not batch_keys.limit(1).count():
            continue
        source_version = prediction_source_version(
            item["receipt"], config, version, source_id, target_id
        )
        source = read_snapshot(spark, config.source_table, source_version)
        source = enrich_observation(
            spark,
            source.join(batch_keys, list(keys), "inner"),
            keys,
            features,
            feature_binding,
            item["receipt"],
        )
        joined = source.select(*dict.fromkeys((*keys, *features)))
        require_unique_keys(joined, keys, "current")
        if joined.count() != batch_keys.count():
            raise ValueError("Prediction keys differ from their pinned source snapshot.")
        validate_observation_snapshot(spark, feature_binding)
        frames.append(joined)
        used.append(
            {
                "run_id": run_id,
                "source_version": source_version,
                "prediction_commit_version": item["commit_version"],
                "committed_us": item["committed_us"],
            }
        )
    if frames:
        return reduce(lambda left, right: left.unionByName(right), frames), used
    empty = enrich_observation(
        spark, spark.table(config.source_table).limit(0), keys, features, feature_binding, None
    ).select(*dict.fromkeys((*keys, *features)))
    return empty, used


def read_spark_observation(
    spark: Any,
    config: MonitorConfig,
    version: str,
    keys: tuple[str, ...],
    features: tuple[str, ...],
    *,
    probabilities: int,
    as_of: datetime,
    start: datetime,
    end: datetime,
    feature_binding: dict | None = None,
) -> tuple[Any, Any, dict, datetime | None]:
    """Return distributed current features/predictions and bounded provenance receipts."""
    observation_window(as_of, start, end)
    functions = importlib.import_module("pyspark.sql.functions")
    source_id = table_identity(spark, config.source_table)
    target_id = table_identity(spark, config.prediction_table)
    prediction_version = snapshot_at(
        spark, config.prediction_table, end - timedelta(microseconds=1)
    )
    receipts = window_receipts(spark, config, start, end)
    frame = model_predictions(
        read_snapshot(spark, config.prediction_table, prediction_version),
        config,
        version,
        keys,
        probabilities,
    )
    selected = frame.where(functions.col("run_id").isin(list(receipts))).select(
        *prediction_columns(frame, keys, probabilities)
    )
    require_unique_keys(selected, keys, "predictions")
    current, used = _matching_features(
        spark,
        config,
        version,
        selected,
        receipts,
        keys,
        features,
        source_id,
        target_id,
        feature_binding,
    )
    _unchanged_identity(spark, config.source_table, source_id)
    _unchanged_identity(spark, config.prediction_table, target_id)
    evidence = {
        "source_table": config.source_table,
        "source_table_id": source_id,
        "prediction_table": config.prediction_table,
        "prediction_table_id": target_id,
        "prediction_version": prediction_version,
        "batches": used,
        "window_basis": "prediction_commit_timestamp",
        "execution_engine": "spark",
    }
    if feature_binding is not None:
        evidence["feature_lookup"] = feature_binding
    observed = (
        datetime.fromtimestamp(max(item["committed_us"] for item in used) / 1e6, UTC)
        if used
        else None
    )
    return current, selected.drop("run_id"), evidence, observed


def label_table_kind(spark: Any, table: str) -> str:
    """Return the catalog type; dedicated compute may name an ST only in its error."""
    try:
        return table_kind(spark, table)
    except Exception as error:  # noqa: BLE001 - dedicated compute may refuse DESCRIBE on an ST
        kind = unversioned_kind(spark, table, error)
        if kind is None:
            raise
        return kind


def read_label_source(spark: Any, table: str, kind: str, as_of: datetime) -> tuple[Any, dict]:
    """Pin a Delta label table; read a Streaming Table, view or MV as it is now."""
    if kind in UNVERSIONED_KINDS:
        # shortcut: no history pin, so a corrected label is read with its latest value;
        # add a first-seen ledger if label corrections matter.
        return spark.table(table), {
            "label_table": table,
            "label_table_id": source_table_identity(spark, table, kind),
            "label_version": None,
        }
    label_id = table_identity(spark, table)
    version = snapshot_at(spark, table, as_of)
    evidence = {"label_table": table, "label_table_id": label_id, "label_version": version}
    return read_snapshot(spark, table, version), evidence


def _text_key(frame: Any, key: str) -> Any:
    """Match request JSON text: DATE keys as yyyy-MM-dd, other keys cast to string."""
    functions = importlib.import_module("pyspark.sql.functions")
    column = functions.col(key)
    if frame.schema[key].dataType.typeName() == "date":
        return functions.date_format(column, "yyyy-MM-dd").alias(key)
    return column.cast("string").alias(key)


def read_spark_labels(
    spark: Any,
    config: MonitorConfig,
    keys: tuple[str, ...],
    target: str,
    predictions: Any,
    as_of: datetime,
    *,
    text_keys: bool = False,
) -> tuple[Any | None, dict]:
    """Join relevant outcomes without constructing a driver-side key list.

    ``text_keys`` compares keys as request text, for joins on keys taken from serving payloads.
    """
    table = config.label_table
    if table is None:
        return None, {"label_table": None, "label_version": None}
    kind = label_table_kind(spark, table)
    source, evidence = read_label_source(spark, table, kind, as_of)
    available = str(config.result_available_at_column)
    missing = sorted({*keys, target, available} - set(source.columns))
    if missing:
        raise ValueError(f"Label table {table} lacks columns: {', '.join(missing)}.")
    if source.schema[available].dataType.typeName() not in {"timestamp", "string", "date"}:
        raise ValueError("Label availability requires a UTC timestamp, aware ISO string or date.")
    if text_keys:
        source = source.select(*(_text_key(source, key) for key in keys), target, available)
    selected = source.join(predictions.select(*keys), list(keys), "left_semi").select(
        *dict.fromkeys((*keys, target, available))
    )
    if evidence["label_version"] is not None:
        _unchanged_identity(spark, table, evidence["label_table_id"])
    return selected, evidence
