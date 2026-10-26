"""Join delayed labels to endpoint predictions by request rows or model-set record keys."""

import importlib
from datetime import datetime
from functools import reduce
from operator import or_
from typing import Any

from ..monitoring_config import MonitorConfig
from ..monitoring_sources import AWAITING_LABELS
from .serving_payloads import REQUEST_KEY, REQUEST_TIME, ROW_KEY

REQUEST_KEYS = (REQUEST_KEY, ROW_KEY)
_INTEGRAL = {"byte", "short", "integer", "long"}
# Request keys arrive as JSON text; these label types cast to the same text.
_TEXT_KEY_TYPES = _INTEGRAL | {"string", "boolean", "date"}


def serving_record_keys(config: MonitorConfig, spec: Any) -> tuple[str, ...]:
    """Model-set requests carry the record keys; single-model requests do not."""
    return tuple(spec.record_key_columns) if config.model_set_name else ()


def label_join_keys(table: str, columns: set[str], record_keys: tuple[str, ...]) -> tuple[str, ...]:
    """Prefer request columns when the label table has them, else the model-set record keys."""
    if set(REQUEST_KEYS) <= columns:
        return REQUEST_KEYS
    if not record_keys:
        raise ValueError(
            f"Serving label table {table} requires databricks_request_id and "
            "request_row_index for a single-model endpoint; business record keys alone "
            "cannot identify requests."
        )
    missing = sorted(set(record_keys) - columns)
    if missing:
        raise ValueError(
            f"Serving label table {table} needs the model-set record keys "
            f"({', '.join(record_keys)}) or databricks_request_id and request_row_index; "
            f"missing: {', '.join(missing)}."
        )
    return record_keys


def check_label_schema(
    table: str, types: dict[str, str], available: str, record_keys: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the join keys after checking key and availability types from the schema alone."""
    keys = label_join_keys(table, set(types), record_keys)
    if keys == REQUEST_KEYS and (types[REQUEST_KEY] != "string" or types[ROW_KEY] not in _INTEGRAL):
        raise ValueError(
            f"Serving label table {table} requires databricks_request_id STRING and "
            "request_row_index integral."
        )
    unmatched = [key for key in keys if types[key] not in _TEXT_KEY_TYPES]
    if keys != REQUEST_KEYS and unmatched:
        raise ValueError(
            f"Serving label table {table} key {unmatched[0]!r} is {types[unmatched[0]]}; "
            "record keys must be string, integer, boolean or date to match request values."
        )
    if types.get(available) not in {"timestamp", "string", "date"}:
        raise ValueError(
            f"Serving label availability {available!r} requires a UTC timestamp, "
            "aware ISO string or date."
        )
    return keys


def label_table_missing(spark: Any, table: str) -> bool:
    """True only while the schema exists but the label table does not; a wrong schema fails."""
    if spark.catalog.tableExists(table):
        return False
    schema = table.rsplit(".", 1)[0]
    if not spark.catalog.databaseExists(schema):
        raise ValueError(
            f"Label table {table} cannot be found: schema {schema} does not exist. "
            "Check monitoring_label_table."
        )
    return True


def label_types(spark: Any, table: str) -> dict[str, str]:
    """Read column types from the label table schema without reading rows."""
    return {field.name: field.dataType.typeName() for field in spark.table(table).schema.fields}


def latest_keyed_rows(predictions: Any, keys: tuple[str, ...]) -> tuple[Any, dict[str, int]]:
    """Keep the latest request row per record key and count what was left out."""
    functions = importlib.import_module("pyspark.sql.functions")
    window = importlib.import_module("pyspark.sql.window").Window
    columns = [functions.col(key) for key in keys]
    unkeyed = reduce(or_, [column.isNull() | (functions.trim(column) == "") for column in columns])
    keyed = predictions.where(~unkeyed)
    order = window.partitionBy(*columns).orderBy(
        functions.desc(REQUEST_TIME), functions.desc(REQUEST_KEY), functions.desc(ROW_KEY)
    )
    rank = "__skyulf_key_rank"
    latest = keyed.withColumn(rank, functions.row_number().over(order)).where(
        functions.col(rank) == 1
    )
    latest = latest.drop(rank)
    return latest, {
        "unkeyed_requests": predictions.where(unkeyed).select(REQUEST_KEY).distinct().count(),
        "duplicate_key_rows": keyed.count() - latest.count(),
    }


def read_serving_labels(
    spark: Any, config: MonitorConfig, spec: Any, predictions: Any, as_of: datetime
) -> tuple[Any, Any | None, dict]:
    """Return performance predictions and labels keyed by request rows, plus label evidence.

    A label table with request columns keeps the request join. Otherwise a model-set
    endpoint joins on its record keys: only the latest request per key counts, and rows
    without keys are left out of performance (drift keeps every request row).
    """
    from ..spark.spark_monitoring_sources import read_spark_labels  # noqa: PLC0415

    record = serving_record_keys(config, spec)
    target = spec.target_column
    table = config.label_table
    request_rows = predictions.drop(*record, REQUEST_TIME) if record else predictions
    if table is not None and label_table_missing(spark, table):
        return request_rows, None, {"label_table": table, "label_state": AWAITING_LABELS}
    keys = REQUEST_KEYS
    # Single-model requests carry no record keys, so only a model set can switch join.
    if record and table is not None:
        available = str(config.result_available_at_column)
        keys = check_label_schema(table, label_types(spark, table), available, record)
    if keys == REQUEST_KEYS:
        labels, evidence = read_spark_labels(spark, config, keys, target, predictions, as_of)
        return request_rows, labels, evidence
    latest, counts = latest_keyed_rows(predictions, keys)
    labels, evidence = read_spark_labels(spark, config, keys, target, latest, as_of, text_keys=True)
    if labels is not None:
        labels = latest.select(*REQUEST_KEYS, *keys).join(labels, list(keys)).drop(*keys)
    evidence |= counts | {"label_join_keys": list(keys)}
    return latest.drop(*keys, REQUEST_TIME), labels, evidence
