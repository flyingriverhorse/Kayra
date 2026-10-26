"""Join explicitly selected, immutable outcome snapshots into training inputs."""

from dataclasses import replace
from typing import Any

from ...shared._contracts import table_name
from ..delta_io.delta import history, source_table_identity, unversioned_kind

LABEL_FIELDS = ("training_label_table", "training_label_version", "training_label_table_id")


def validate_training_labels(spec: Any) -> None:
    """Reject orphaned pins and unguarded external outcome sources before I/O."""
    if spec.training_label_table is None:
        if any(getattr(spec, name) is not None for name in LABEL_FIELDS[1:]):
            raise ValueError("training_label metadata requires training_label_table.")
        return
    table_name(spec.training_label_table)
    if spec.training_label_table.casefold() == spec.table.casefold():
        raise ValueError(
            "training_label_table must differ from training_table; use embedded labels otherwise."
        )
    if not spec.filter_unavailable_results:
        raise ValueError("training_label_table requires explicit result availability filtering.")
    _validate_label_pin(spec)


def _validate_label_pin(spec: Any) -> None:
    """Admit unresolved invocation pins or a complete saved outcome identity."""
    version = spec.training_label_version
    if version is not None and (type(version) is not int or version < 0):
        raise ValueError("training_label_version must be a nonnegative Delta version.")
    identity = spec.training_label_table_id
    if identity is not None and (not isinstance(identity, str) or not identity.strip()):
        raise ValueError("training_label_table_id must be a nonempty immutable Delta identity.")
    if identity is not None and version is None:
        raise ValueError("training_label_table_id requires training_label_version.")


def pin_training_labels(spark: Any, spec: Any) -> Any:
    """Resolve outcome version and identity once; never silently repin saved evidence."""
    if spec.training_label_table is None:
        return spec
    kind, version = spec.training_label_kind, spec.training_label_version
    if version is None:
        kind, version = _latest_label_version(spark, spec.training_label_table)
    identity = source_table_identity(spark, spec.training_label_table, kind)
    if spec.training_label_table_id is not None and spec.training_label_table_id != identity:
        raise ValueError("training_label_table was replaced after its identity was pinned.")
    result = replace(
        spec,
        training_label_version=version,
        training_label_table_id=identity,
        training_label_kind=kind,
    )
    validate_label_identity(spark, result)
    return result


def _latest_label_version(spark: Any, table: str) -> tuple[str | None, int]:
    """Pin the latest Delta version, or version 0 with the kind of an unversioned table."""
    try:
        latest = history(spark, table, limit=1).select("version").first()
    except Exception as exc:
        # The joined rows are frozen into the saved training snapshot, so a live read is enough.
        kind = unversioned_kind(spark, table, exc)
        if kind is None:
            raise
        return kind, 0
    if latest is None or type(latest["version"]) is not int or latest["version"] < 0:
        raise ValueError("training_label_table has no concrete Delta snapshot version.")
    return None, latest["version"]


def validate_label_identity(spark: Any, spec: Any) -> None:
    """Detect same-name replacement before and after materializing a joined read."""
    if spec.training_label_table is not None and (
        spec.training_label_table_id is None
        or source_table_identity(spark, spec.training_label_table, spec.training_label_kind)
        != spec.training_label_table_id
    ):
        raise ValueError("training_label_table identity changed or was not pinned.")


def label_binding(spec: Any) -> dict[str, Any] | None:
    """Expose the same concrete binding in training evidence and prepared frame metadata."""
    if spec.training_label_table is None:
        return None
    return {name: getattr(spec, name) for name in LABEL_FIELDS}


def validate_label_binding(frame: Any, spec: Any) -> None:
    """Prevent a prepared-data caller from fitting against another outcome snapshot."""
    expected = label_binding(spec)
    if frame.attrs.get("training_labels") != expected:
        raise ValueError("Prepared training data has a different outcome binding.")


def join_training_labels(spark: Any, source: Any, spec: Any) -> Any:
    """Left-join unique exact-typed keys; external outcomes replace embedded target fields."""
    if spec.training_label_table is None:
        return source
    from ...observability.monitoring.spark.spark_monitoring_sources import (  # noqa: PLC0415
        require_unique_keys,
    )

    validate_label_identity(spark, spec)
    reader = spark.read
    if spec.training_label_kind is None:
        reader = reader.format("delta").option("versionAsOf", spec.training_label_version)
    labels = reader.table(spec.training_label_table)
    required = [*spec.record_key_columns, spec.target_column, spec.result_available_at_column]
    if missing := set(required) - set(labels.columns):
        raise ValueError(f"training_label_table is missing columns: {sorted(missing)}.")
    labels = labels.select(*required)
    require_unique_keys(source, spec.record_key_columns, "Training source")
    require_unique_keys(labels, spec.record_key_columns, "training_label_table")
    if any(
        source.schema[key].dataType != labels.schema[key].dataType
        for key in spec.record_key_columns
    ):
        raise ValueError(
            "training_label_table keys must have exactly the training source key types."
        )
    return source.drop(spec.target_column, spec.result_available_at_column).join(
        labels, list(spec.record_key_columns), "left"
    )
