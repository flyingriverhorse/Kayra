"""Prepared reference populations retain typed nulls across Delta writes."""

from types import SimpleNamespace

import pandas as pd


def test_prepared_frame_preserves_all_null_source_type(spark):
    """An all-null training column must not become a NullType Delta column."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_reference import (
        _prepared_frame,
    )

    original = spark.createDataFrame([(1, None)], "id long, optional string")
    frame = _prepared_frame(
        spark,
        pd.DataFrame({"id": [1, 2], "optional": [None, None]}),
        original.schema,
        SimpleNamespace(event_column=None, result_available_at_column=None),
    )
    assert frame.schema["optional"].dataType.typeName() == "string"
    assert frame.where("optional IS NULL").count() == 2


def test_replayed_schema_keeps_saved_types_and_never_returns_null_type(spark):
    """Unversioned references take types from saved rows, not a possibly redefined table."""
    from skyulf.integrations.databricks.observability.monitoring.spark.spark_monitoring_reference import (
        _replayed_schema,
    )

    schema = _replayed_schema(pd.DataFrame({"id": [1, 2], "x": [1.5, None], "empty": [None, None]}))
    assert [(f.name, f.dataType.typeName()) for f in schema.fields] == [
        ("id", "long"),
        ("x", "double"),
        ("empty", "string"),
    ]
