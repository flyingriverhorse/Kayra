"""Feature and training sources without Delta history fail clearly or are snapshotted."""

import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from skyulf.integrations.databricks.data.delta_io import delta as delta_io
from skyulf.integrations.databricks.features import delta, runtime
from skyulf.integrations.databricks.features.config import (
    FeatureGroup,
    FeaturePlan,
    parse_feature_config,
)
from skyulf.integrations.databricks.features.graph import plan_digest


class _Result:
    """A Spark SQL result with rows for collect() and first()."""

    def __init__(self, rows):
        """Keep rows as tuples or mappings."""
        self.rows = rows

    def collect(self):
        """Return all rows."""
        return self.rows

    def first(self):
        """Return the first row or None."""
        return self.rows[0] if self.rows else None

    def select(self, *columns):
        """Keep the version rows as they are."""
        return self


class _Spark:
    """Answer catalog questions from a table-kind map and record every statement."""

    def __init__(self, kinds, existing=None):
        """``kinds`` maps table name to its catalog Type; ``existing`` maps snapshots to owners."""
        self.kinds, self.existing, self.statements = kinds, dict(existing or {}), []
        self.catalog = SimpleNamespace(tableExists=lambda name: name in self.existing)

    def sql(self, statement):
        """Serve DESCRIBE, SHOW TBLPROPERTIES and DESCRIBE HISTORY; record DDL."""
        self.statements.append(statement)
        if statement.startswith("DESCRIBE TABLE EXTENDED"):
            name = statement.split()[-1].replace("`", "")
            return _Result([("Type", self.kinds.get(name, "MANAGED"))])
        if statement.startswith("SHOW TBLPROPERTIES"):
            name = statement.split()[2]
            return _Result([("skyulf.source_snapshot", self.existing[name])])
        if statement.startswith("DESCRIBE HISTORY"):
            return _Result([(7,)])
        if statement.startswith("CREATE OR REPLACE TABLE"):
            self.existing[statement.split()[4]] = "created"
        return _Result([])


def _snapshot_name(table, schema="c.out"):
    """Return the documented snapshot table name for ``table``."""
    return f"{schema}.skyulf_src_{hashlib.sha256(table.lower().encode()).hexdigest()[:16]}"


def test_regular_delta_tables_are_pinned_directly():
    """Managed and external tables keep today's DESCRIBE HISTORY pin; nothing is copied."""
    spark = _Spark({"c.raw.events": "MANAGED"})
    assert delta.pin_source(spark, "c.raw.events", policy="fail", snapshot_schema="c.out") == (
        "c.raw.events",
        7,
    )
    assert not any(s.startswith("CREATE") for s in spark.statements)


@pytest.mark.parametrize("kind", ["STREAMING_TABLE", "MATERIALIZED_VIEW", "VIEW"])
def test_unversioned_sources_fail_with_the_setting_to_use(kind):
    """With the copy turned off, a Lakeflow table must name the fix instead of a Spark error."""
    spark = _Spark({"c.raw.bronze": kind})
    with pytest.raises(ValueError, match=f"{kind}.*remove source_snapshot: fail"):
        delta.pin_source(spark, "c.raw.bronze", policy="fail", snapshot_schema="c.out")


def test_materialize_copies_into_an_owned_snapshot_and_pins_it():
    """The snapshot is a plain read of the source, tagged so later runs know it is ours."""
    spark = _Spark({"c.raw.bronze": "STREAMING_TABLE"})
    pinned = delta.pin_source(spark, "c.raw.bronze", policy="materialize", snapshot_schema="c.out")
    snapshot = _snapshot_name("c.raw.bronze")
    assert pinned == (snapshot, 7)
    create = next(s for s in spark.statements if s.startswith("CREATE OR REPLACE TABLE"))
    assert f"'skyulf.source_snapshot' = 'c.raw.bronze'" in create
    assert create.replace("`", "").endswith("AS SELECT * FROM c.raw.bronze")


def test_materialize_refuses_a_table_it_does_not_own():
    """A same-named table created by someone else must never be replaced."""
    snapshot = _snapshot_name("c.raw.bronze")
    spark = _Spark({"c.raw.bronze": "STREAMING_TABLE"}, {snapshot: "c.raw.other"})
    with pytest.raises(ValueError, match="not a Skyulf snapshot"):
        delta.pin_source(spark, "c.raw.bronze", policy="materialize", snapshot_schema="c.out")


def _config(**changes):
    """A base_transform-only feature configuration."""
    return {
        "version": 1,
        "base_table": "c.raw.bronze",
        "base_transform": "src/features/groups/base.py:prepare",
        "output_table": "c.out.observations",
        "keys": ["id"],
        "timestamp": "at",
        "groups": {},
    } | changes


def test_snapshot_settings_parse_with_the_output_schema_as_default():
    """Copying is the default and stores snapshots next to the output table."""
    plan = parse_feature_config(_config())
    assert plan is not None
    assert (plan.source_snapshot, plan.snapshot_schema) == ("materialize", "c.out")
    plan = parse_feature_config(
        _config(source_snapshot="materialize", source_snapshot_schema="c.snaps")
    )
    assert plan is not None and plan.snapshot_schema == "c.snaps"
    plan = parse_feature_config(_config(source_snapshot="fail"))
    assert plan is not None and plan.source_snapshot == "fail"
    with pytest.raises(ValueError, match="source_snapshot"):
        parse_feature_config(_config(source_snapshot="copy"))


def test_existing_feature_plans_keep_their_digest():
    """Deployed feature jobs carry plan_sha256; new optional fields must not change it."""
    group = FeatureGroup("g", "c.raw.src", "c.out.g", "src/features/groups/g.py:run", ("x",))
    plan = FeaturePlan("c.raw.base", "c.out.merged", ("id",), "at", (group,))
    legacy = {
        key: value
        for key, value in asdict(plan).items()
        if key not in {"base_transform", "source_snapshot", "source_snapshot_schema"}
    }
    expected = hashlib.sha256(json.dumps(legacy, sort_keys=True).encode()).hexdigest()
    assert plan_digest(plan) == expected
    assert plan_digest(replace(plan, source_snapshot="fail")) != expected


def test_initialize_records_snapshots_and_merge_reads_them(tmp_path, monkeypatch):
    """Group and merge tasks must read the pinned snapshot, not the live Lakeflow table."""
    path = tmp_path / "src/features/groups/base.py"
    path.parent.mkdir(parents=True)
    path.write_text("def prepare(frame):\n    return frame\n", encoding="utf-8")
    plan = parse_feature_config(_config(source_snapshot="materialize"))
    assert plan is not None
    spark = _Spark({"c.raw.bronze": "STREAMING_TABLE"})
    monkeypatch.setattr(runtime, "publication_state", lambda spark, table: None)
    snapshot = runtime.initialize_features(spark, tmp_path, plan, "*")
    assert snapshot["source_snapshots"] == {"c.raw.bronze": _snapshot_name("c.raw.bronze")}
    reads = []
    monkeypatch.setattr(runtime, "_require_snapshot", lambda plan, snapshot: None)
    monkeypatch.setattr(runtime, "_check_publication", lambda *args: {})
    monkeypatch.setattr(
        runtime, "read_version", lambda spark, table, version: reads.append(table) or [table]
    )
    monkeypatch.setattr(runtime, "join_feature_groups", lambda base, frames, plan: base)
    monkeypatch.setattr(runtime, "publish_feature_table", lambda *args, **kwargs: {"version": 1})
    runtime.merge_feature_groups(spark, plan, snapshot, {}, tmp_path)
    assert reads == [_snapshot_name("c.raw.bronze")]


def test_unversioned_kind_names_lakeflow_tables_and_views():
    """A history failure on an ST or MV maps to its type; a regular table keeps its error."""
    spark = _Spark({"c.raw.bronze": "STREAMING_TABLE", "c.raw.mv": "MATERIALIZED_VIEW"})
    error = RuntimeError("history failed")
    assert delta_io.unversioned_kind(spark, "c.raw.bronze", error) == "STREAMING_TABLE"
    assert delta_io.unversioned_kind(spark, "c.raw.mv", error) == "MATERIALIZED_VIEW"
    assert delta_io.unversioned_kind(spark, "c.raw.events", error) is None


def test_snapshots_are_dropped_after_a_successful_merge(tmp_path, monkeypatch):
    """Snapshots only serve one run; keeping them would cost storage on every run."""
    path = tmp_path / "src/features/groups/base.py"
    path.parent.mkdir(parents=True)
    path.write_text("def prepare(frame):\n    return frame\n", encoding="utf-8")
    plan = parse_feature_config(_config(source_snapshot="materialize"))
    assert plan is not None
    spark = _Spark({"c.raw.bronze": "STREAMING_TABLE"})
    monkeypatch.setattr(runtime, "publication_state", lambda spark, table: None)
    snapshot = runtime.initialize_features(spark, tmp_path, plan, "*")
    name = _snapshot_name("c.raw.bronze")
    spark.existing[name] = "c.raw.bronze"  # the tag written by CREATE ... TBLPROPERTIES
    monkeypatch.setattr(runtime, "_require_snapshot", lambda plan, snapshot: None)
    monkeypatch.setattr(runtime, "_check_publication", lambda *args: {})
    monkeypatch.setattr(runtime, "read_version", lambda spark, table, version: [table])
    monkeypatch.setattr(runtime, "join_feature_groups", lambda base, frames, plan: base)
    monkeypatch.setattr(runtime, "publish_feature_table", lambda *args, **kwargs: {"version": 1})
    runtime.merge_feature_groups(spark, plan, snapshot, {}, tmp_path)
    assert f"DROP TABLE IF EXISTS {name}" in spark.statements


def test_a_failed_merge_keeps_snapshots_for_repair(tmp_path, monkeypatch):
    """Repairing the failed task must still find the pinned copy."""
    path = tmp_path / "src/features/groups/base.py"
    path.parent.mkdir(parents=True)
    path.write_text("def prepare(frame):\n    return frame\n", encoding="utf-8")
    plan = parse_feature_config(_config(source_snapshot="materialize"))
    assert plan is not None
    spark = _Spark({"c.raw.bronze": "STREAMING_TABLE"})
    monkeypatch.setattr(runtime, "publication_state", lambda spark, table: None)
    snapshot = runtime.initialize_features(spark, tmp_path, plan, "*")
    monkeypatch.setattr(runtime, "_require_snapshot", lambda plan, snapshot: None)
    monkeypatch.setattr(runtime, "_check_publication", lambda *args: {})
    monkeypatch.setattr(runtime, "read_version", lambda spark, table, version: [table])
    monkeypatch.setattr(runtime, "join_feature_groups", lambda base, frames, plan: base)

    def fail(*args, **kwargs):
        """Simulate a failed publication."""
        raise RuntimeError("publish failed")

    monkeypatch.setattr(runtime, "publish_feature_table", fail)
    with pytest.raises(RuntimeError):
        runtime.merge_feature_groups(spark, plan, snapshot, {}, tmp_path)
    assert not any(s.startswith("DROP TABLE") for s in spark.statements)


def test_drop_refuses_tables_that_are_not_our_snapshot():
    """Only a table tagged as a Skyulf snapshot of that source is ever dropped."""
    name = _snapshot_name("c.raw.bronze")
    spark = _Spark({}, {name: "c.raw.other"})
    delta.drop_source_snapshots(spark, {"c.raw.bronze": name})
    assert not any(s.startswith("DROP TABLE") for s in spark.statements)


_DEDICATED = "[STREAMING_TABLE_OPERATION_NOT_ALLOWED.REQUIRES_SHARED_COMPUTE] not allowed"


class _Dedicated(_Spark):
    """Refuse history and detail on Lakeflow tables, like dedicated (SINGLE_USER) compute."""

    def __init__(self, kinds, *, error=_DEDICATED, describe=True):
        """``describe=False`` also refuses DESCRIBE TABLE EXTENDED."""
        super().__init__(kinds)
        self.error, self.describe = error, describe

    def sql(self, statement):
        """Raise ``error`` for refused statements and answer the rest."""
        refused = ("DESCRIBE HISTORY", "DESCRIBE DETAIL")
        if statement.startswith(refused) or (
            not self.describe and statement.startswith("DESCRIBE TABLE")
        ):
            raise RuntimeError(self.error)
        return super().sql(statement)


def _resolve(spark, workflow_config, **changes):
    """Resolve the latest training source for an offline project."""
    from skyulf.integrations.databricks.lifecycle import workflow

    config = {**workflow_config, "training_version": None, **changes}
    return workflow.resolve_training_spec(spark, config, datetime(2027, 1, 3, tzinfo=UTC))


@pytest.mark.parametrize("describe", [True, False])
def test_training_marks_streaming_table_on_dedicated_compute(
    workflow_config, monkeypatch, describe
):
    """Dedicated compute refuses ST history; training must mark it unversioned, not fail."""
    monkeypatch.delenv("DATABRICKS_RUNTIME_VERSION", raising=False)
    spark = _Dedicated({"workspace.test.source": "STREAMING_TABLE"}, describe=describe)
    spec = _resolve(spark, workflow_config)
    assert (spec.version, spec.source_kind) == (0, "STREAMING_TABLE")


def test_training_marks_materialized_view_and_keeps_other_errors(workflow_config):
    """An MV has no history on any compute; other history failures must not be hidden."""
    view = _Dedicated(
        {"workspace.test.source": "MATERIALIZED_VIEW"}, error="[EXPECT_TABLE_NOT_VIEW] a view"
    )
    spec = _resolve(view, workflow_config)
    assert (spec.version, spec.source_kind) == (0, "MATERIALIZED_VIEW")
    denied = _Dedicated({"workspace.test.source": "MANAGED"}, error="PERMISSION_DENIED")
    with pytest.raises(RuntimeError, match="PERMISSION_DENIED"):
        _resolve(denied, workflow_config)


def test_feature_lookup_rejects_unversioned_sources(workflow_config):
    """Feature lineage is rebuilt from Delta history, so an unversioned source must fail clearly."""
    lookup = {
        "lookups": [{"table_name": "c.features.f", "lookup_key": ["id"], "feature_names": ["x"]}],
    }
    spark = _Dedicated({"workspace.test.source": "STREAMING_TABLE"})
    with pytest.raises(ValueError, match="feature_lookup needs Delta history.*STREAMING_TABLE"):
        _resolve(
            spark, workflow_config, feature_lookup=lookup, engine="pandas", inference_mode="spark"
        )
