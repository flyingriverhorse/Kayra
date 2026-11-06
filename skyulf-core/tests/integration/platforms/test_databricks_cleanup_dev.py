"""The development cleanup_dev job removes only what one personal deployment created."""

import json
import re
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

pytest.importorskip("mlflow")
pytest.importorskip("databricks.sdk")

from databricks.sdk.errors import NotFound  # noqa: E402
from test_databricks_bundle_generation import (  # noqa: E402
    CLI,
    OFFLINE_CLI,
    PROFILE,
    _generate_project,
)

from skyulf.integrations.databricks.jobs.cleanup_dev import plan_cleanup, run_cleanup  # noqa: E402

EXAMPLES = Path(__file__).resolve().parents[3] / "templates/databricks/examples"
ENDPOINT = EXAMPLES / "nyctaxi/taxi_fare_endpoint"
MODEL_SET = EXAMPLES / "real_data/bike_demand_set"
SUFFIX = "_test_dev_42"
MODEL = f"ws.ml.taxi_fare_endpoint_model{SUFFIX}"
FUNCTION = f"ws.ml.taxi_fare_endpoint_score{SUFFIX}"
ENDPOINT_NAME = "taxi-fare-endpoint-test-development-endpoint-test-dev-42"
STORE = "ws.dev_mon"
DEV_TARGETS = ("test_development", "syst_development", "prod_development")
OWNER = {
    "project": "taxi_fare_endpoint",
    "target": "test_development",
    "suffix": SUFFIX,
    "managed_by": "serving_endpoint",
}
needs_cli = pytest.mark.skipif(
    not (PROFILE or OFFLINE_CLI) or not CLI,
    reason="Set SKYULF_BUNDLE_OFFLINE_CLI=1 to render the template with the installed CLI.",
)


def _values(project: Path = ENDPOINT, **overrides: str) -> dict[str, str]:
    """Return the job parameters a personal development target passes."""
    return {
        "config_path": str(project / "config/training.yml"),
        "serving_config_path": str(project / "config/serving.yml"),
        "bundle_project": project.name,
        "bundle_target": "test_development",
        "bundle_mode": "development",
        "catalog": "ws",
        "input_schema": "silver",
        "output_schema": "gold",
        "metadata_schema": "ml",
        "resource_suffix": SUFFIX,
        "monitoring_catalog": "ws",
        "monitoring_schema": "mon",
        "development_monitoring_schema": "dev_mon",
        "experiment_name": "/Workspace/Users/a/.bundle/p/test_development/experiment",
        **overrides,
    }


class FakeSpark:
    """Record SQL and keep a tiny catalog of objects and monitoring-store rows."""

    def __init__(self, objects: dict[str, str] | None = None) -> None:
        """Start with the given objects (name -> TABLE or VIEW) and empty stores."""
        self.objects = {name.lower(): kind for name, kind in (objects or {}).items()}
        self.properties: dict[str, dict[str, str]] = {}
        self.rows: dict[str, list[dict]] = {}
        self.statements: list[str] = []
        self.catalog = SimpleNamespace(
            tableExists=lambda name: name.lower() in self.objects,
            getTable=lambda name: SimpleNamespace(tableType=self.objects[name.lower()]),
        )

    def add_store(self, rows: dict[str, list[dict]]) -> None:
        """Create the owned development monitoring tables with the given rows."""
        for table in ("model_inventory", "monitoring_results", "performance_actions"):
            name = f"{STORE}.{table}"
            self.objects[name] = "TABLE"
            self.properties[name] = {"skyulf.monitoring.owner": "skyulf.monitoring.v1"}
            self.rows[name] = [dict(row) for row in rows.get(table, [])]

    def sql(self, query: str, args: dict | None = None) -> SimpleNamespace:
        """Apply the few statement shapes cleanup_dev issues."""
        self.statements.append(query)
        names = [part.replace("`", "") for part in re.findall(r"`[^\s(),]+`", query)]
        if query.startswith("SHOW TBLPROPERTIES"):
            props = self.properties.get(names[0].lower(), {})
            key = re.search(r"\('([^']+)'\)", query)
            rows = [{"key": k, "value": v} for k, v in props.items() if not key or k == key[1]]
            return _result(rows)
        if query.startswith("SELECT count(*)"):
            return _result([{"n": len(self._matching(names[0], args))}])
        if query.startswith("DELETE FROM") and "report_id IN" in query:
            reports = {row["report_id"] for row in self._matching(names[1], args)}
            self.rows[names[0]] = [r for r in self.rows[names[0]] if r["report_id"] not in reports]
        elif query.startswith("DELETE FROM"):
            kept = [r for r in self.rows[names[0]] if r not in self._matching(names[0], args)]
            self.rows[names[0]] = kept
        elif query.startswith("DROP"):
            self.objects.pop(names[0].lower(), None)
        elif query.startswith("SHOW TABLES"):
            base = query.split("LIKE '", 1)[1].removesuffix("_v*'")
            schema = ".".join(names[:2]).lower()
            listed = [n.rsplit(".", 1)[1] for n in self.objects if n.startswith(schema + ".")]
            return _result([{"tableName": n} for n in listed if n.startswith(base + "_v")])
        return _result([])

    def _matching(self, table: str, args: dict | None) -> list[dict]:
        """Rows of one store table for the requested model, project and environment."""
        args = args or {}
        return [
            row
            for row in self.rows.get(table, [])
            if row["model_name"].lower() == args["model"]
            and row["project"] == args["project"]
            and row["environment"] == args["environment"]
        ]


def _result(rows: list[dict]) -> SimpleNamespace:
    """Mimic the DataFrame methods cleanup_dev reads."""
    return SimpleNamespace(collect=lambda: rows, first=lambda: rows[0] if rows else None)


def _client(owner: dict[str, str] = OWNER) -> Mock:
    """Workspace client whose endpoint, function and model exist with the given owner."""
    client = Mock()
    tags = [{"key": key, "value": value} for key, value in owner.items()]
    client.serving_endpoints.get.return_value = {"name": ENDPOINT_NAME, "tags": tags}
    comment = {"ownership": owner, "endpoint_name": ENDPOINT_NAME, "schema_sha256": "x"}
    client.functions.get.return_value = {
        "comment": "Managed serving function " + json.dumps(comment, sort_keys=True)
    }
    client.registered_models.get.return_value = {"full_name": MODEL}
    return client


def _deployed_spark() -> FakeSpark:
    """A development store and payload objects as one endpoint deployment leaves them."""
    prefix = f"ws.ml.taxi_fare_endpoint_serving{SUFFIX}"
    spark = FakeSpark(
        {
            f"{prefix}_payload": "VIEW",
            f"{prefix}_otel_logs": "TABLE",
            f"{prefix}_otel_spans": "TABLE",
            f"{prefix}_otel_metrics": "TABLE",
        }
    )
    other = f"ws.ml.taxi_fare_endpoint_model_test_dev_99"
    row = {"project": "taxi_fare_endpoint", "environment": "test_development"}
    spark.add_store(
        {
            "model_inventory": [{**row, "model_name": MODEL}, {**row, "model_name": other}],
            "monitoring_results": [
                {**row, "model_name": MODEL, "report_id": "mine"},
                {**row, "model_name": other, "report_id": "theirs"},
            ],
            "performance_actions": [{"report_id": "mine"}, {"report_id": "theirs"}],
        }
    )
    return spark


def _status(results: list[dict]) -> dict[tuple[str, str], str]:
    """Index run results by kind and name."""
    return {(item["kind"], item["name"]): item["status"] for item in results}


@pytest.mark.parametrize(
    "overrides",
    [{"bundle_mode": "production"}, {"bundle_mode": ""}, {"resource_suffix": ""}],
)
def test_refuses_shared_targets_before_reading_anything(overrides):
    """A production target or an empty suffix must stop cleanup_dev before any lookup or delete."""
    spark, client = FakeSpark(), _client()
    with pytest.raises(ValueError, match="development"):
        plan_cleanup(_values(**overrides))
    with pytest.raises(ValueError, match="development"):
        run_cleanup(_values(**overrides), spark=spark, client=client, dry_run=False)
    assert spark.statements == []
    assert client.method_calls == []


def test_plan_lists_only_this_deployments_outputs():
    """Source, label and training tables are never planned; deleted names carry the suffix."""
    items = plan_cleanup(_values())
    names = {item.name for item in items}
    assert not any(table in name for name in names for table in ("trips_train", "trips_score"))
    assert {ENDPOINT_NAME, FUNCTION, MODEL} <= names
    for item in items:
        if item.action == "delete" and item.kind not in {"serving_endpoint", "sql_function"}:
            assert SUFFIX in item.name, item
    kinds = [item.kind for item in items]
    assert kinds.index("serving_endpoint") < kinds.index("sql_function")
    assert kinds.index("sql_function") < kinds.index("registered_model")
    skipped = {item.kind for item in items if item.action == "skip"}
    assert "mlflow_experiment" in skipped
    # Monitoring references live in the training run's artifacts; there is no table to plan.
    assert "monitoring_references" not in kinds


def test_logging_prefix_without_suffix_is_never_planned(tmp_path):
    """Payload tables shared by name with other deployments are reported, never dropped."""
    project = tmp_path / "taxi_fare_endpoint"
    shutil.copytree(ENDPOINT / "config", project / "config")
    serving = project / "config/serving.yml"
    text = serving.read_text(encoding="utf-8")
    serving.write_text(
        text.replace('"taxi_fare_endpoint_serving{resource_suffix}"', '"shared_serving"'),
        encoding="utf-8",
    )
    payload = [item for item in plan_cleanup(_values(project)) if item.kind == "payload_log"]
    assert payload
    assert all(item.action == "skip" and "resource_suffix" in item.reason for item in payload)


def test_model_set_plan_covers_the_set_and_its_component_models():
    """A model-set project removes the set, every branch model and their prediction tables."""
    values = _values(MODEL_SET, serving_config_path=str(MODEL_SET / "config/serving.yml"))
    items = plan_cleanup(values)
    planned = {(item.kind, item.name) for item in items if item.action == "delete"}
    models = [item.name for item in items if item.kind == "registered_model"]
    assert models[0] == f"ws.ml.demand_set{SUFFIX}"
    assert {
        f"ws.ml.bike_demand_set_casual{SUFFIX}",
        f"ws.ml.bike_demand_set_registered{SUFFIX}",
    } <= set(models)
    assert ("prediction_table", f"ws.gold.predictions{SUFFIX}") in planned
    assert not any("observations_" in name for _, name in planned)
    assert not any(item.kind == "serving_endpoint" for item in items)


def test_foreign_endpoint_and_function_are_skipped():
    """An endpoint or function without this deployment's owner tags is never deleted."""
    client = _client({**OWNER, "suffix": "_test_dev_99"})
    results = run_cleanup(_values(), spark=_deployed_spark(), client=client, dry_run=False)
    status = _status(results)
    assert status[("serving_endpoint", ENDPOINT_NAME)] == "skipped"
    assert status[("sql_function", FUNCTION)] == "skipped"
    client.serving_endpoints.delete.assert_not_called()
    client.functions.delete.assert_not_called()


def test_other_models_monitoring_rows_stay_in_the_shared_development_store():
    """Teammates share the dev store; only this deployment's model rows are deleted."""
    spark = _deployed_spark()
    results = run_cleanup(_values(), spark=spark, client=_client(), dry_run=False)
    assert _status(results)[("monitoring_rows", MODEL)] == "deleted"
    for table in ("model_inventory", "monitoring_results"):
        assert [row["model_name"] for row in spark.rows[f"{STORE}.{table}"]] == [
            "ws.ml.taxi_fare_endpoint_model_test_dev_99"
        ]
    assert spark.rows[f"{STORE}.performance_actions"] == [{"report_id": "theirs"}]
    assert f"{STORE}.model_inventory" in spark.objects


def test_dry_run_reads_but_never_deletes():
    """dry_run=true lists what would go and issues no delete or drop."""
    spark, client = _deployed_spark(), _client()
    results = run_cleanup(_values(), spark=spark, client=client, dry_run=True)
    assert not [s for s in spark.statements if s.startswith(("DROP", "DELETE"))]
    for api in (client.serving_endpoints, client.functions, client.registered_models):
        api.delete.assert_not_called()
    status = _status(results)
    assert status[("serving_endpoint", ENDPOINT_NAME)] == "would_delete"
    assert status[("registered_model", MODEL)] == "would_delete"
    assert status[("monitoring_rows", MODEL)] == "would_delete"
    assert len(spark.rows[f"{STORE}.model_inventory"]) == 2


def test_rerun_after_partial_failure_completes_with_already_absent():
    """A failed delete is reported, the rest continues, and a rerun finishes the job."""
    spark, client = _deployed_spark(), _client()
    client.registered_models.delete.side_effect = RuntimeError("permission denied")
    first = _status(run_cleanup(_values(), spark=spark, client=client, dry_run=False))
    assert first[("registered_model", MODEL)] == "failed"
    assert first[("serving_endpoint", ENDPOINT_NAME)] == "deleted"

    client.registered_models.delete.side_effect = None
    for api in (client.serving_endpoints, client.functions):
        api.get.side_effect = NotFound("gone")
    second = run_cleanup(_values(), spark=spark, client=client, dry_run=False)
    status = _status(second)
    assert "failed" not in status.values()
    assert status[("registered_model", MODEL)] == "deleted"
    assert status[("serving_endpoint", ENDPOINT_NAME)] == "already_absent"
    assert status[("monitoring_rows", MODEL)] == "already_absent"
    payload = f"ws.ml.taxi_fare_endpoint_serving{SUFFIX}_payload"
    assert status[("payload_log", payload)] == "already_absent"


def test_full_rebuild_generations_of_other_models_are_kept():
    """Only generations tagged with one of this deployment's models are dropped."""
    table = f"ws.gold.predictions{SUFFIX}"
    spark = FakeSpark({table: "VIEW", f"{table}_v1": "TABLE", f"{table}_v2": "TABLE"})
    for name, model in ((f"{table}_v1", f"ws.ml.demand_set{SUFFIX}"), (f"{table}_v2", "x.y.z")):
        spark.properties[name] = {"skyulf.mode": "full_rebuild", "skyulf.model_name": model}
    values = _values(MODEL_SET, serving_config_path="")
    status = _status(run_cleanup(values, spark=spark, client=_client(), dry_run=False))
    assert status[("prediction_table", table)] == "deleted"
    assert status[("prediction_generations", f"{table}_v*")] == "deleted"
    assert set(spark.objects) == {f"{table}_v2".lower()}


def _targets(project: Path) -> dict:
    """Parse the rendered Bundle targets."""
    return yaml.safe_load((project / "deployment/targets.yml").read_text(encoding="utf-8"))[
        "targets"
    ]


@needs_cli
def test_cleanup_dev_job_exists_only_in_personal_development_targets(tmp_path):
    """Shared targets never get the job; each dev target gets a dry-run-by-default job."""
    project = _generate_project(
        tmp_path,
        personal_development_targets="true",
        monitoring="true",
        monitoring_schema="taxi_monitoring",
        development_monitoring="true",
        monitoring_dashboard="create",
        monitoring_dashboard_warehouse_id="d047a4d9aa276958",
    )
    targets = _targets(project)
    for name in ("test", "syst", "prod"):
        assert "cleanup_dev" not in ((targets[name].get("resources") or {}).get("jobs") or {})
    for name in DEV_TARGETS:
        job = targets[name]["resources"]["jobs"]["cleanup_dev"]
        assert job["parameters"] == [{"name": "dry_run", "default": "true"}]
        assert job["max_concurrent_runs"] == 1
        assert "schedule" not in job
        task = job["tasks"][0]
        parameters = task["notebook_task"]["base_parameters"]
        assert parameters["bundle_mode"] == "${bundle.mode}"
        assert parameters["resource_suffix"] == "${var.resource_suffix}"
        assert (project / "deployment" / task["notebook_task"]["notebook_path"]).resolve() == (
            project / "src/jobs/cleanup_dev.py"
        ).resolve()
    # The development dashboard and its refresh task survive next to the new job.
    dev = targets["test_development"]["resources"]
    assert "monitoring_dashboard" in dev["dashboards"]
    assert "monitoring" in dev["jobs"]
    assert (project / "src/jobs/cleanup_dev.py").exists()


@needs_cli
def test_projects_without_personal_targets_have_no_cleanup_dev(tmp_path):
    """Without development targets there is nothing to tear down, so no notebook ships."""
    project = _generate_project(tmp_path)
    assert not (project / "src/jobs/cleanup_dev.py").exists()
    assert "cleanup_dev" not in (project / "deployment/targets.yml").read_text(encoding="utf-8")
