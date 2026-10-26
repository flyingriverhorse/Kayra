"""Shared-store defaults and the job identities that hold the store admission."""

import json

import pytest
import yaml
from test_databricks_bundle_generation import CLI_ONLY, TEMPLATE_ROOT, _generate_project


def test_monitoring_schema_default_is_shared():
    """New projects should reuse a stable store without inventing a per-project schema."""
    schema = json.loads((TEMPLATE_ROOT / "databricks_template_schema.json").read_text())
    assert schema["properties"]["monitoring_schema"]["default"] == "skyulf_monitoring"


@CLI_ONLY
@pytest.mark.parametrize(
    ("settings", "catalog", "schema"),
    [
        (
            {"monitoring": "true", "omit_fields": ("monitoring_schema",)},
            "${var.catalog}",
            "skyulf_monitoring",
        ),
        ({"monitoring": "true", "monitoring_schema": ""}, "${var.catalog}", "skyulf_monitoring"),
        ({"monitoring": "true", "monitoring_schema": "team_store"}, "${var.catalog}", "team_store"),
        (
            {
                "monitoring": "true",
                "monitoring_schema": "",
                "performance_policy_mode": "report",
                "label_table_name": "outcomes",
                "label_available_at_column": "available_at",
            },
            "${var.catalog}",
            "skyulf_monitoring",
        ),
        ({"monitoring": "false"}, "", ""),
        ({"monitoring": "false", "monitoring_schema": "unused_store"}, "", ""),
    ],
)
def test_rendered_monitoring_destination_defaults(tmp_path, settings, catalog, schema):
    """Monitoring uses each target's catalog and the shared default; off keeps it fully off."""
    project = _generate_project(tmp_path, **settings)
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    assert variables["monitoring_catalog"]["default"] == catalog
    assert variables["monitoring_schema"]["default"] == schema
    assert variables["monitoring_enabled"]["default"] == ("true" if catalog else "false")
    assert not (project / "resources/monitoring_cleanup.job.yml").exists()


@CLI_ONLY
@pytest.mark.parametrize(
    "identity", ["deployer", "shared_service_principal", "separate_service_principals"]
)
def test_store_writers_record_their_job_identity_and_no_cleanup_job_exists(tmp_path, identity):
    """Admission recovery needs each writer's run and task; no retention job is generated."""
    project = _generate_project(
        tmp_path,
        monitoring="true",
        deployment_identity=identity,
        manage_job_permissions="true",
        personal_development_targets="true",
        rescore_when_history_expires="true",
    )
    variables = yaml.safe_load((project / "deployment/variables.yml").read_text())["variables"]
    assert not {
        "monitoring_cleanup_cron",
        "monitoring_cleanup_confirm",
        "monitoring_cleanup_pause_status",
        "monitoring_reference_retention_days",
    } & set(variables)
    assert not (project / "resources/monitoring_cleanup.job.yml").exists()
    for resource, task_key in (
        ("monitoring", "monitor_model"),
        ("score", "prepare_monitoring"),
        ("score", "score"),
        ("score", "recover_predictions"),
    ):
        producer = yaml.safe_load((project / f"resources/{resource}.job.yml").read_text())[
            "resources"
        ]["jobs"][resource]
        task = next(item for item in producer["tasks"] if item["task_key"] == task_key)
        owner = task["notebook_task"]["base_parameters"]
        assert owner["job_run_id"] == "{{job.run_id}}"
        assert owner["task_key"] == task_key
        assert owner["workspace_host"] == "${workspace.host}"
    targets = yaml.safe_load((project / "deployment/targets.yml").read_text())["targets"]
    for target in targets.values():
        assert "monitoring_cleanup_pause_status" not in target.get("variables", {})
        jobs = target.get("resources", {}).get("jobs") or {}
        assert "monitoring_cleanup" not in jobs
        if target["mode"] == "development":
            # Only the personal cleanup_dev job; no short-name override, no shared identity.
            assert set(jobs) == {"cleanup_dev"}
            assert "run_as" not in jobs["cleanup_dev"]
