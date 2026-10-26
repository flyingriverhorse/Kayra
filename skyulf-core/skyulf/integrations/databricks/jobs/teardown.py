"""Remove what one personal development deployment created, and nothing else.

Only a Bundle target in ``development`` mode with a non-empty ``resource_suffix`` is
accepted. Every name comes from the project's own configuration, bound to this target
by the resolvers the train, score and serving jobs use; nothing is found by scanning a
schema. A table or model is deleted only when its bound name contains the suffix. The
serving endpoint and its SQL function are deleted only when they carry this
deployment's owner tags. Source, label and training tables are never planned.

Order: endpoint, SQL function, monitoring rows, registered models (set first),
prediction views and tables, endpoint payload logs. A missing object is reported as
``already_absent``, so rerunning finishes a partial run. ``dry_run`` reads but never
deletes.
"""

import json
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from ..lifecycle.workflow import resolve_target_config
from ..observability.monitoring.monitoring_registration import monitoring_store_schema
from ..observability.monitoring.monitoring_store import ensure_owned_object
from ..projects.yaml_config import read_training_config, read_workflow_config, read_yaml_mapping
from ..projects.yaml_models import static_workflows
from ..shared._contracts import table_name

_BINDINGS = ("catalog", "input_schema", "output_schema", "metadata_schema", "resource_suffix")
_PAYLOAD_SUFFIXES = ("_payload", "_otel_logs", "_otel_spans", "_otel_metrics")
_MODEL_KEYED = ("monitoring_results", "model_inventory")
_ROW_FILTER = "lower(model_name) = :model AND project = :project AND environment = :environment"


@dataclass(frozen=True, slots=True)
class TeardownItem:
    """One object teardown deletes or deliberately leaves.

    Attributes:
        kind: Object kind, such as ``registered_model`` or ``prediction_table``.
        name: Full object name.
        action: ``delete`` or ``skip``.
        reason: Why the object is deleted or left.
        owners: Models that write a prediction table; they guard generation drops.
    """

    kind: str
    name: str
    action: str
    reason: str
    owners: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _Deployment:
    """Names and ownership of one deployment, resolved from its configuration."""

    values: dict[str, str]
    suffix: str
    models: list[str] = field(default_factory=list)
    outputs: dict[str, list[str]] = field(default_factory=dict)
    views: list[str] = field(default_factory=list)
    serving: dict[str, Any] | None = None
    namespace: str | None = None


def require_development_target(values: dict[str, str]) -> str:
    """Return the target's resource suffix, refusing anything but a personal target.

    Args:
        values: Job parameters with ``bundle_mode`` and ``resource_suffix``.

    Raises:
        ValueError: The Bundle mode is not ``development`` or the suffix is empty.
    """
    suffix = values.get("resource_suffix", "")
    if values.get("bundle_mode") != "development" or not suffix.strip():
        raise ValueError(
            "Teardown runs only in a personal development target (mode: development "
            "with a resource_suffix); shared targets are never torn down."
        )
    return suffix


def plan_teardown(values: dict[str, str]) -> list[TeardownItem]:
    """List every object this deployment created, in deletion order.

    Args:
        values: The teardown job parameters for one development target.

    Returns:
        Items to delete and items deliberately left, each with a reason.

    Raises:
        ValueError: The target is not a personal development target, or the
            project configuration does not resolve for it.
    """
    return _plan(_resolve(values))


def run_teardown(
    values: dict[str, str], *, spark: Any, client: Any, dry_run: bool
) -> list[dict[str, str]]:
    """Delete, or with ``dry_run`` only check, every planned object.

    A failed item is reported and the run continues, so one missing permission
    does not hide the rest. Rerun after fixing it; finished items report
    ``already_absent``.

    Args:
        values: The teardown job parameters for one development target.
        spark: Active Spark session.
        client: Databricks ``WorkspaceClient``.
        dry_run: Read and report only.

    Returns:
        One entry per item with ``kind``, ``name``, ``action``, ``status`` and
        ``message``. Status is ``deleted``, ``would_delete``, ``already_absent``,
        ``skipped`` or ``failed``.
    """
    deployment = _resolve(values)
    return [_run_item(item, deployment, spark, client, dry_run) for item in _plan(deployment)]


def run_teardown_notebook(spark: Any, dbutils: Any) -> dict[str, Any]:
    """Run the teardown job task and fail it when any item failed.

    Args:
        spark: Active Spark session.
        dbutils: Databricks notebook utilities.

    Returns:
        ``dry_run`` and the per-item results.

    Raises:
        ValueError: ``dry_run`` is not ``true`` or ``false``.
        RuntimeError: At least one item failed; the printed table names it.
    """
    values = dbutils.widgets.getAll()
    if values.get("dry_run") not in {"true", "false"}:
        raise ValueError("dry_run must be true or false.")
    from databricks.sdk import WorkspaceClient  # noqa: PLC0415 - optional integration dependency

    dry_run = values["dry_run"] == "true"
    items = run_teardown(values, spark=spark, client=WorkspaceClient(), dry_run=dry_run)
    output = {"dry_run": dry_run, "items": items}
    print(json.dumps(output, indent=2))
    failed = [item["name"] for item in items if item["status"] == "failed"]
    if failed:
        raise RuntimeError(f"Teardown failed for {len(failed)} item(s): {', '.join(failed)}.")
    return output


def _resolve(values: dict[str, str]) -> _Deployment:
    """Bind this target's names with the same resolvers the jobs use."""
    suffix = require_development_target(values)
    bindings = {key: values.get(key, "") for key in _BINDINGS}
    config = read_workflow_config(values["config_path"])
    training = read_training_config(Path(values["config_path"]).parent)
    workflows = static_workflows(training, config) if training is not None else [config]
    deployment = _Deployment(values, suffix, serving=_serving(values), namespace=_store(values))
    for workflow in (resolve_target_config(item, bindings) for item in workflows):
        _add_model(deployment, workflow["model_name"], workflow.get("prediction_table"))
    _add_model_set(deployment, values, resolve_target_config(config, bindings))
    return deployment


def _add_model(deployment: _Deployment, model: str, prediction_table: str | None) -> None:
    """Record one registered model and the prediction table it writes, once each."""
    if model not in deployment.models:
        deployment.models.append(model)
    if prediction_table:
        writers = deployment.outputs.setdefault(prediction_table, [])
        writers += [] if model in writers else [model]


def _add_model_set(deployment: _Deployment, values: dict[str, str], base: dict) -> None:
    """Put the model set before its component models, with its consumer views."""
    from ..model_sets.model_set_project import load_project_model_set  # noqa: PLC0415

    settings = load_project_model_set(values, base)
    if settings is None:
        return
    deployment.models.insert(0, settings["model_name"])
    table = settings.get("prediction_table")
    if table:
        deployment.outputs.setdefault(table, []).insert(0, settings["model_name"])
        deployment.views.extend(_publication_views(settings["publication"], table, deployment))


def _publication_views(policy: dict, table: str, deployment: _Deployment) -> list[str]:
    """Name the separate consumer views a model set publishes, as publication does."""
    if policy.get("mode") != "separate_views":
        return []
    explicit = policy.get("model_views")
    if explicit is not None:
        views = list(explicit.values())
    else:
        template = policy.get("model_view_template") or table + "_{branch}"
        views = [template.replace("{branch}", name) for name in _branch_names(deployment)]
    return [policy.get("combined_view") or f"{table}_combined", *views]


def _branch_names(deployment: _Deployment) -> list[str]:
    """Model entry names from training.yml, the keys a model set uses as branches."""
    training = read_training_config(Path(deployment.values["config_path"]).parent)
    return [] if training is None else list(training["models"])


def _store(values: dict[str, str]) -> str | None:
    """Return the development monitoring store, or None when none is configured."""
    catalog = values.get("monitoring_catalog", "")
    if not catalog or not values.get("development_monitoring_schema"):
        return None
    return f"{catalog}.{monitoring_store_schema(values, development=True)}"


def _serving(values: dict[str, str]) -> dict[str, Any] | None:
    """Bind the endpoint names exactly as the serving job does, or None without one."""
    path = values.get("serving_config_path", "")
    if not path or not Path(path).exists():
        return None
    settings = read_yaml_mapping(path).get("serving_endpoint")
    if not isinstance(settings, dict) or settings.get("enabled") is not True:
        return None
    from .serving_endpoint_job import _bound_settings  # noqa: PLC0415

    # Cost tags are only written on create; teardown never needs the real value.
    return _bound_settings(
        dict(settings), {**values, "cost_tag_key": "skyulf_teardown", "cost_tag_value": "none"}
    )


def _plan(deployment: _Deployment) -> list[TeardownItem]:
    """Order every item: serving, monitoring rows, models, prediction outputs, logs."""
    items = _serving_items(deployment.serving)
    # Rows go before models: monitoring_cleanup aborts on a row naming a deleted model.
    items += _monitoring_items(deployment)
    items += [_owned("registered_model", model, deployment) for model in deployment.models]
    items += [_owned("prediction_view", view, deployment) for view in deployment.views]
    for table, models in deployment.outputs.items():
        items.append(_owned("prediction_table", table, deployment))
        generations = _owned("prediction_generations", f"{table}_v*", deployment)
        items.append(replace(generations, owners=tuple(models)))
    items += _payload_items(deployment)
    items.append(
        TeardownItem(
            "mlflow_experiment",
            deployment.values.get("experiment_name", "<bundle root>/experiment"),
            "skip",
            "lives in the Bundle folder; `databricks bundle destroy` removes it",
        )
    )
    return items


def _owned(kind: str, name: str, deployment: _Deployment) -> TeardownItem:
    """Delete a name only when it carries this target's resource suffix."""
    if deployment.suffix.lower() in name.lower():
        return TeardownItem(kind, name, "delete", "name carries this target's resource_suffix")
    return TeardownItem(kind, name, "skip", "name lacks this target's resource_suffix")


def _serving_items(settings: dict[str, Any] | None) -> list[TeardownItem]:
    """Plan the endpoint and its SQL function; both are checked against owner tags."""
    if settings is None:
        return []
    reason = "deleted only when it carries this deployment's owner tags"
    items = [TeardownItem("serving_endpoint", settings["endpoint_name"], "delete", reason)]
    function = settings["sql_function"]
    if function.get("enabled", True):
        items.append(TeardownItem("sql_function", function["name"], "delete", reason))
    return items


def _monitoring_items(deployment: _Deployment) -> list[TeardownItem]:
    """Plan this deployment's rows in the development store; shared objects stay."""
    if deployment.namespace is None:
        return []
    reason = f"this model's rows in {deployment.namespace}; other models' rows stay"
    items = [
        TeardownItem("monitoring_rows", model, "delete", reason) for model in deployment.models
    ]
    items.append(
        TeardownItem(
            "monitoring_references",
            f"{deployment.namespace}.monitor_ref_*",
            "skip",
            "reference tables are shared per project and target; teardown never drops them. "
            "See the teardown guide to remove them by hand.",
        )
    )
    return items


def _payload_items(deployment: _Deployment) -> list[TeardownItem]:
    """Plan the endpoint's payload view and telemetry tables."""
    settings = deployment.serving
    if settings is None or settings.get("logging_mode", "telemetry") == "disabled":
        return []
    prefix = ".".join(
        settings[key] for key in ("logging_catalog", "logging_schema", "logging_table_prefix")
    )
    return [_owned("payload_log", prefix + suffix, deployment) for suffix in _PAYLOAD_SUFFIXES]


def _run_item(
    item: TeardownItem, deployment: _Deployment, spark: Any, client: Any, dry_run: bool
) -> dict[str, str]:
    """Run one item; any error becomes a failed result instead of stopping the run."""
    result = {"kind": item.kind, "name": item.name, "action": item.action}
    if item.action == "skip":
        return result | {"status": "skipped", "message": item.reason}
    try:
        status, message = _HANDLERS[item.kind](item, deployment, spark, client, dry_run)
    except Exception as error:  # noqa: BLE001 - report every failure and continue
        status, message = "failed", f"{type(error).__name__}: {error}"
    return result | {"status": status, "message": message}


def _not_found() -> type[Exception]:
    """Return the SDK's NotFound error."""
    from databricks.sdk.errors import NotFound  # noqa: PLC0415 - optional integration dependency

    return NotFound


def _as_dict(value: Any) -> dict[str, Any]:
    """Read an SDK object or a plain mapping as a dictionary."""
    return dict(value) if isinstance(value, dict) else value.as_dict()


def _endpoint(item, deployment, spark, client, dry_run) -> tuple[str, str]:
    """Delete the serving endpoint after checking its owner tags."""
    from ..serving.managed_endpoint import _require_owner  # noqa: PLC0415

    try:
        endpoint = _as_dict(client.serving_endpoints.get(item.name))
    except _not_found():
        return "already_absent", ""
    try:
        _require_owner(endpoint, deployment.serving["ownership"])
    except ValueError as error:
        return "skipped", str(error)
    if dry_run:
        return "would_delete", ""
    client.serving_endpoints.delete(item.name)
    return "deleted", ""


def _function(item, deployment, spark, client, dry_run) -> tuple[str, str]:
    """Delete the UC SQL function after checking its owner comment."""
    from ..serving.managed_endpoint import _require_function_owner  # noqa: PLC0415

    try:
        function = _as_dict(client.functions.get(item.name))
    except _not_found():
        return "already_absent", ""
    serving = deployment.serving
    expected = {"ownership": serving["ownership"], "endpoint_name": serving["endpoint_name"]}
    try:
        _require_function_owner(function.get("comment"), expected)
    except ValueError as error:
        return "skipped", str(error)
    if dry_run:
        return "would_delete", ""
    client.functions.delete(item.name)
    return "deleted", ""


def _model(item, deployment, spark, client, dry_run) -> tuple[str, str]:
    """Delete a UC registered model; Unity Catalog removes its versions and aliases."""
    try:
        client.registered_models.get(item.name)
    except _not_found():
        return "already_absent", ""
    if dry_run:
        return "would_delete", ""
    client.registered_models.delete(item.name)
    return "deleted", ""


def _monitoring_rows(item, deployment, spark, client, dry_run) -> tuple[str, str]:
    """Delete one model's inventory, results and actions rows from the dev store."""
    tables = {
        name: f"{deployment.namespace}.{name}"
        for name in ("performance_actions", "monitoring_results", "model_inventory")
    }
    present = [name for name, full in tables.items() if ensure_owned_object(spark, full)]
    values = deployment.values
    args = {
        "model": item.name.lower(),
        "project": values.get("bundle_project", ""),
        "environment": values.get("bundle_target", ""),
    }
    found = sum(_count(spark, tables[name], args) for name in present if name in _MODEL_KEYED)
    if not found:
        return "already_absent", ""
    if dry_run:
        return "would_delete", f"{found} row(s)"
    for statement in _delete_statements(present, tables):
        spark.sql(statement, args=args).collect()
    return "deleted", f"{found} row(s)"


def _delete_statements(present: list[str], tables: dict[str, str]) -> list[str]:
    """Return one DELETE per present store table, filtered to one model's rows."""
    statements = [
        f"DELETE FROM {table_name(tables[name])} WHERE {_ROW_FILTER}"
        for name in present
        if name in _MODEL_KEYED
    ]
    if {"performance_actions", "monitoring_results"} <= set(present):
        # Actions are keyed by report: delete them before the results that name them.
        reports = f"SELECT report_id FROM {table_name(tables['monitoring_results'])}"
        statements.insert(
            0,
            f"DELETE FROM {table_name(tables['performance_actions'])} "
            f"WHERE report_id IN ({reports} WHERE {_ROW_FILTER})",
        )
    return statements


def _count(spark: Any, table: str, args: dict[str, str]) -> int:
    """Count one model's rows in a store table."""
    query = f"SELECT count(*) AS n FROM {table_name(table)} WHERE {_ROW_FILTER}"
    return int(spark.sql(query, args=args).first()["n"])


def _drop(item, deployment, spark, client, dry_run) -> tuple[str, str]:
    """Drop one table or view."""
    return _drop_object(spark, item.name, dry_run)


def _drop_object(spark: Any, name: str, dry_run: bool) -> tuple[str, str]:
    """Drop a table or view by its catalog type."""
    if not spark.catalog.tableExists(name):
        return "already_absent", ""
    if dry_run:
        return "would_delete", ""
    kind = "VIEW" if spark.catalog.getTable(name).tableType.upper() == "VIEW" else "TABLE"
    spark.sql(f"DROP {kind} IF EXISTS {table_name(name)}").collect()
    return "deleted", ""


def _generations(item, deployment, spark, client, dry_run) -> tuple[str, str]:
    """Drop every full-rebuild generation this model wrote for the prediction table."""
    from ..scoring.shared.prediction_output import _owned_generation  # noqa: PLC0415

    logical = item.name.removesuffix("_v*")
    catalog, schema, base = logical.split(".")
    rows = spark.sql(f"SHOW TABLES IN {table_name(catalog + '.' + schema)} LIKE '{base}_v*'")
    names = [f"{catalog}.{schema}.{row['tableName']}" for row in rows.collect()]
    owned = [
        name
        for name in names
        if any(_owned_generation(spark, name, model) for model in item.owners)
    ]
    if not owned:
        return "already_absent", ""
    for name in owned:
        _drop_object(spark, name, dry_run)
    return ("would_delete" if dry_run else "deleted"), ", ".join(owned)


_HANDLERS: dict[str, Callable[..., tuple[str, str]]] = {
    "serving_endpoint": _endpoint,
    "sql_function": _function,
    "registered_model": _model,
    "monitoring_rows": _monitoring_rows,
    "prediction_view": _drop,
    "prediction_table": _drop,
    "prediction_generations": _generations,
    "payload_log": _drop,
}
