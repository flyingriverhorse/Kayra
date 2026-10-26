"""Readable notebook output must preserve failures and machine handoffs."""

import json
from types import SimpleNamespace

import pytest

from skyulf.inference.fitted_pipeline import ArtifactRuntimeMismatchError


def test_summary_omits_absent_options_but_keeps_false_zero_and_reasons():
    """Optional training choices must not obscure meaningful negative decisions."""
    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import result_summary

    result = result_summary(
        {
            "training": {"model_type": "linear_regression", "tuning": None},
            "eligible": False,
            "rows": 0,
            "reason": "No new rows",
            "empty": {},
            "parameters": {"fit_intercept": False},
        }
    )
    assert "linear_regression" in result and "No new rows" in result
    assert "Eligible: False" in result and "Rows: 0" in result
    assert "None" not in result and "Tuning" not in result and "Empty" not in result


def test_notebook_summary_preserves_original_json_and_task_values(capsys):
    """Human output cleanup must never remove fields consumed by downstream tasks."""
    from skyulf.integrations.databricks.jobs.shared.job_runtime import notebook_output

    payload = {"model_name": "catalog.schema.risk", "tuning": None, "eligible": False}
    output = notebook_output(
        payload, None, render=lambda _: "", display_html=None, exit_notebook=False
    )
    visible = capsys.readouterr().out
    assert json.loads(output) == payload
    assert "Model name: catalog.schema.risk" in visible and '"tuning"' not in visible


def test_failure_context_keeps_original_exception_and_source(capsys):
    """A failed task must expose its location and remain the same failed exception."""
    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

    failure = ValueError("Missing feature: age")
    dbutils = SimpleNamespace(
        widgets=SimpleNamespace(
            getAll=lambda: {
                "job_run_id": "42",
                "model_key": "risk",
                "secret_token": "never-print-me",
            }
        )
    )
    with pytest.raises(ValueError) as caught, notebook_task("train_model", dbutils):
        raise failure
    visible = capsys.readouterr().out
    assert caught.value is failure
    assert "FAILED | train_model" in visible and "Missing feature: age" in visible
    assert "test_notebook_diagnostics.py:" in visible and "risk" in visible
    assert "never-print-me" not in visible and "COMPLETED" not in visible


def test_unavailable_widget_context_cannot_mask_original_error(capsys):
    """Diagnostics must still work when notebook parameter reading is the failure."""
    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

    with (
        pytest.raises(RuntimeError, match="original failure"),
        notebook_task("initialize_run", None),
    ):
        raise RuntimeError("original failure")
    assert "FAILED | initialize_run" in capsys.readouterr().out


def test_success_context_reports_elapsed_time(capsys):
    """Successful execution should be distinguishable from a started or failed step."""
    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

    with notebook_task("score", None):
        pass
    visible = capsys.readouterr().out
    assert "STARTED | score" in visible and "COMPLETED | score" in visible
    assert "Elapsed:" in visible


def test_task_progress_flushes_before_slow_work(monkeypatch):
    """Operators must see the last started stage while a remote call is still pending."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

    output = Mock()
    monkeypatch.setattr("builtins.print", output)
    with notebook_task("reference.load_model", None):
        assert output.call_args.kwargs.get("flush") is True
    assert all(call.kwargs.get("flush") is True for call in output.call_args_list)


def test_generated_training_notebook_identifies_failure_without_exiting(monkeypatch, capsys):
    """The actual deployed entrypoint must wire diagnostics before any task execution."""
    import runpy
    from pathlib import Path
    from unittest.mock import Mock

    from skyulf.integrations.databricks.jobs.shared import job_runtime

    execute = Mock(side_effect=ValueError("Unknown training column"))
    monkeypatch.setattr(job_runtime, "run_lifecycle_notebook", execute)
    dbutils = SimpleNamespace(widgets=SimpleNamespace(getAll=lambda: {}), notebook=Mock())
    root = Path(__file__).resolve().parents[3] / "templates/databricks/template/{{.project_name}}"
    notebook = next(root.rglob("train_and_tune.py"))
    with pytest.raises(ValueError, match="Unknown training column"):
        runpy.run_path(
            str(notebook), run_name="__main__", init_globals={"dbutils": dbutils, "spark": None}
        )
    dbutils.notebook.exit.assert_not_called()
    assert "FAILED | train_and_tune" in capsys.readouterr().out


def test_summary_marks_truncation_and_does_not_dump_deep_optional_dicts():
    """Large reports must stay readable without presenting missing nested options as text."""
    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import result_summary

    text = result_summary(
        {"reason": "x" * 510, "a": {"b": {"c": {"items": [{"optional": None, "value": 1}]}}}}
    )
    assert "... (see JSON result)" in text
    assert "None" not in text


def test_chained_failure_shows_underlying_cause(capsys):
    """A wrapper error should point operators at its underlying failure too."""
    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

    with pytest.raises(ValueError, match="Unable to load"), notebook_task("load_data", None):
        try:
            raise FileNotFoundError("missing workflow.json")
        except FileNotFoundError as cause:
            raise ValueError("Unable to load") from cause
    assert "Caused by: FileNotFoundError: missing workflow.json" in capsys.readouterr().out


def test_non_json_result_values_keep_existing_string_serialization(capsys):
    """Array/nullable metadata must not turn a completed task into a display failure."""
    import numpy as np
    import pandas as pd

    from skyulf.integrations.databricks.jobs.shared.job_runtime import notebook_output

    payload = {"parameters": {"classes": np.array([1, 2]), "unknown": pd.NA}}
    output = notebook_output(
        payload, None, render=lambda _: "", display_html=None, exit_notebook=False
    )
    assert json.loads(output)["parameters"]["classes"] == "[1 2]"
    assert "Classes: [1 2]" in capsys.readouterr().out


def test_summary_failure_cannot_fail_completed_task(monkeypatch, capsys):
    """A presentation bug must fall back to the unchanged successful JSON result."""
    from unittest.mock import Mock

    from skyulf.integrations.databricks.jobs.shared import job_runtime

    monkeypatch.setattr(job_runtime, "result_summary", Mock(side_effect=ValueError("display")))
    output = job_runtime.notebook_output(
        {"rows": 12}, None, render=lambda _: "", display_html=None, exit_notebook=False
    )
    assert json.loads(output) == {"rows": 12}
    assert '"rows": 12' in capsys.readouterr().out


@pytest.mark.parametrize(
    ("error", "anchor"),
    [
        (
            ValueError(
                "Serving endpoint config update failed: UPDATE_FAILED. Served entity m-1: "
                "build failed; PERMISSION_DENIED in log tail"
            ),
            "serving-endpoint-update-failed",
        ),
        (
            ValueError("Owned serving endpoint must have exactly one served entity."),
            "serving-endpoint-does-not-serve-one-entity",
        ),
        (
            ArtifactRuntimeMismatchError(
                "Local artifact runtime mismatch for skyulf-core: requires 0.9.5, found 0.9.6."
            ),
            "champion-saved-by-another-core-release",
        ),
        (ValueError("on_drift must be disabled or retrain."), "on-drift-value-is-not-supported"),
        (
            ValueError(
                "Active performance_policy requires label_table and result_available_at_column."
            ),
            "performance-policy-without-a-label-table",
        ),
        (
            ValueError("Initialize the central monitoring store before writing."),
            "monitoring-store-is-not-initialized",
        ),
        (
            RuntimeError("[STREAMING_TABLE_OPERATION_NOT_ALLOWED] DESCRIBE HISTORY is not allowed"),
            "streaming-table-operation-not-allowed",
        ),
        (
            ValueError("only the generated job graph (workflow_contract: 3) can train from it"),
            "streaming-table-operation-not-allowed",
        ),
        (
            RuntimeError("PERMISSION_DENIED: User does not have SELECT"),
            "permission-denied-in-a-job-run",
        ),
        (
            RuntimeError("[INSUFFICIENT_PERMISSIONS] Insufficient privileges"),
            "permission-denied-in-a-job-run",
        ),
    ],
)
def test_known_failure_prints_fix_and_docs_link(error, anchor, capsys):
    """Known task failures must name the fix and its docs section, still re-raising the original."""
    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

    with pytest.raises(type(error)) as caught, notebook_task("deploy", None):
        raise error
    visible = capsys.readouterr().out
    docs = (
        f"Docs: https://www.skyulf.com/manual/user_guide/databricks_troubleshooting.html#{anchor}"
    )
    assert caught.value is error
    assert "Fix: " in visible and docs in visible
    assert "Inspect the traceback below" not in visible
    assert any(docs in note for note in error.__notes__)


def test_known_failure_matches_the_wrapped_cause(capsys):
    """A wrapper raised from a known failure must still point at the cause's fix."""
    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

    with pytest.raises(RuntimeError, match="Scoring failed"), notebook_task("score", None):
        try:
            raise ValueError("Initialize the central monitoring store before reading.")
        except ValueError as cause:
            raise RuntimeError("Scoring failed") from cause
    assert "#monitoring-store-is-not-initialized" in capsys.readouterr().out


def test_unknown_failure_keeps_generic_advice_and_original_exception(capsys):
    """An unrecognized failure must keep today's generic line, add no fix and stay unchanged."""
    from skyulf.integrations.databricks.jobs.shared.notebook_diagnostics import notebook_task

    failure = KeyError("unexpected column")
    with pytest.raises(KeyError) as caught, notebook_task("train", None):
        raise failure
    visible = capsys.readouterr().out
    assert caught.value is failure and caught.value.args == ("unexpected column",)
    assert "Inspect the traceback below and this task's inputs before retrying." in visible
    assert "Fix:" not in visible and "Docs:" not in visible
    assert len(failure.__notes__) == 1
