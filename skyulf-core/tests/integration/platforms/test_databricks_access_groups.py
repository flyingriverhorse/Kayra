"""Each shared target sets its four access groups in one block of deployment/targets.yml."""

import yaml
from test_databricks_bundle_generation import (
    CLI_ONLY,
    MANAGERS,
    _generate_project,
    _read_bundle,
)
from test_databricks_quick_setup import PROPERTIES, TEMPLATE, _cli_asked

ROLES = ("viewers", "runners", "serving_callers", "data_readers")
ACCESS = tuple(f"access_{role}" for role in ROLES)
SHARED = ("test", "syst", "prod")


def _answers(**groups):
    """Return wizard answers that set the given per-target access groups."""
    return {"access_groups": "true", **groups}


def test_quick_setup_never_asks_for_access_groups():
    """Quick setup stays short; empty groups grant nothing."""
    asked = _cli_asked(setup_mode="quick")
    assert not any("access" in name for name in asked)


def test_detailed_setup_asks_groups_only_after_opting_in():
    """Twelve group questions appear only when the operator chooses to set them now."""
    assert [n for n in _cli_asked(setup_mode="detailed") if "access" in n] == ["access_groups"]
    asked = _cli_asked(setup_mode="detailed", access_groups="true")
    expected = [f"{env}_{name}" for env in SHARED for name in ACCESS]
    assert [n for n in asked if "access" in n] == ["access_groups", *expected]


def test_old_view_group_answers_are_removed_answer_guards():
    """An old answer file with a view group fails instead of being silently dropped."""
    for env in SHARED:
        assert PROPERTIES[f"{env}_view_group"]["enum"] == [""]


@CLI_ONLY
def test_access_variables_are_declared_empty(tmp_path):
    """Empty defaults mean no access group applies until a target sets one."""
    variables = _read_bundle(_generate_project(tmp_path))["variables"]
    for name in ACCESS:
        assert variables[name]["default"] == ""
        assert variables[name]["description"]


@CLI_ONLY
def test_empty_groups_render_no_permission_entry(tmp_path):
    """DAB cannot skip an empty group_name everywhere, so none is rendered."""
    project = _generate_project(tmp_path, personal_development_targets="true")
    targets = _read_bundle(project)["targets"]
    for env in SHARED:
        assert targets[env]["permissions"] == MANAGERS
        assert {name: targets[env]["variables"][name] for name in ACCESS} == dict.fromkeys(
            ACCESS, ""
        )
    text = (project / "deployment/targets.yml").read_text(encoding="utf-8")
    assert "#   group_name: ${var.access_viewers}" in text


@CLI_ONLY
def test_shared_targets_render_their_groups_and_dev_targets_do_not(tmp_path):
    """Each environment grants its own company groups; personal targets stay private."""
    answers = _answers(
        test_access_viewers="fT-DBX-test-view",
        test_access_runners="fT-DBX-test-run",
        test_access_serving_callers="fT-DBX-test-call",
        test_access_data_readers="fT-DBX-test-read",
        prod_access_viewers="Data Team",
    )
    bundle = _read_bundle(
        _generate_project(tmp_path, personal_development_targets="true", **answers)
    )
    targets = bundle["targets"]
    viewer = {"level": "CAN_VIEW", "group_name": "${var.access_viewers}"}
    runner = {"level": "CAN_RUN", "group_name": "${var.access_runners}"}
    assert targets["test"]["permissions"] == [*MANAGERS, viewer, runner]
    assert targets["syst"]["permissions"] == MANAGERS
    assert targets["prod"]["permissions"] == [*MANAGERS, viewer]
    for env in SHARED:
        for name in ACCESS:
            assert targets[env]["variables"][name] == answers.get(f"{env}_{name}", "")
        development = targets[f"{env}_development"]
        assert development["permissions"] == MANAGERS
        assert not set(ACCESS) & set(development["variables"])


@CLI_ONLY
def test_jobs_receive_the_groups_they_apply(tmp_path):
    """Score and recovery grant data readers; the serving job grants callers and viewers."""
    project = _generate_project(tmp_path, prediction_delivery="both")
    score = yaml.safe_load((project / "resources/score.job.yml").read_text())
    tasks = {task["task_key"]: task for task in score["resources"]["jobs"]["score"]["tasks"]}
    parameters = tasks["score"]["notebook_task"]["base_parameters"]
    assert parameters["access_data_readers"] == "${var.access_data_readers}"
    serving = yaml.safe_load((project / "resources/serving_endpoint.job.yml").read_text())
    task = serving["resources"]["jobs"]["serving_endpoint"]["tasks"][0]
    parameters = task["notebook_task"]["base_parameters"]
    assert parameters["access_serving_callers"] == "${var.access_serving_callers}"
    assert parameters["access_viewers"] == "${var.access_viewers}"


@CLI_ONLY
def test_recovery_task_also_grants_data_readers(tmp_path):
    """A rebuilt prediction table must reach the same readers as a normal score."""
    project = _generate_project(tmp_path, rescore_when_history_expires="true")
    score = yaml.safe_load((project / "resources/score.job.yml").read_text())
    tasks = {task["task_key"]: task for task in score["resources"]["jobs"]["score"]["tasks"]}
    parameters = tasks["recover_predictions"]["notebook_task"]["base_parameters"]
    assert parameters["access_data_readers"] == "${var.access_data_readers}"
    jobs = TEMPLATE / "src/jobs"
    for notebook in ("score.py", "score_models.py", "recover_predictions.py"):
        assert "grant_prediction_readers(" in (jobs / notebook).read_text(encoding="utf-8")


def test_guide_tells_operators_where_the_access_groups_live():
    """The generated GUIDE is the operator's reference; it must name all four groups."""
    guide = (TEMPLATE / "GUIDE.md.tmpl").read_text(encoding="utf-8")
    section = guide.split("## Permissions and identities", 1)[1].split("\n## ", 1)[0]
    assert "deployment/targets.yml" in section
    for name in ACCESS:
        assert f"`{name}`" in section
    assert "uncomment its entry" in section
