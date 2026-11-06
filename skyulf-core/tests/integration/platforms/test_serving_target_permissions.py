"""Serving grants can differ per Bundle target; each target applies only its own principals."""

import pytest
import yaml
from test_serving_endpoint_job import endpoint_project  # noqa: F401 - pytest fixture

QUERY = "CAN_QUERY"
PER_TARGET = {
    "dev": [{"group_name": "test-callers", "level": QUERY}, {"user_name": "alice@example.com"}],
    "prod": [{"group_name": "prod-callers", "level": QUERY}],
}


def _settings(endpoint_project, permissions, target="dev"):  # noqa: F811
    """Read the deployed YAML exactly as the serving job does for one target."""
    from skyulf.integrations.databricks.jobs.serving_endpoint_job import (
        read_serving_endpoint_settings,
    )

    document, path, values = endpoint_project
    document["serving_endpoint"]["permissions"] = permissions
    path.write_text(yaml.safe_dump(document))
    return read_serving_endpoint_settings({**values, "bundle_target": target})


def test_mapping_applies_only_the_current_targets_principals(endpoint_project):  # noqa: F811
    """A prod deployment must never try to grant the test group, which may not exist there."""
    assert _settings(endpoint_project, PER_TARGET, "dev")["permissions"] == [
        {"group_name": "test-callers", "level": QUERY},
        {"user_name": "alice@example.com", "level": QUERY},
    ]
    assert _settings(endpoint_project, PER_TARGET, "prod")["permissions"] == [
        {"group_name": "prod-callers", "level": QUERY}
    ]


def test_unlisted_target_grants_nothing(endpoint_project):  # noqa: F811
    """Personal development endpoints stay private unless their target is listed."""
    assert _settings(endpoint_project, PER_TARGET, "dev_alice")["permissions"] == []


def test_flat_list_still_applies_to_every_target(endpoint_project):  # noqa: F811
    """Single-environment projects keep the existing list form."""
    principals = [{"group_name": "data_products"}]
    for target in ("dev", "prod"):
        assert _settings(endpoint_project, principals, target)["permissions"] == [
            {"group_name": "data_products", "level": QUERY}
        ]


@pytest.mark.parametrize(
    "permissions",
    [
        {"dev": [], "prod": [{"group_name": "prod-callers", "level": "CAN_MANAGE"}]},
        {"dev": {"group_name": "test-callers"}},
        {"bad target!": []},
        {"": []},
    ],
)
def test_every_targets_entries_are_checked(endpoint_project, permissions):  # noqa: F811
    """A malformed entry for another target fails now, not only when that target deploys."""
    with pytest.raises(ValueError, match="permissions"):
        _settings(endpoint_project, permissions, "dev")


def _smoke_project(tmp_path, endpoint_project, permissions):  # noqa: F811
    """Write a minimal project with real target names and the serving document."""
    document, _, values = endpoint_project
    document["serving_endpoint"]["permissions"] = permissions
    (tmp_path / "config").mkdir()
    (tmp_path / "deployment").mkdir()
    (tmp_path / "config/serving.yml").write_text(yaml.safe_dump(document))
    targets = {"targets": {name: {"mode": "production"} for name in ("test", "syst", "prod")}}
    (tmp_path / "deployment/targets.yml").write_text(yaml.safe_dump(targets))
    return {key: value for key, value in values.items() if key != "serving_config_path"}


def test_smoke_check_rejects_a_mistyped_target(tmp_path, endpoint_project):  # noqa: F811
    """A typo such as porod would silently grant nothing in prod; the offline check stops it."""
    from skyulf.integrations.databricks.projects.project_checks import _check_serving

    bindings = _smoke_project(tmp_path, endpoint_project, {"test": [], "porod": []})
    with pytest.raises(ValueError, match="porod"):
        _check_serving(tmp_path, bindings)


def test_smoke_check_accepts_known_targets(tmp_path, endpoint_project):  # noqa: F811
    """Listing only some of the project's targets is valid."""
    from skyulf.integrations.databricks.projects.project_checks import _check_serving

    bindings = _smoke_project(tmp_path, endpoint_project, {"test": [], "prod": []})
    _check_serving(tmp_path, bindings)


def test_view_level_is_kept_per_target(endpoint_project):  # noqa: F811
    """Viewers must reach the job as CAN_VIEW, never upgraded to CAN_QUERY."""
    permissions = {"dev": [{"group_name": "dev-viewers", "level": "CAN_VIEW"}], "prod": []}
    assert _settings(endpoint_project, permissions, "dev")["permissions"] == [
        {"group_name": "dev-viewers", "level": "CAN_VIEW"}
    ]


ACCESS = {"access_serving_callers": "fT-DBX-callers", "access_viewers": "fT-DBX-viewers"}


def _with_access(endpoint_project, permissions, extra=None, target="prod"):  # noqa: F811
    """Read the settings with this target's access groups passed as job parameters."""
    document, path, values = endpoint_project
    document["serving_endpoint"]["permissions"] = permissions
    document["serving_endpoint"].update(extra or {})
    path.write_text(yaml.safe_dump(document))
    from skyulf.integrations.databricks.jobs.serving_endpoint_job import (
        read_serving_endpoint_settings,
    )

    return read_serving_endpoint_settings({**values, **ACCESS, "bundle_target": target})


def test_empty_serving_permissions_use_the_target_access_groups(endpoint_project):  # noqa: F811
    """One place per target: callers query and execute, viewers only see the endpoint."""
    settings = _with_access(endpoint_project, [])
    expected = [
        {"group_name": "fT-DBX-callers", "level": QUERY},
        {"group_name": "fT-DBX-viewers", "level": "CAN_VIEW"},
    ]
    assert settings["permissions"] == expected
    assert settings["function_permissions"] == expected


def test_unlisted_target_also_uses_the_access_groups(endpoint_project):  # noqa: F811
    """A target missing from a per-target mapping is empty, so its access groups apply."""
    settings = _with_access(endpoint_project, {"dev": [{"group_name": "dev-callers"}]})
    assert settings["permissions"][0] == {"group_name": "fT-DBX-callers", "level": QUERY}


def test_explicit_serving_permissions_win_over_access_groups(endpoint_project):  # noqa: F811
    """A nonempty serving.yml list for the target is an explicit override."""
    settings = _with_access(endpoint_project, {"prod": [{"group_name": "prod-callers"}]})
    assert settings["permissions"] == [{"group_name": "prod-callers", "level": QUERY}]


def test_explicit_function_permissions_stay_separate(endpoint_project):  # noqa: F811
    """UC EXECUTE grantees set in serving.yml are kept when the endpoint uses access groups."""
    extra = {"function_permissions": [{"group_name": "uc-callers"}]}
    settings = _with_access(endpoint_project, [], extra)
    assert settings["function_permissions"] == [{"group_name": "uc-callers", "level": QUERY}]
    assert settings["permissions"][0]["group_name"] == "fT-DBX-callers"


def test_one_group_as_viewer_and_caller_only_queries(endpoint_project):  # noqa: F811
    """An early CAN_VIEW grant must not downgrade the same group's CAN_QUERY."""
    document, path, values = endpoint_project
    path.write_text(yaml.safe_dump(document))
    from skyulf.integrations.databricks.jobs.serving_endpoint_job import (
        read_serving_endpoint_settings,
    )

    same = {"access_serving_callers": "fT-DBX-team", "access_viewers": "fT-DBX-team"}
    settings = read_serving_endpoint_settings({**values, **same, "bundle_target": "prod"})
    assert settings["permissions"] == [{"group_name": "fT-DBX-team", "level": QUERY}]


def test_no_access_groups_and_no_permissions_grant_nothing(endpoint_project):  # noqa: F811
    """Personal targets pass empty groups and keep their endpoint private."""
    assert _settings(endpoint_project, [], "dev_alice")["permissions"] == []
