"""The score job grants SELECT on its published predictions to the target's data readers."""

import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from skyulf.integrations.databricks.scoring.shared.prediction_output import (
    grant_prediction_readers,
)


def _dbutils(group):
    """Return notebook utilities whose widgets hold the access_data_readers parameter."""
    widgets = SimpleNamespace(getAll=lambda: {"access_data_readers": group})
    return SimpleNamespace(widgets=widgets)


def _granted(spark):
    """Return the SQL statements the grant step sent."""
    return [call.args[0] for call in spark.sql.call_args_list]


def test_grants_select_on_the_table_and_every_view():
    """Readers need SELECT on each published object, views of a model set included."""
    spark = Mock()
    output = json.dumps(
        {
            "prediction_table": "main.gold.risk_predictions",
            "prediction_views": ["main.gold.risk_churn", "main.gold.risk_all"],
        }
    )
    names = grant_prediction_readers(spark, _dbutils("fT-DBX-readers"), output)
    assert names == ["main.gold.risk_predictions", "main.gold.risk_churn", "main.gold.risk_all"]
    assert _granted(spark) == [
        f"GRANT SELECT ON TABLE `main`.`gold`.`{name}` TO `fT-DBX-readers`"
        for name in ("risk_predictions", "risk_churn", "risk_all")
    ]


@pytest.mark.parametrize("group", ["", "  "])
def test_empty_group_grants_nothing(group):
    """An empty access_data_readers variable means no grant, as on personal targets."""
    spark = Mock()
    output = json.dumps({"prediction_table": "main.gold.p"})
    assert grant_prediction_readers(spark, _dbutils(group), output) == []
    spark.sql.assert_not_called()


def test_pending_recovery_grants_nothing():
    """The recovery task writes the predictions and grants them; the score task must not."""
    spark = Mock()
    output = json.dumps({"recovery_required": True, "prediction_table": "main.gold.p"})
    assert grant_prediction_readers(spark, _dbutils("readers"), output) == []
    spark.sql.assert_not_called()


def test_refused_grant_fails_and_names_the_group_and_privilege():
    """Silent failure would hide missing access, so the task fails with a clear message."""
    spark = Mock()
    spark.sql.side_effect = RuntimeError("PERMISSION_DENIED: User does not have MANAGE")
    output = json.dumps({"prediction_table": "main.gold.p"})
    with pytest.raises(PermissionError, match=r"SELECT.*main\.gold\.p.*'fT-DBX-readers'"):
        grant_prediction_readers(spark, _dbutils("fT-DBX-readers"), output)


@pytest.mark.parametrize("group", ["bad\ngroup", "a`b"])
def test_group_names_cannot_break_out_of_the_statement(group):
    """Control characters are rejected and backticks are escaped in the principal."""
    spark = Mock()
    output = json.dumps({"prediction_table": "main.gold.p"})
    if "\n" in group:
        with pytest.raises(ValueError, match="access_data_readers"):
            grant_prediction_readers(spark, _dbutils(group), output)
        return
    grant_prediction_readers(spark, _dbutils(group), output)
    assert _granted(spark) == ["GRANT SELECT ON TABLE `main`.`gold`.`p` TO `a``b`"]
