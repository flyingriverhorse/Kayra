"""Combined reference populations keep training membership and every user feature."""

import pandas as pd
import pytest

from skyulf.integrations.databricks.observability.monitoring import reference_layout


@pytest.mark.parametrize("temporal", [False, True])
@pytest.mark.parametrize("collision", [False, True])
def test_population_preserves_membership_duplicates_and_feature_names(temporal, collision):
    """Duplicate training rows survive and a user's is_train feature is never overwritten."""
    frame = pd.DataFrame({"x": [1.0, 1.0, 3.0, 4.0], "y": [2.0, 2.0, 6.0, 8.0]})
    columns = ["x", "y"]
    if collision:
        frame["is_train"] = [False, True, False, True]
        columns.append("is_train")
    population, membership = reference_layout.combine_reference_population(
        frame.head(3), frame.tail(1), columns, "temporal" if temporal else "random"
    )
    assert population[membership].tolist() == ([True] * 3 if temporal else [True] * 3 + [False])
    assert population.loc[population[membership], "x"].tolist() == [1.0, 1.0, 3.0]
    assert population["x"].tolist() == ([1.0, 1.0, 3.0] if temporal else [1.0, 1.0, 3.0, 4.0])
    if collision:
        assert membership != "is_train"
        assert population["is_train"].tolist() == (
            [False, True, False] if temporal else [False, True, False, True]
        )


def test_membership_name_avoids_case_insensitive_spark_columns():
    """Spark's default case-insensitive resolver must not merge an existing feature with the flag."""
    train = pd.DataFrame({"x": [1], "IS_TRAIN": [7]})
    population, marker = reference_layout.combine_reference_population(
        train, train, ["x", "IS_TRAIN"], "random"
    )
    assert marker == "_is_train"
    assert population["IS_TRAIN"].tolist() == [7, 7]
    assert population[marker].tolist() == [True, False]
