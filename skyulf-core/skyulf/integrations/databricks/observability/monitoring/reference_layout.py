"""Keep training membership when a monitoring reference combines its populations."""

import pandas as pd


def combine_reference_population(
    train: pd.DataFrame,
    holdout: pd.DataFrame,
    columns: list[str],
    split_strategy: str,
) -> tuple[pd.DataFrame, str]:
    """Keep train membership and duplicate rows without consuming a caller's feature name."""
    membership = "is_train"
    while membership.casefold() in {name.casefold() for name in columns}:
        membership = "_" + membership
    population = train.loc[:, columns].assign(**{membership: True})
    if split_strategy != "temporal":
        population = pd.concat(
            [population, holdout.loc[:, columns].assign(**{membership: False})],
            ignore_index=True,
        )
    return population, membership
