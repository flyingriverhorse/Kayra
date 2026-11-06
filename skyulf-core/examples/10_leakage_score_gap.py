"""Measure how much a leaky preprocessing order inflates the reported score.

Run from the repository root after installing skyulf-core:
    python skyulf-core/examples/10_leakage_score_gap.py

Uses seeded synthetic pandas data built in memory: no backend, download, or files.
The same model is trained twice per seed: once with feature selection BEFORE the
train/test split (deliberately allowed with ``on_leakage="ignore"``) and once in the
safe split-first order. Each pipeline's reported held-out accuracy is compared with
its accuracy on a fresh dataset drawn from the same process ("production").
Runtime checks pin the expected gaps so the numbers cannot drift silently.
"""

import logging
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from skyulf import SkyulfPipeline, validate_leakage_safety
from skyulf.leakage import OnLeakage

# A wide, small-sample table on purpose: that is where selection leakage bites.
N_ROWS = 300
N_FEATURES = 2000
N_INFORMATIVE = 10
N_SELECTED = 30
N_PRODUCTION_ROWS = 5000
SEEDS = range(10)


def _require(condition: bool, message: str) -> None:
    """Keep tutorial correctness checks active under optimized Python."""
    if not condition:
        raise RuntimeError(message)


def make_dataset(rng: np.random.Generator, n_rows: int) -> pd.DataFrame:
    """Draw rows where only the first N_INFORMATIVE features drive the label.

    P(target = 1) = sigmoid(f0000 + ... + f0009); the other features are pure noise.
    """
    features = rng.normal(size=(n_rows, N_FEATURES))
    probability = 1.0 / (1.0 + np.exp(-features[:, :N_INFORMATIVE].sum(axis=1)))
    data = pd.DataFrame(features, columns=[f"f{i:04d}" for i in range(N_FEATURES)])
    data["target"] = (rng.random(n_rows) < probability).astype(int)
    return data


SPLIT_STEP = {
    "name": "split",
    "transformer": "TrainTestSplitter",
    "params": {"test_size": 0.3, "random_state": 42, "target_column": "target", "stratify": True},
}
# Keeps the 30 columns with the highest ANOVA F-score against the target.
SELECT_STEP = {
    "name": "select",
    "transformer": "UnivariateSelection",
    "params": {"method": "SelectKBest", "k": N_SELECTED, "target_column": "target"},
}
MODEL = {"type": "logistic_regression"}
LEAKY_CONFIG = {"preprocessing": [SELECT_STEP, SPLIT_STEP], "modeling": MODEL}
SAFE_CONFIG = {"preprocessing": [SPLIT_STEP, SELECT_STEP], "modeling": MODEL}


def reported_and_production(
    config: dict[str, Any], data: pd.DataFrame, production: pd.DataFrame, on_leakage: OnLeakage
) -> tuple[float, float]:
    """Fit once; return (held-out accuracy skyulf reports, accuracy on fresh rows)."""
    pipeline = SkyulfPipeline(deepcopy(config))
    metrics = pipeline.fit(data, target_column="target", on_leakage=on_leakage)
    reported = metrics["modeling"]["splits"]["test"].metrics["accuracy"]
    predictions = pipeline.predict(production.drop(columns="target"))
    return float(reported), float(accuracy_score(production["target"], predictions))


def main() -> None:
    """Show the leakage diagnostic, then measure reported vs production accuracy."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")

    print("1. The leakage check flags selection before the split and passes the safe order")
    leaky_messages = validate_leakage_safety(LEAKY_CONFIG, on_leakage="warn")
    _require(bool(leaky_messages), "Selection before the split must be flagged")
    print("Leaky config:", leaky_messages[0])
    _require(validate_leakage_safety(SAFE_CONFIG) == [], "The split-first config must pass")
    print("Safe config:  no diagnostics")
    try:
        SkyulfPipeline(deepcopy(LEAKY_CONFIG)).fit(
            make_dataset(np.random.default_rng(0), 50), target_column="target"
        )
    except ValueError:
        print("With the default on_leakage='raise', fit() refuses the leaky order.")
    else:
        raise AssertionError("The default policy must reject the leaky order")

    print(f"\n2. Same model, two orders, {len(SEEDS)} seeds (metric: accuracy)")
    rows = []
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        data = make_dataset(rng, N_ROWS)
        production = make_dataset(rng, N_PRODUCTION_ROWS)
        # "ignore" deliberately runs the unsafe order without repeating section 1's warning.
        leaky = reported_and_production(LEAKY_CONFIG, data, production, "ignore")
        # The safe order runs under the default "raise" policy.
        safe = reported_and_production(SAFE_CONFIG, data, production, "raise")
        rows.append(
            {
                "seed": seed,
                "leaky_reported": leaky[0],
                "leaky_production": leaky[1],
                "safe_reported": safe[0],
                "safe_production": safe[1],
                "base_rate": production["target"].mean(),
            }
        )
    results = pd.DataFrame(rows).set_index("seed")
    print(results.round(3).to_string())

    mean = results.mean()
    leaky_gap = mean["leaky_reported"] - mean["leaky_production"]
    safe_gap = mean["safe_reported"] - mean["safe_production"]
    summary = pd.DataFrame(
        {
            "reported (held-out test)": [mean["leaky_reported"], mean["safe_reported"]],
            "production (fresh rows)": [mean["leaky_production"], mean["safe_production"]],
            "gap": [leaky_gap, safe_gap],
        },
        index=["leaky: select -> split", "safe: split -> select"],
    )
    print(f"\nMean accuracy over {len(SEEDS)} seeds (positive-class rate {mean['base_rate']:.1%}):")
    print((summary * 100).round(1).to_string())

    _require(
        leaky_gap >= 0.10,
        f"The leaky report must overstate production by >= 10 points: {leaky_gap:.3f}",
    )
    _require(
        abs(safe_gap) <= 0.05,
        f"The safe report must match production within 5 points: {safe_gap:.3f}",
    )

    print("\n3. What this does and does not show")
    # The selector saw the test rows' labels, so it kept noise columns that happen to
    # predict exactly those rows. The held-out score therefore grades its own answer key.
    print("The leaky report overstates production; the safe report predicts it.")
    # The leaky model is not worse here: picking features from all 300 rows finds the
    # real ones a little more often. The defect is the REPORT, not the model.
    print("The leaky model can do as well in production; only its reported score is wrong.")
    print("Remedy: report the split-first score; retrain on all rows only after recording it.")


if __name__ == "__main__":
    main()
