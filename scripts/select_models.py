"""Cross-validate all models on all datasets and save the per-fold scores.

Usage::

    uv run python scripts/select_models.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from iml_feature_effects import get_cv_models, load_german_credit, load_heart_failure, load_parole_violators
from iml_feature_effects.evaluate import cross_validate_models

RESULTS = Path("results")
SEED = 0


def main() -> None:
    datasets = [load_parole_violators(), load_heart_failure(), load_german_credit()]
    models = get_cv_models(seed=SEED)

    cv_scores = []

    for ds in tqdm(datasets, desc="datasets"):
        cv_scores.append(cross_validate_models(ds, models, seed=SEED))

    cv_scores = pd.concat(cv_scores, ignore_index=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    cv_scores.to_csv(RESULTS / "cv_scores.csv", index=False)

    print(cv_scores.to_string(index=False))


if __name__ == "__main__":
    main()
