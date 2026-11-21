"""Fit all models, compute feature effects and store scores, curves and plots.

Usage::

    uv run python scripts/run_experiment.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

from iml_feature_effects import (
    PartialDependence,
    Roughness,
    evaluate,
    get_selected_models,
    load_german_credit,
    load_heart_failure,
    load_parole_violators,
)

RESULTS = Path("results")
SEED = 0


def plot_curves(curves: pd.DataFrame, dataset: str, path: Path) -> None:
    """One panel per feature, one line per model."""
    data = curves[curves["dataset"] == dataset]
    features = data["feature"].unique()
    fig, axes = plt.subplots(1, len(features), figsize=(4 * len(features), 3.5), squeeze=False)
    for ax, feature in zip(axes[0], features, strict=True):
        for model, curve in data[data["feature"] == feature].groupby("model", sort=False):
            ax.plot(curve["value"], curve["effect"], marker=".", label=model)
        ax.set_xlabel(feature)
        ax.set_ylabel("centered partial dependence")
    axes[0][0].legend()
    fig.suptitle(dataset)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def main() -> None:
    datasets = [
        (load_parole_violators(), ["age", "time.served", "max.sentence"]),
        (load_heart_failure(), ["ejection_fraction", "serum_creatinine", "age"]),
        (load_german_credit(), ["duration", "credit_amount", "age"]),
    ]
    models = get_selected_models(seed=SEED)

    scores, curves = zip(
        *[
            evaluate(
                datasets=[(ds, features)],
                models=models[ds.name],
                effect=PartialDependence(),
                properties=[Roughness()],
                seed=SEED,
            )
            for ds, features in tqdm(datasets, desc="datasets")
        ],
        strict=True,
    )
    scores = pd.concat(scores, ignore_index=True)
    curves = pd.concat(curves, ignore_index=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    scores.to_csv(RESULTS / "scores.csv", index=False)
    curves.to_csv(RESULTS / "curves.csv", index=False)
    for dataset in curves["dataset"].unique():
        plot_curves(curves, dataset, RESULTS / f"effects_{dataset}.png")
    print(scores.to_string(index=False))


if __name__ == "__main__":
    main()
