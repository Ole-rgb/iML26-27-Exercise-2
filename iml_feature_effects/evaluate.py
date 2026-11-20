"""Fit every model on every dataset, quantify its fit and compute feature effects."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import train_test_split

from iml_feature_effects.data import Dataset
from iml_feature_effects.effects import EffectProperty, FeatureEffect


def evaluate(
    datasets: Iterable[tuple[Dataset, list[str]]],
    models: dict[str, BaseEstimator],
    effect: FeatureEffect,
    properties: Iterable[EffectProperty],
    seed: int = 0,
    test_size: float = 0.25,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every model on every dataset, then explain the requested features.

    Train and test R² are reported so you can argue whether a model under- or overfits. Train R² is
    computed on the data the model was actually fit on (``X_fit_``/``y_fit_`` if the model has them, as
    :class:`~iml_feature_effects.models.SubsampledRegressor` does). Effects are computed on the test set.

    :param datasets: Pairs of a dataset and the names of the features to explain on it.
    :param models: Unfitted models keyed by name, e.g. from :func:`iml_feature_effects.models.get_models`.
    :param effect: Feature effect method.
    :param properties: Properties computed for each effect curve.
    :param seed: Random seed for the train/test split.
    :param test_size: Fraction of each dataset held out.
    :return: ``scores`` with one row per (dataset, model, feature) and columns ``train_r2``, ``test_r2``
        and one per property; ``curves`` with one row per grid point and columns ``dataset``, ``model``,
        ``feature``, ``value`` and ``effect``.
    :rtype: tuple[pd.DataFrame, pd.DataFrame]
    """
    properties = list(properties)
    score_rows, curves = [], []
    for dataset, features in datasets:
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.X, dataset.y, test_size=test_size, random_state=seed, stratify=dataset.y
        )
        for model_name, model in models.items():
            # Clone so that the same unfitted model can be reused across datasets.
            fitted = clone(model).fit(X_train, y_train)
            X_fit, y_fit = getattr(fitted, "X_fit_", X_train), getattr(fitted, "y_fit_", y_train)
            fit = {
                "dataset": dataset.name,
                "model": model_name,
                "train_log_loss": log_loss(y_fit, fitted.predict_proba(X_fit)[:, 1]),
                "test_log_loss": log_loss(y_test, fitted.predict_proba(X_test)[:, 1]),
                "train_roc_auc": roc_auc_score(y_fit, fitted.predict_proba(X_fit)[:, 1]),
                "test_roc_auc": roc_auc_score(y_test, fitted.predict_proba(X_test)[:, 1]),
            }
            for feature in features:
                curve = effect.explain(fitted, X_test, feature)
                score_rows.append({**fit, "feature": feature, **{p.name: p.score(curve) for p in properties}})
                curves.append(curve.assign(dataset=dataset.name, model=model_name, feature=feature))
    curves_df = pd.concat(curves, ignore_index=True)[["dataset", "model", "feature", "value", "effect"]]
    return pd.DataFrame(score_rows), curves_df
