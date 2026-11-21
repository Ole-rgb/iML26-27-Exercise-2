"""Fit every model on every dataset, quantify its fit and compute feature effects."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split
from tqdm import tqdm

from iml_feature_effects.data import Dataset
from iml_feature_effects.effects import EffectProperty, FeatureEffect


def split_test(dataset: Dataset, seed: int = 0, test_size: float = 0.25):
    """Hold out a stratified test set. Use this in *every* script so all of them share the same test set.

    :return: ``X_trainval, X_test, y_trainval, y_test``
    """
    return train_test_split(dataset.X, dataset.y, test_size=test_size, random_state=seed, stratify=dataset.y)


def _alpha(model: BaseEstimator) -> float | None:
    """L2 strength of a model"""
    params = model.get_params()
    keys = [k for k in params if k == "alpha" or k.endswith("__alpha")]
    return params[keys[0]] if keys else None


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
        X_train, X_test, y_train, y_test = split_test(dataset, seed=seed, test_size=test_size)
        for model_name, model in tqdm(models.items(), desc=dataset.name):
            # Clone so that the same unfitted model can be reused across datasets.
            fitted = clone(model).fit(X_train, y_train)
            X_fit, y_fit = getattr(fitted, "X_fit_", X_train), getattr(fitted, "y_fit_", y_train)
            fit = {
                "dataset": dataset.name,
                "model": model_name,
                "n_iter": fitted[-1].n_iter_,
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


def cross_validate_models(
    dataset: Dataset,
    models: dict[str, BaseEstimator],
    seed: int = 0,
    test_size: float = 0.25,
    n_splits: int = 5,
    n_repeats: int = 1,
) -> pd.DataFrame:
    """Repeated stratified CV of every model on the training ``dataset``.

    :return: One row per (model, fold) with columns ``dataset``, ``model``, ``alpha``, ``fold``,
        ``train_log_loss``, ``val_log_loss``, ``train_roc_auc``, ``val_roc_auc``.
    :rtype: pd.DataFrame
    """
    X, _, y, _ = split_test(dataset, seed=seed, test_size=test_size)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    rows = []
    splits = list(cv.split(X, y))

    for fold, (train_idx, val_idx) in enumerate(tqdm(splits, desc=dataset.name)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        for name, model in models.items():
            fitted = clone(model).fit(X_tr, y_tr)
            p_tr = fitted.predict_proba(X_tr)[:, 1]
            p_val = fitted.predict_proba(X_val)[:, 1]
            rows.append(
                {
                    "dataset": dataset.name,
                    "model": name,
                    "alpha": _alpha(model),
                    "fold": fold,
                    "n_iter": fitted[-1].n_iter_,
                    "train_log_loss": log_loss(y_tr, p_tr, labels=[0, 1]),
                    "val_log_loss": log_loss(y_val, p_val, labels=[0, 1]),
                    "train_roc_auc": roc_auc_score(y_tr, p_tr),
                    "val_roc_auc": roc_auc_score(y_val, p_val),
                }
            )
    return pd.DataFrame(rows)
