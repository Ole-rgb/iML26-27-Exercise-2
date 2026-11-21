"""Models ranging from underfit to overfit.

Extend or replace :func:`get_models` with your own models. They may be of the same class (as in
this example: three MLPs) or of different classes. For every model you have to be able to argue
*why* it is underfit, well fit or overfit, e.g. with train/test scores, learning curves or the
model's capacity relative to the amount of training data.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler


class SubsampledClassifier(ClassifierMixin, BaseEstimator):
    """Fit ``estimator`` on a random subset of ``n_samples`` training points only.

    :param estimator: Unfitted regressor.
    :param n_samples: Number of training points the estimator actually sees.
    :param random_state: Seed for drawing the subset.

    After fitting, ``X_fit_`` and ``y_fit_`` hold the subset, so the training error can be computed on
    the points the model actually saw.
    """

    def __init__(self, estimator: BaseEstimator, n_samples: int = 20, random_state: int | None = None):
        self.estimator = estimator
        self.n_samples = n_samples
        self.random_state = random_state

    def fit(self, X, y):
        rng = np.random.default_rng(self.random_state)
        idx = rng.choice(len(X), size=min(self.n_samples, len(X)), replace=False)
        self.X_fit_ = X.iloc[idx] if hasattr(X, "iloc") else X[idx]
        self.y_fit_ = y.iloc[idx] if hasattr(y, "iloc") else y[idx]
        self.estimator_ = clone(self.estimator).fit(self.X_fit_, self.y_fit_)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class _QuietMLPClassifier(MLPClassifier):
    """:class:`MLPClassifier` that does not warn when it stops at ``max_iter`` (intended here)."""

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            return super().fit(X, y)


def _mlp(**kwargs) -> Pipeline:
    """An MLP with standardized inputs and target."""
    return make_pipeline(StandardScaler(), _QuietMLPClassifier(**kwargs))


def get_models(seed: int = 0) -> dict[str, BaseEstimator]:
    """Return unfitted models, keyed by a unique name used in result tables.

    - ``mlp_underfit``: a single hidden neuron. The model is (up to the output layer) a ReLU of one
      linear projection of the inputs, far too little capacity for e.g. the bimodal daily rental pattern.
    - ``mlp_fit``: a moderately sized MLP with early stopping on a validation split.
    - ``mlp_overfit``: a large MLP without regularization, trained for many epochs on only 20
      training points, so it can memorize them.

    :param seed: Random seed passed to every model.
    :return: Mapping from model name to unfitted scikit-learn regressor.
    :rtype: dict[str, BaseEstimator]
    """
    ALPHAS = np.unique(np.concatenate([
        np.logspace(-4, 0, 6),      # overfit side: 6 points are enough, it's mostly the plateau
        np.logspace(0, 1.5, 10),    # transition: 1 → ~32, step factor ≈ 1.5 instead of ≈ 3.5
        [1e2],                      # one point on the constant plateau
    ]))
    return {
        f"mlp_alpha_{a:.0e}": _mlp(
            hidden_layer_sizes=(128, 128), alpha=a, max_iter=3000, tol=1e-6, random_state=seed
        )
        for a in ALPHAS
    }