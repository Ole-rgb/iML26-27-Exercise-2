"""Models ranging from underfit to overfit.

Extend or replace :func:`get_models` with your own models. They may be of the same class (as in
this example: three MLPs) or of different classes. For every model you have to be able to argue
*why* it is underfit, well fit or overfit, e.g. with train/test scores, learning curves or the
model's capacity relative to the amount of training data.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

ALPHA_GRID = np.unique(
    np.concatenate(
        [
            np.logspace(-4, 0, 6),  # overfit side
            np.logspace(0, 1.5, 10),  # transition
            [1e2],  # underfitting (constant plateau)
        ]
    )
)

# Chosen from results/cv_scores.csv (scripts/select_models.py)
SELECTED_ALPHAS = {
    "parole": {"overfit": 1e-4, "fit": ..., "underfit": ...},
    "heart_failure": {"overfit": 1e-4, "fit": ..., "underfit": ...},
    "german_credit": {"overfit": 1e-4, "fit": ..., "underfit": ...},
}


class _QuietMLPClassifier(MLPClassifier):
    """:class:`MLPClassifier` that does not warn when it stops at ``max_iter`` (intended here)."""

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            return super().fit(X, y)


def _mlp(**kwargs) -> Pipeline:
    """An MLP with standardized inputs and target."""
    return make_pipeline(StandardScaler(), _QuietMLPClassifier(**kwargs))


def make_mlp(alpha: float, seed: int = 0) -> Pipeline:
    """The MLP used for all experiments: two hidden layers of 128, trained to convergence.

    Only the L2 strength ``alpha`` varies between models, so the fit is controlled by regularization alone.

    :param alpha: L2 regularization strength.
    :param seed: Random seed for weight initialization and batch shuffling.
    :return: Unfitted pipeline of a standard scaler and the MLP.
    :rtype: Pipeline
    """
    return _mlp(
        hidden_layer_sizes=(128, 128), alpha=alpha, max_iter=3000, tol=1e-6, n_iter_no_change=100, random_state=seed
    )


def get_selected_models(seed: int = 0) -> dict[str, dict[str, BaseEstimator]]:
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
    return {
        name: {fit: make_mlp(alpha, seed) for fit, alpha in alphas.items()} for name, alphas in SELECTED_ALPHAS.items()
    }


def get_cv_models(seed: int = 0) -> dict[str, BaseEstimator]:
    """ """
    return {f"mlp_alpha_{alpha:.2e}": make_mlp(alpha, seed) for alpha in ALPHA_GRID}
