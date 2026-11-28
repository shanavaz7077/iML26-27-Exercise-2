"""Models ranging from underfit to overfit.

Extend or replace :func:`get_models` with your own models. They may be of the same class (as in
this example: three MLPs) or of different classes. For every model you have to be able to argue
*why* it is underfit, well fit or overfit, e.g. with train/test scores, learning curves or the
model's capacity relative to the amount of training data.
"""

from __future__ import annotations

import warnings

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor


class SubsampledRegressor(RegressorMixin, BaseEstimator):
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
        return self

    def predict(self, X):
        return self.estimator_.predict(X)


class _QuietMLPRegressor(MLPRegressor):
    """:class:`MLPRegressor` that does not warn when it stops at ``max_iter`` (intended here)."""

    def fit(self, X, y):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            return super().fit(X, y)


def _mlp(**kwargs) -> TransformedTargetRegressor:
    """An MLP with standardized inputs and target."""
    return TransformedTargetRegressor(
        regressor=make_pipeline(StandardScaler(), _QuietMLPRegressor(**kwargs)),
        transformer=StandardScaler(),
    )


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
    return {
        "mlp_underfit": _mlp(hidden_layer_sizes=(1,), max_iter=200, random_state=seed),
        "mlp_fit": _mlp(hidden_layer_sizes=(64, 64), early_stopping=True, max_iter=500, random_state=seed),
        "mlp_overfit": SubsampledRegressor(
            # tol=0 and n_iter_no_change=max_iter disable the convergence check, so all epochs run.
            _mlp(
                hidden_layer_sizes=(256, 256),
                alpha=0.0,
                max_iter=10_000,
                tol=0.0,
                n_iter_no_change=10_000,
                random_state=seed,
            ),
            n_samples=20,
            random_state=seed,
        ),
    }


def get_tree_depth_candidates() -> dict[int | None, DecisionTreeRegressor]:
    """Return unfitted decision trees for a controlled maximum-depth sweep."""
    return {
        depth: DecisionTreeRegressor(max_depth=depth, random_state=0)
        for depth in (1, 2, 3, 4, 6, 8, 12, None)
    }
