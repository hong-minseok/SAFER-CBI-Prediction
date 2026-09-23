"""Small, serializable binary beta-calibration implementation.

The calibration model is a logistic regression on ``log(p)`` and
``-log(1-p)``. Its fitted coefficients can be serialized as JSON without
pickling sklearn objects.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.special import expit
from sklearn.linear_model import LogisticRegression


_EPS = 1e-8


def _design(probabilities) -> np.ndarray:
    probabilities = np.clip(np.asarray(probabilities, dtype=float).reshape(-1), _EPS, 1 - _EPS)
    return np.column_stack((np.log(probabilities), -np.log1p(-probabilities)))


@dataclass(frozen=True)
class BetaCalibrationModel:
    coefficients: tuple[float, float]
    intercept: float

    def predict(self, probabilities) -> np.ndarray:
        design = _design(probabilities)
        return expit(design @ np.asarray(self.coefficients) + self.intercept)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "beta_abm",
            "features": ["log(p)", "-log(1-p)"],
            "coefficients": [float(value) for value in self.coefficients],
            "intercept": float(self.intercept),
            "clip_epsilon": _EPS,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "BetaCalibrationModel":
        if payload.get("method") != "beta_abm":
            raise ValueError("Unsupported calibration method")
        coefficients = payload.get("coefficients")
        if not isinstance(coefficients, list) or len(coefficients) != 2:
            raise ValueError("Beta calibration requires two coefficients")
        return cls(tuple(float(value) for value in coefficients), float(payload["intercept"]))


def fit_beta_calibration(probabilities, labels) -> BetaCalibrationModel:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(labels) == 0 or len(np.unique(labels)) != 2:
        raise ValueError("Beta calibration requires both outcome classes")
    estimator = LogisticRegression(C=1e6, solver="lbfgs", max_iter=2000)
    estimator.fit(_design(probabilities), labels)
    return BetaCalibrationModel(
        coefficients=tuple(float(value) for value in estimator.coef_[0]),
        intercept=float(estimator.intercept_[0]),
    )


def crossfit_beta_calibration(probabilities, labels, folds) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float).reshape(-1)
    labels = np.asarray(labels, dtype=int).reshape(-1)
    folds = np.asarray(folds, dtype=int).reshape(-1)
    if not (len(probabilities) == len(labels) == len(folds)):
        raise ValueError("probabilities, labels, and folds must align")
    result = np.full(len(probabilities), np.nan, dtype=float)
    for fold in np.unique(folds):
        held_out = folds == fold
        model = fit_beta_calibration(probabilities[~held_out], labels[~held_out])
        result[held_out] = model.predict(probabilities[held_out])
    if not np.isfinite(result).all():
        raise RuntimeError("Cross-fitted calibration left incomplete probabilities")
    return result
