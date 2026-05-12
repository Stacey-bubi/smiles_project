from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class HallucinationProbe(nn.Module):
    """Linear probe for hallucination detection..
    """

    _PCA_COMPONENTS = 64
    _C_GRID = [0.001, 0.01, 0.1, 1.0, 10.0]
    _CV_FOLDS = 3

    def __init__(self) -> None:
        super().__init__()
        self._pipeline: Pipeline | None = None
        self._threshold: float = 0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Use fit / predict / predict_proba.")

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationProbe":
        """Fit scaler, PCA and logistic regression with internal C selection"""
        n_comp = min(self._PCA_COMPONENTS, X.shape[0] - 1, X.shape[1])
        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=n_comp, random_state=42)),
            ("clf", LogisticRegressionCV(
                Cs=self._C_GRID,
                cv=self._CV_FOLDS,
                class_weight="balanced",
                scoring="roc_auc",
                solver="lbfgs",
                max_iter=3000,
                random_state=42,
                n_jobs=-1,
            )),
        ])
        self._pipeline.fit(X, y)
        return self

    def fit_hyperparameters(
        self, X_val: np.ndarray, y_val: np.ndarray
    ) -> "HallucinationProbe":
        """Tune the decision threshold on a validation set to maximise F1"""
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold, best_f1 = 0.5, -1.0
        for t in candidates:
            score = f1_score(y_val, (probs >= t).astype(int), zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(t)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict binary  labels using the tuned decision threshold"""
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Rreturn class probability estimates of shape ``(n_samples, 2)``"""
        return self._pipeline.predict_proba(X)
