"""P1: decision-layer correction for the JM-LLM reproduction.

The correction only ever touches validation labels and the *predicted*
distribution on the target split; test labels are used solely for reporting.

Components
----------
* ``PlattCalibrator`` -- p' = sigmoid(a * logit(p) + b), fitted on validation
* ``best_threshold``  -- macro-F1 optimal decision threshold on validation
* ``PriorShift``      -- target-prior estimation from unlabelled predictions
                         (EM: Saerens et al.; BBSE: Lipton et al.) followed by
                         posterior re-weighting
* ``calc_metrics``    -- accuracy / balanced accuracy / macro-F1 / per-class
                         recall, the four numbers requested for the paper
"""
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                             confusion_matrix, f1_score)


def calc_metrics(y_true, y_pred):
    """Class 0 = Normal, class 1 = Attack."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "acc": float(accuracy_score(y_true, y_pred)),
        "bacc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "normal_recall": float(tn / max(tn + fp, 1)),
        "attack_recall": float(tp / max(tp + fn, 1)),
    }


def _logit(p, eps=1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class PlattCalibrator:
    """Logistic recalibration of a probability score.

    By default only the temperature (slope) is fitted, so calibration does not
    absorb the validation prior; the bias term is optional because a fitted
    bias would have to be undone by the prior shift below.
    """

    def __init__(self, fit_bias=False):
        self.fit_bias = fit_bias
        self.a, self.b = 1.0, 0.0

    def fit(self, p, y):
        z = _logit(p)
        if self.fit_bias:
            lr = LogisticRegression(C=1e6, max_iter=1000)
            lr.fit(z.reshape(-1, 1), y)
            self.a = float(lr.coef_[0, 0])
            self.b = float(lr.intercept_[0])
        else:
            def nll(log_a):
                az = np.exp(log_a) * z
                return float(np.mean(np.logaddexp(0.0, az) - y * az))
            res = minimize_scalar(nll, bounds=(-3.0, 3.0), method="bounded")
            self.a = float(np.exp(res.x))
            self.b = 0.0
        return self

    def transform(self, p):
        return 1.0 / (1.0 + np.exp(-(self.a * _logit(p) + self.b)))


def best_threshold(p, y, grid=None):
    grid = np.linspace(0.02, 0.98, 97) if grid is None else grid
    scores = [f1_score(y, (p > t).astype(int), average="macro") for t in grid]
    return float(grid[int(np.argmax(scores))])


class PriorShift:
    """Estimate the target prior P(y=1) without using target labels."""

    def __init__(self, source_prior, method="em", max_iter=200, tol=1e-6):
        self.pi_s = float(source_prior)
        self.method = method
        self.max_iter = max_iter
        self.tol = tol
        self.pi_t = self.pi_s

    def _weights(self, pi_t):
        return pi_t / max(self.pi_s, 1e-9), (1.0 - pi_t) / max(1.0 - self.pi_s, 1e-9)

    def _apply(self, p, pi_t):
        w1, w0 = self._weights(pi_t)
        p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0 - 1e-12)
        num = p * w1
        return num / (num + (1.0 - p) * w0)

    def _em(self, p_target):
        pi_t = float(np.mean(p_target))
        for _ in range(self.max_iter):
            new = float(np.mean(self._apply(p_target, pi_t)))
            if abs(new - pi_t) < self.tol:
                pi_t = new
                break
            pi_t = new
        return pi_t

    def _bbse(self, p_source, y_source, p_target):
        # C[i, j] = P(pred = i | y = j), estimated on the validation split
        cm = confusion_matrix(y_source, (p_source > 0.5).astype(int),
                              labels=[0, 1], normalize="true").T
        q = np.array([np.mean(p_target <= 0.5), np.mean(p_target > 0.5)])
        try:
            pi = np.linalg.solve(cm, q)
        except np.linalg.LinAlgError:
            pi = np.linalg.lstsq(cm, q, rcond=None)[0]
        pi = np.clip(pi, 1e-3, 1.0 - 1e-3)
        pi = pi / pi.sum()
        return float(pi[1])

    def fit(self, p_source, y_source, p_target):
        if self.method == "em":
            self.pi_t = self._em(p_target)
        elif self.method == "bbse":
            self.pi_t = self._bbse(p_source, y_source, p_target)
        return self

    def apply(self, p):
        return self._apply(p, self.pi_t)
