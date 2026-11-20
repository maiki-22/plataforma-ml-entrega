"""Splines adaptativos y modelos aditivos de regresión."""

from pygam import LinearGAM
from pymars import Earth
from sklearn.base import BaseEstimator, RegressorMixin


class RegresionGAM(RegressorMixin, BaseEstimator):
    """Suavizado aditivo mediante un término P-spline por predictor."""

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y):
        self.ajuste_ = LinearGAM(n_splines=6, lam=self.lam).fit(X, y)
        return self

    def predict(self, X):
        return self.ajuste_.predict(X)


def modelos_spline():
    """MARS con bisagras y poda GCV, y GAM con suavidad ajustable."""
    return {"MARS": (Earth(max_degree=1), {"max_terms": [5, 9]}),
            "GAM": (RegresionGAM(), {"lam": [0.1, 1, 10]})}
