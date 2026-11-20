"""Splines adaptativos y modelos aditivos de regresión."""

from pygam import LinearGAM
from pymars import Earth
from sklearn.base import BaseEstimator, RegressorMixin

from models.supervised.candidato import Candidato


class RegresionGAM(RegressorMixin, BaseEstimator):
    """Suavizado aditivo mediante un término P-spline por predictor."""

    def __init__(self, lam=1.0):
        self.lam = lam

    def fit(self, X, y):
        """Ajusta un modelo aditivo a partir de términos spline por predictor.

        Parameters
        ----------
        X : numpy.ndarray
            Matriz numérica densa, con una observación por fila.
        y : array-like
            Valores objetivo alineados con las filas de X.

        Returns
        -------
        RegresionGAM
            Estimador ajustado.
        """
        self.ajuste_ = LinearGAM(n_splines=6, lam=self.lam).fit(X, y)
        return self

    def predict(self, X):
        """Evalúa el modelo aditivo en nuevas observaciones.

        Parameters
        ----------
        X : numpy.ndarray
            Matriz numérica densa, con una observación por fila.

        Returns
        -------
        numpy.ndarray
            Una predicción por fila.
        """
        return self.ajuste_.predict(X)


def modelos_spline():
    """MARS con bisagras y poda GCV, y GAM con suavidad ajustable.

    Returns
    -------
    dict[str, Candidato]
        Candidatos identificados por su nombre visible.
    """
    return {
        "MARS": Candidato(Earth(max_degree=1), {"max_terms": [5, 9]}, escalar=False, requiere_densa=True),
        "GAM": Candidato(RegresionGAM(), {"lam": [0.1, 1, 10]}, escalar=False, requiere_densa=True),
    }
