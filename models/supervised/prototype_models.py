"""Vecinos euclidianos, prototipos LVQ1 y vecinos discriminantes DANN."""

import numpy as np
from scipy.spatial.distance import cdist
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

from models.supervised.candidato import Candidato


class LVQ1(ClassifierMixin, BaseEstimator):
    """Aprende un prototipo por clase mediante atracción y repulsión."""

    def __init__(self, tasa=0.05, epocas=20, semilla=42):
        self.semilla = semilla
        self.tasa = tasa
        self.epocas = epocas

    def fit(self, X, y):
        """Aprende un prototipo por clase mediante actualizaciones secuenciales.

        Parameters
        ----------
        X : numpy.ndarray
            Matriz numérica densa, con una observación por fila.
        y : array-like
            Valores objetivo alineados con las filas de X.

        Returns
        -------
        LVQ1
            Estimador ajustado.
        """
        self.classes_, etiquetas = np.unique(y, return_inverse=True)
        self.prototipos_ = np.array([X[etiquetas == i].mean(axis=0) for i in range(len(self.classes_))])
        azar = np.random.default_rng(self.semilla)
        for epoca in range(self.epocas):
            paso = self.tasa * (1 - epoca / self.epocas)
            for i in azar.permutation(len(X)):
                diferencias = X[i] - self.prototipos_
                ganador = np.argmin(np.sum(diferencias ** 2, axis=1))
                signo = 1 if ganador == etiquetas[i] else -1
                self.prototipos_[ganador] += signo * paso * diferencias[ganador]
        return self

    def predict_proba(self, X):
        """Puntajes relativos por cercanía al prototipo (sin calibración).

        Parameters
        ----------
        X : numpy.ndarray
            Matriz numérica densa, con una observación por fila.

        Returns
        -------
        numpy.ndarray
            Puntajes de forma (filas, clases), en el orden de classes_.
        """
        distancias = cdist(X, self.prototipos_, metric="sqeuclidean")
        pesos = np.exp(-(distancias - distancias.min(axis=1, keepdims=True)))
        return pesos / pesos.sum(axis=1, keepdims=True)

    def predict(self, X):
        """Asigna a cada fila la clase de su prototipo más cercano.

        Parameters
        ----------
        X : numpy.ndarray
            Matriz numérica densa, con una observación por fila.

        Returns
        -------
        numpy.ndarray
            Una predicción por fila.
        """
        distancias = cdist(X, self.prototipos_, metric="sqeuclidean")
        return self.classes_[distancias.argmin(axis=1)]


class DANN(ClassifierMixin, BaseEstimator):
    """Calcula una métrica discriminante alrededor de cada observación consultada."""

    def __init__(self, vecinos=3, entorno=25, epsilon=0.1):
        self.vecinos = vecinos
        self.entorno = entorno
        self.epsilon = epsilon

    def fit(self, X, y):
        """Conserva las observaciones para estimar las métricas locales durante la consulta.

        Parameters
        ----------
        X : numpy.ndarray
            Matriz numérica densa, con una observación por fila.
        y : array-like
            Valores objetivo alineados con las filas de X.

        Returns
        -------
        DANN
            Estimador ajustado.
        """
        self.datos_ = np.asarray(X)
        self.classes_, self.etiquetas_ = np.unique(y, return_inverse=True)
        return self

    def _votos(self, x):
        diferencias = self.datos_ - x
        cercanos = np.argsort(np.sum(diferencias ** 2, axis=1))[:self.entorno]
        entorno = self.datos_[cercanos]
        etiquetas = self.etiquetas_[cercanos]
        media = entorno.mean(axis=0)
        identidad = np.eye(len(x))
        intra = np.zeros_like(identidad)
        entre = np.zeros_like(identidad)
        for clase in np.unique(etiquetas):
            grupo = entorno[etiquetas == clase]
            centro = grupo.mean(axis=0)
            residuos = grupo - centro
            intra += residuos.T @ residuos / len(entorno)
            entre += len(grupo) / len(entorno) * np.outer(centro - media, centro - media)
        # W regularizada permite invertir incluso columnas constantes o redundantes.
        inversa = np.linalg.solve(intra + 1e-5 * identidad, identidad)
        metrica = inversa @ entre @ inversa + self.epsilon * inversa
        distancias = np.sum((diferencias @ metrica) * diferencias, axis=1)
        vecinos = np.argsort(distancias)[:self.vecinos]
        return np.bincount(self.etiquetas_[vecinos], minlength=len(self.classes_)) / len(vecinos)

    def predict_proba(self, X):
        """Fracción de votos por clase en la vecindad adaptada.

        Parameters
        ----------
        X : numpy.ndarray
            Matriz numérica densa, con una observación por fila.

        Returns
        -------
        numpy.ndarray
            Puntajes de forma (filas, clases), en el orden de classes_.
        """
        return np.array([self._votos(x) for x in X])

    def predict(self, X):
        """Devuelve la clase con mayor proporción de votos en la vecindad adaptada.

        Parameters
        ----------
        X : numpy.ndarray
            Matriz numérica densa, con una observación por fila.

        Returns
        -------
        numpy.ndarray
            Una predicción por fila.
        """
        return self.classes_[self.predict_proba(X).argmax(axis=1)]


def modelos_vecinos(tarea, max_vecinos, semilla=42):
    """Construye los candidatos de vecinos compatibles con la tarea y los pliegues.

    Parameters
    ----------
    tarea : str
        Tarea supervisada, «Clasificación» o «Regresión».
    max_vecinos : int
        Número de filas del pliegue de entrenamiento más pequeño.
    semilla : int
        Semilla de la ejecución para las operaciones aleatorias.

    Returns
    -------
    dict[str, Candidato]
        Candidatos identificados por su nombre visible.
    """
    candidatos = [k for k in (1, 3, 5) if k <= max_vecinos]
    vecinos = KNeighborsClassifier if tarea == "Clasificación" else KNeighborsRegressor
    modelos = {
        "k-NN": Candidato(vecinos(weights="distance", metric="euclidean"),
                          {"n_neighbors": candidatos}, escalar=True),
    }
    if tarea == "Clasificación":
        modelos.update({
            "LVQ1": Candidato(LVQ1(semilla=semilla), {"tasa": [0.01, 0.05]}, escalar=True, requiere_densa=True),
            "DANN": Candidato(DANN(), {"vecinos": candidatos}, escalar=True, requiere_densa=True),
        })
    return modelos
