"""Contrato de los modelos de la plataforma."""

from abc import ABC, abstractmethod

import joblib


class ModeloBase(ABC):
    """Interfaz común para entrenar, predecir, evaluar y guardar modelos."""

    @abstractmethod
    def fit(self, predictores, objetivo):
        """Ajusta el modelo con observaciones de entrenamiento.

        Parameters
        ----------
        predictores : pandas.DataFrame
            Filas con las columnas predictoras originales.
        objetivo : pandas.Series
            Valores conocidos, en el mismo orden que los predictores.

        Returns
        -------
        ModeloBase
            La propia instancia ajustada con las filas de entrenamiento.
        """

    @abstractmethod
    def predict(self, predictores):
        """Predice el objetivo para nuevas observaciones.

        Parameters
        ----------
        predictores : pandas.DataFrame
            Filas con las columnas predictoras originales.

        Returns
        -------
        numpy.ndarray
            Predicción para cada fila en el orden recibido.
        """

    @abstractmethod
    def score(self, predictores, objetivo):
        """Evalúa el modelo contra objetivos conocidos.

        Parameters
        ----------
        predictores : pandas.DataFrame
            Filas con las columnas predictoras originales.
        objetivo : pandas.Series
            Valores conocidos, en el mismo orden que los predictores.

        Returns
        -------
        float
            Exactitud en clasificación o coeficiente de determinación en regresión.
        """

    def save(self, ruta):
        """Guarda el modelo y sus transformaciones ajustadas con joblib.

        Parameters
        ----------
        ruta : str o pathlib.Path
            Archivo de destino para el modelo ajustado y sus transformaciones.
        """
        joblib.dump(self, ruta)
