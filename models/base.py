"""Contrato de los modelos de la plataforma."""

from abc import ABC, abstractmethod

import joblib


class ModeloBase(ABC):
    """Interfaz común para entrenar, predecir, evaluar y guardar modelos."""

    @abstractmethod
    def fit(self, predictores, objetivo):
        """Ajusta el modelo con observaciones de entrenamiento."""

    @abstractmethod
    def predict(self, predictores):
        """Predice el objetivo para nuevas observaciones."""

    @abstractmethod
    def score(self, predictores, objetivo):
        """Evalúa el modelo contra objetivos conocidos."""

    def save(self, ruta):
        """Guarda el modelo y sus transformaciones ajustadas con joblib."""
        joblib.dump(self, ruta)
