"""Ejecuciones supervisadas independientes de la interfaz."""

from dataclasses import dataclass

import pandas as pd

from core.particion import separar_datos
from evaluation.metrics import evaluar_supervisado
from evaluation.resultados import ResultadoSupervisado
from models.supervised.tree_models import ModeloCART


@dataclass
class ResultadoCART(ResultadoSupervisado):
    """Evaluación, partición y modelo de una ejecución reproducible."""

    modelo: ModeloCART
    metricas: dict
    detalle: pd.DataFrame
    profundidad: int


def ejecutar_cart(tabla, objetivo, tarea, profundidad=3, semilla=42):
    """Reserva la prueba antes de ajustar cualquier transformación.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Datos de entrada con predictores y objetivo.
    objetivo : str
        Nombre de la columna que se desea predecir.
    tarea : str
        Clasificación o Regresión.
    profundidad : int
        Profundidad máxima del árbol.
    semilla : int
        Semilla de la partición y del árbol.

    Returns
    -------
    ResultadoCART
        Modelo ajustado y evaluación fuera de entrenamiento.
    """
    particion = separar_datos(tabla, objetivo, tarea, semilla)
    modelo = ModeloCART(tarea, profundidad, semilla)
    modelo.fit(particion.predictores_entrenamiento, particion.objetivo_entrenamiento)
    predicciones = modelo.predict(particion.predictores_prueba)
    metricas, detalle = evaluar_supervisado(particion.objetivo_prueba, predicciones, tarea)
    return ResultadoCART(objetivo=objetivo, tarea=tarea, semilla=semilla, particion=particion,
                         modelo=modelo, metricas=metricas, detalle=detalle, profundidad=profundidad)
