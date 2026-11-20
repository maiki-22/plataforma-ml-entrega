"""Ejecuciones supervisadas independientes de la interfaz."""

from dataclasses import dataclass

import pandas as pd

from core.particion import separar_datos
from evaluation.metrics import evaluar_supervisado
from models.supervised.tree_models import ModeloCART


@dataclass
class ResultadoCART:
    """Evaluación, partición y modelo de una ejecución reproducible."""

    modelo: ModeloCART
    metricas: dict
    detalle: pd.DataFrame
    entrenamiento: int
    prueba: int
    omitidas: int


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
    x_entrenar, x_probar, y_entrenar, y_probar, omitidas = separar_datos(tabla, objetivo, tarea, semilla)
    modelo = ModeloCART(tarea, profundidad, semilla).fit(x_entrenar, y_entrenar)
    predicciones = modelo.predict(x_probar)
    metricas, detalle = evaluar_supervisado(y_probar, predicciones, tarea)
    return ResultadoCART(modelo, metricas, detalle, len(y_entrenar), len(y_probar), omitidas)
