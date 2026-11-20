"""Reserva reproducible de filas para las tareas supervisadas."""

from math import ceil
from typing import NamedTuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class Particion(NamedTuple):
    """Filas utilizadas en cada etapa, con los índices y el orden del muestreo."""

    predictores_entrenamiento: pd.DataFrame
    predictores_prueba: pd.DataFrame
    objetivo_entrenamiento: pd.Series
    objetivo_prueba: pd.Series
    filas_omitidas: int


def separar_datos(tabla: pd.DataFrame, objetivo: str, tarea: str, semilla: int = 42) -> Particion:
    """Reserva las filas de prueba antes de ajustar los modelos.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Observaciones con predictores y columna objetivo.
    objetivo : str
        Nombre de la columna que se desea predecir.
    tarea : str
        Clasificación usa muestreo estratificado. Regresión requiere objetivos finitos.
    semilla : int
        Semilla del muestreo.

    Returns
    -------
    Particion
        Predictores y objetivos alineados por fila, más el número de objetivos ausentes.
    """
    etiquetas = tabla[objetivo].dropna()
    omitidas = len(tabla) - len(etiquetas)
    predictores = tabla.loc[tabla[objetivo].notna()].drop(columns=objetivo)
    if predictores.shape[1] == 0:
        raise ValueError("Se necesita al menos una columna predictora además del objetivo.")
    if len(etiquetas) < 4:
        raise ValueError("Se necesitan al menos cuatro filas con objetivo.")
    reserva = max(2, ceil(len(etiquetas) * 0.2))
    estratos = None
    if tarea == "Clasificación":
        etiquetas = etiquetas.astype(str)
        clases = etiquetas.value_counts()
        if len(clases) < 2 or clases.min() < 2:
            raise ValueError("Selecciona un objetivo con dos clases o más y al menos dos filas por clase.")
        reserva = max(reserva, len(clases))
        estratos = etiquetas
    else:
        etiquetas = pd.to_numeric(etiquetas, errors="coerce")
        if not np.isfinite(etiquetas).all():
            raise ValueError("La regresión requiere valores objetivo numéricos y finitos.")
    x_entrenar, x_probar, y_entrenar, y_probar = train_test_split(
        predictores, etiquetas, test_size=reserva, stratify=estratos, random_state=semilla)
    return Particion(x_entrenar, x_probar, y_entrenar, y_probar, omitidas)
