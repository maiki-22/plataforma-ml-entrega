"""Reserva reproducible de filas para las tareas supervisadas."""

from math import ceil

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def separar_datos(tabla, objetivo, tarea, semilla=42):
    """Genera la misma reserva de prueba para todos los candidatos."""
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
    return x_entrenar, x_probar, y_entrenar, y_probar, omitidas
