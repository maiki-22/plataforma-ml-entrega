"""Métricas de evaluación sobre las filas reservadas."""

import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error, r2_score, root_mean_squared_error


def evaluar_supervisado(reales, predicciones, tarea):
    """Calcula las métricas y la tabla de resultados de una evaluación supervisada.

    Parameters
    ----------
    reales : pandas.Series
        Valores observados en la partición de prueba.
    predicciones : array-like
        Predicciones en el mismo orden que los valores reales.
    tarea : str
        Tarea supervisada, «Clasificación» o «Regresión».

    Returns
    -------
    tuple[dict, pandas.DataFrame]
        Métricas y matriz de confusión normalizada o tabla de residuales, según la tarea.
    """
    if tarea == "Clasificación":
        etiquetas = sorted(set(reales) | set(predicciones))
        matriz = confusion_matrix(reales, predicciones, labels=etiquetas, normalize="true")
        detalle = pd.DataFrame(matriz, index=pd.Index(etiquetas, name="Real"),
                               columns=pd.Index(etiquetas, name="Predicción"))
        return {"Exactitud": accuracy_score(reales, predicciones)}, detalle
    detalle = pd.DataFrame({"Real": reales, "Predicción": predicciones, "Residual": reales - predicciones})
    return {"MAE": mean_absolute_error(reales, predicciones),
            "RMSE": root_mean_squared_error(reales, predicciones),
            "R²": r2_score(reales, predicciones)}, detalle


# Nombre utilizado en las primeras versiones del módulo.
evaluar_cart = evaluar_supervisado
