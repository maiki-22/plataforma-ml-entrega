"""Transformaciones ajustables dentro del entrenamiento de cada modelo."""

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler


def _numericas(tabla):
    valores = tabla.to_numpy(dtype=float, na_value=np.nan, copy=True)
    valores[~np.isfinite(valores)] = np.nan
    return valores


def _categoricas(tabla):
    # La marca de ausencia queda separada de cualquier valor observado.
    texto = tabla.astype("string").radd("valor=").fillna("ausente")
    return texto.to_numpy(dtype=object)


def crear_preprocesador(entrenamiento: pd.DataFrame, *, escalar: bool) -> ColumnTransformer:
    """Define la preparación de predictores a partir del esquema de entrenamiento.

    Parameters
    ----------
    entrenamiento : pandas.DataFrame
        Predictores sin la columna objetivo.
    escalar : bool
        Activa la estandarización numérica para modelos sensibles a la escala.

    Returns
    -------
    ColumnTransformer
        Transformador sin ajustar. Su ajuste se realiza dentro del pipeline.
    """
    numericas = entrenamiento.select_dtypes(include="number", exclude="timedelta").columns.tolist()
    categoricas = [columna for columna in entrenamiento if columna not in numericas]
    pasos = [
        ("normalizar", FunctionTransformer(_numericas, feature_names_out="one-to-one")),
        ("imputar", SimpleImputer(strategy="median", keep_empty_features=True)),
    ]
    if escalar:
        pasos.append(("escalar", StandardScaler()))
    categorias = Pipeline([
        ("normalizar", FunctionTransformer(_categoricas, feature_names_out="one-to-one")),
        ("codificar", OneHotEncoder(handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("numericas", Pipeline(pasos), numericas),
        ("categoricas", categorias, categoricas),
    ])
