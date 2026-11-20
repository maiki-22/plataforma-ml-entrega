"""Diagnóstico descriptivo de tablas antes del preprocesamiento."""

from dataclasses import dataclass

import numpy as np
import pandas as pd


UMBRAL_CORRELACION = 0.85
UMBRAL_VARIANZA_BAJA = 1e-6
UMBRAL_ASIMETRIA = 1.0


@dataclass
class Diagnostico:
    """Tablas del análisis exploratorio, calculadas sobre los datos observados."""

    estructura: pd.DataFrame
    descriptivos: pd.DataFrame
    covarianza: pd.DataFrame
    pearson: pd.DataFrame
    spearman: pd.DataFrame
    pares: pd.DataFrame


def analizar_tabla(tabla: pd.DataFrame) -> Diagnostico:
    """Calcula el diagnóstico estructural, univariado y bivariado.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Tabla cargada con nombres de columna únicos.

    Returns
    -------
    Diagnostico
        Estadísticas por variable y asociaciones sobre pares de valores finitos.
    """
    estructura = pd.DataFrame({
        "Tipo": tabla.dtypes.astype(str),
        "Faltantes": tabla.isna().sum(),
        "Faltantes (%)": tabla.isna().mean() * 100,
    })
    numericas = tabla.select_dtypes(include="number", exclude="timedelta")
    numericas = numericas.replace([np.inf, -np.inf], np.nan)
    cardinalidad = {}
    for nombre in tabla.columns.difference(numericas.columns):
        valores = tabla[nombre].dropna()
        try:
            cardinalidad[nombre] = valores.nunique()
        except TypeError:
            # Parquet también admite columnas con listas o registros anidados.
            cardinalidad[nombre] = valores.map(repr).nunique()
    estructura["Valores distintos (no numéricas)"] = pd.Series(cardinalidad, dtype="Int64")
    resumen = pd.DataFrame()
    if len(numericas.columns):
        resumen = numericas.describe().T.set_axis(
            ["Válidos", "Media", "Desv. estándar", "Mínimo", "Q1", "Mediana", "Q3", "Máximo"], axis=1
        )
        rango = resumen["Q3"] - resumen["Q1"]
        fuera = numericas.lt(resumen["Q1"] - 1.5 * rango) | numericas.gt(resumen["Q3"] + 1.5 * rango)
        resumen["Atípicos (IQR)"] = fuera.sum()
        resumen["Varianza"] = numericas.var()
        resumen["Varianza nula"] = resumen["Varianza"].eq(0)
        resumen["Varianza baja"] = resumen["Varianza"].between(0, UMBRAL_VARIANZA_BAJA, inclusive="right")
        resumen["Asimetría"] = numericas.skew()
        resumen["Asimetría marcada"] = resumen["Asimetría"].abs() > UMBRAL_ASIMETRIA

    pearson = numericas.corr()
    pares = []
    for i, primera in enumerate(pearson.columns):
        for segunda, coeficiente in pearson[primera].iloc[i + 1:].dropna().items():
            pares.append((primera, segunda, coeficiente))
    pares = pd.DataFrame(pares, columns=["Variable A", "Variable B", "Pearson"])
    pares = pares.sort_values("Pearson", key=lambda valores: valores.abs(), ascending=False)
    covarianza = numericas.cov(min_periods=2) if len(tabla) >= 2 else pearson * np.nan
    return Diagnostico(estructura, resumen, covarianza, pearson, numericas.corr(method="spearman"), pares)
