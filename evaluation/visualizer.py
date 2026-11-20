"""Figuras del diagnóstico exploratorio para pantalla e informe."""

from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from core.eda import Diagnostico


def guardar_figuras_eda(tabla: pd.DataFrame, diagnostico: Diagnostico, destino: Path) -> dict[str, Path]:
    """Guarda mapas anotados y la dispersión del par con mayor |r|.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Observaciones usadas en el diagnóstico.
    diagnostico : Diagnostico
        Matrices y pares calculados sobre la tabla.
    destino : pathlib.Path
        Carpeta de reportes. Cada análisis genera un subdirectorio propio.

    Returns
    -------
    dict
        Rutas a figuras PNG de 300 DPI, identificadas por título.
    """
    figuras = {}
    if not diagnostico.pearson.notna().any().any():
        return figuras
    destino.mkdir(parents=True, exist_ok=True)
    carpeta = Path(mkdtemp(prefix="eda-", dir=destino))
    nombres = diagnostico.pearson.columns[diagnostico.pearson.notna().any()][:8]
    for metodo, matriz in [("Pearson", diagnostico.pearson), ("Spearman", diagnostico.spearman)]:
        figura = Figure(figsize=(8, 6), layout="constrained")
        eje = figura.subplots()
        sns.heatmap(matriz.loc[nombres, nombres], vmin=-1, vmax=1, cmap="vlag", annot=True,
                    fmt=".2f", square=True, ax=eje, cbar_kws={"label": "Correlación"})
        eje.set_title(f"Correlación de {metodo}")
        eje.tick_params(axis="y", labelrotation=0)
        ruta = carpeta / f"{metodo.lower()}.png"
        figura.savefig(ruta, dpi=300, bbox_inches="tight")
        figuras[metodo] = ruta

    if diagnostico.pares.empty:
        return figuras

    primera, segunda = diagnostico.pares.iloc[0][["Variable A", "Variable B"]]
    puntos = tabla[[primera, segunda]].replace([np.inf, -np.inf], np.nan).dropna()
    figura = Figure(figsize=(7, 4), layout="constrained")
    eje = figura.subplots()
    eje.scatter(puntos[primera], puntos[segunda], alpha=0.65)
    eje.set(xlabel=primera, ylabel=segunda, title="Dispersión del par con mayor |r|")
    ruta = carpeta / "dispersion.png"
    figura.savefig(ruta, dpi=300, bbox_inches="tight")
    figuras["Dispersión"] = ruta
    return figuras
