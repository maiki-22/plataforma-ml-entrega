"""Figuras del diagnóstico exploratorio para pantalla e informe."""

from pathlib import Path
from tempfile import mkdtemp

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.figure import Figure

from core.eda import Diagnostico


def _preparar_figura(figura):
    """Aplica el contraste del tablero a los ejes y a la barra de color."""
    figura.set_facecolor("#212121")
    for eje in figura.axes:
        eje.set_facecolor("#292929")
        eje.tick_params(colors="#D4D4D4")
        eje.title.set_color("#ECECEC")
        eje.xaxis.label.set_color("#ECECEC")
        eje.yaxis.label.set_color("#ECECEC")
        for borde in eje.spines.values():
            borde.set_edgecolor("#595959")


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
        _preparar_figura(figura)
        figura.savefig(ruta, dpi=300, bbox_inches="tight")
        figuras[metodo] = ruta

    if diagnostico.pares.empty:
        return figuras

    primera, segunda = diagnostico.pares.iloc[0][["Variable A", "Variable B"]]
    puntos = tabla[[primera, segunda]].replace([np.inf, -np.inf], np.nan).dropna()
    figura = Figure(figsize=(7, 4), layout="constrained")
    eje = figura.subplots()
    eje.scatter(puntos[primera], puntos[segunda], color="#D4D4D4", alpha=0.75)
    eje.set(xlabel=primera, ylabel=segunda, title="Dispersión del par con mayor |r|")
    _preparar_figura(figura)
    ruta = carpeta / "dispersion.png"
    figura.savefig(ruta, dpi=300, bbox_inches="tight")
    figuras["Dispersión"] = ruta
    return figuras


def guardar_evaluacion_cart(detalle, tarea, destino):
    """Guarda la evaluación de prueba como PNG de 300 DPI en un directorio propio."""
    destino.mkdir(parents=True, exist_ok=True)
    carpeta = Path(mkdtemp(prefix="cart-", dir=destino))
    figura = Figure(figsize=(7, 5), layout="constrained")
    eje = figura.subplots()
    if tarea == "Clasificación":
        sns.heatmap(detalle, annot=len(detalle) <= 20, fmt=".0%", vmin=0, vmax=1,
                    cmap="Blues", ax=eje, cbar_kws={"label": "Proporción por clase real"})
        eje.set_title("Matriz de confusión en prueba")
    else:
        eje.scatter(detalle["Real"], detalle["Predicción"], color="#D4D4D4")
        limites = [detalle[["Real", "Predicción"]].min().min(), detalle[["Real", "Predicción"]].max().max()]
        eje.plot(limites, limites, linestyle="--", color="#8AB6C1", label="Predicción exacta")
        eje.set(xlabel="Valor real", ylabel="Predicción", title="Predicciones en prueba")
        eje.legend(facecolor="#292929", labelcolor="#ECECEC")
    _preparar_figura(figura)
    ruta = carpeta / "evaluacion.png"
    figura.savefig(ruta, dpi=300, bbox_inches="tight")
    return ruta


def guardar_agrupamiento(resultado, destino):
    """Exporta codo, siluetas individuales, PCA y asignaciones por fila."""
    destino.mkdir(parents=True, exist_ok=True)
    carpeta = Path(mkdtemp(prefix="kmeans-", dir=destino))
    resultado.metricas.to_csv(carpeta / "metricas.csv", index=False)
    resultado.filas.to_csv(carpeta / "asignaciones.csv", index_label="Fila")
    curva = resultado.metricas.groupby("K")["Inercia"].min()
    figuras = {}
    for titulo in ("Codo", "Siluetas", "Proyección"):
        figura = Figure(figsize=(7, 4), layout="constrained")
        eje = figura.subplots()
        if titulo == "Codo":
            eje.plot(curva.index, curva, marker="o", color="#ADD8E6")
            eje.axvline(resultado.k, linestyle="--", color="#E6BA80")
            eje.set(xlabel="Número de grupos", ylabel="Inercia", title=f"K seleccionado = {resultado.k}")
        elif titulo == "Siluetas":
            posicion = 0
            for grupo, filas in resultado.filas.groupby("Grupo"):
                valores = filas["Silueta"].sort_values()
                eje.barh(np.arange(len(filas)) + posicion, valores, height=1)
                eje.text(-0.95, posicion + len(filas) / 2, str(grupo), color="white")
                posicion += len(filas) + 1
            eje.axvline(resultado.filas["Silueta"].mean(), color="white", linestyle="--")
            eje.set(xlim=(-1, 1), xlabel="Silueta individual (línea: promedio)", ylabel="Grupo", yticks=[])
        else:
            for grupo, filas in resultado.filas.groupby("Grupo"):
                eje.scatter(filas["CP1"], filas["CP2"], label=str(grupo), alpha=0.7)
            eje.scatter(*resultado.centroides.T, marker="X", color="white", edgecolors="black", label="Centroides")
            eje.set(xlabel="Componente principal 1", ylabel="Componente principal 2")
            eje.legend(facecolor="#292929", labelcolor="white")
        _preparar_figura(figura)
        ruta = carpeta / f"{titulo.lower()}.png"
        figura.savefig(ruta, dpi=300)
        figuras[titulo] = ruta
    return figuras
