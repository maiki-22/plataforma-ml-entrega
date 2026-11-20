"""Figuras comparativas que usan la reserva de prueba y los ajustes de CV."""

import json

import numpy as np
import pandas as pd
from matplotlib import colormaps
from matplotlib.colors import ListedColormap
from matplotlib.figure import Figure
from sklearn.base import clone
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits

from evaluation.visualizer import _preparar_figura, figura_regresion


def guardar_configuracion(carpeta, lectura, **opciones):
    """Identifica el archivo por contenido y documenta la lectura reproducible."""
    huella, extension, delimitador, encabezado = lectura
    datos = dict(sha256=huella, formato=extension, delimitador=delimitador,
                 encabezado=encabezado, **opciones)
    (carpeta / "configuracion.json").write_text(json.dumps(datos, ensure_ascii=False, indent=2), encoding="utf-8")


def _exportar(figura, carpeta, nombre):
    _preparar_figura(figura)
    figura.savefig(carpeta / f"{nombre}.png", dpi=300, bbox_inches="tight")


def _curvas_umbral(ajustes, prueba, reales, carpeta):
    """Compara puntajes uno contra el resto por clase y con promedio micro."""
    figura = Figure(figsize=(12, 5), layout="constrained")
    roc, pr = figura.subplots(1, 2)
    areas, puntos = [], []
    for nombre, ajuste in ajustes.items():
        clases = ajuste.classes_
        indicadores = np.asarray(reales)[:, None] == clases
        puntajes = ajuste.predict_proba(prueba)
        for indice in range(len(clases) + 1):
            micro = indice == len(clases)
            verdad = indicadores.ravel() if micro else indicadores[:, indice]
            scores = puntajes.ravel() if micro else puntajes[:, indice]
            if np.unique(verdad).size < 2:
                continue
            fpr, tpr, _ = roc_curve(verdad, scores)
            precision, recall, _ = precision_recall_curve(verdad, scores)
            clase = "Micro (global)" if micro else str(clases[indice])
            area = auc(fpr, tpr)
            areas.append({"Modelo": nombre, "Clase": clase, "AUC ROC": area})
            for curva, xs, ys in [("ROC", fpr, tpr), ("PR", recall, precision)]:
                puntos.append(pd.DataFrame({"Modelo": nombre, "Clase": clase, "Curva": curva, "x": xs, "y": ys}))
            if micro:
                roc.plot(fpr, tpr, label=f"{nombre} ({area:.2f})")
                pr.step(recall, precision, where="post", label=nombre)
    roc.plot([0, 1], [0, 1], ":", color="gray")
    roc.set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad", title="ROC micro · AUC en leyenda")
    pr.set(xlabel="Sensibilidad", ylabel="Precisión", title="Precisión y sensibilidad micro")
    for eje in (roc, pr):
        eje.legend(facecolor="#292929", labelcolor="white", fontsize=8)
    pd.DataFrame(areas).to_csv(carpeta / "auc.csv", index=False)
    if puntos:
        curvas = pd.concat(puntos)
        curvas.to_csv(carpeta / "curvas_por_clase.csv", index=False)
        # Un panel por clase mantiene legibles los nombres y los trazos.
        por_clase = curvas[curvas["Clase"] != "Micro (global)"].groupby("Clase")
        for numero, (clase, grupo) in enumerate(por_clase, start=1):
            comparativa = Figure(figsize=(11, 4.5), layout="constrained")
            eje_roc, eje_pr = comparativa.subplots(1, 2)
            for (modelo, tipo), linea in grupo.groupby(["Modelo", "Curva"]):
                if tipo == "ROC":
                    eje_roc.plot(linea.x, linea.y, label=modelo)
                else:
                    eje_pr.step(linea.x, linea.y, where="post", label=modelo)
            eje_roc.set(xlabel="Tasa de falsos positivos", ylabel="Sensibilidad", title=f"ROC · {clase}")
            eje_pr.set(xlabel="Sensibilidad", ylabel="Precisión", title=f"PR · {clase}")
            eje_roc.legend(facecolor="#292929", labelcolor="white", fontsize=8)
            _exportar(comparativa, carpeta, f"ROC y PR clase {numero}")
    _exportar(figura, carpeta, "ROC y precisión-recall")


def _aprendizaje(ajuste, x, y, carpeta, nombre, semilla):
    """Mide el error en validación con subconjuntos estratificados de entrenamiento."""
    figura = Figure(figsize=(10, 4), layout="constrained")
    tamano, parametro = figura.subplots(1, 2)
    medidas = []
    for entrenamiento, validacion in ajuste.cv:
        for fraccion in (0.4, 0.7, 1.0):
            indices = entrenamiento
            if fraccion < 1:
                try:
                    indices, _ = train_test_split(entrenamiento, train_size=fraccion,
                                                  stratify=y.iloc[entrenamiento], random_state=semilla)
                except ValueError:
                    continue
            vecinos = ajuste.best_params_.get("modelo__n_neighbors", 1)
            if len(indices) < vecinos:
                continue
            modelo = clone(ajuste.best_estimator_).fit(x.iloc[indices], y.iloc[indices])
            medidas.append((fraccion, len(indices), 1 - modelo.score(x.iloc[indices], y.iloc[indices]),
                            1 - modelo.score(x.iloc[validacion], y.iloc[validacion])))
    medias = pd.DataFrame(medidas, columns=["Fracción", "Filas", "Entrenamiento", "Validación"]).groupby("Fracción").mean()
    for tipo in ("Entrenamiento", "Validación"):
        tamano.plot(medias["Filas"], medias[tipo], marker="o", label=tipo)
    tamano.set(xlabel="Filas de ajuste (media)", ylabel="Fracción de errores", title=nombre)
    tamano.legend(facecolor="#292929", labelcolor="white")
    resultados = ajuste.cv_results_
    clave = next(iter(ajuste.best_params_))
    valores = [str(p[clave]) for p in resultados["params"]]
    parametro.plot(valores, 1 - resultados["mean_test_score"], marker="o")
    parametro.set(xlabel=clave.removeprefix("modelo__"), ylabel="Error medio en validación", title="Ajuste de hiperparámetros")
    _exportar(figura, carpeta, "Curvas de aprendizaje")


def _superficie(ajuste, x, y, nombre, carpeta):
    if x.shape[1] != 2 or len(x.select_dtypes("number").columns) != 2:
        return
    rangos = x.replace([np.inf, -np.inf], np.nan).agg(["min", "max"])
    if not np.isfinite(rangos).all().all() or (rangos.iloc[0] == rangos.iloc[1]).any():
        return
    a, b = np.meshgrid(*(np.linspace(*rangos[col], 55) for col in x))
    malla = pd.DataFrame(np.column_stack([a.ravel(), b.ravel()]), columns=x.columns)
    clases = pd.Index(ajuste.classes_)
    zonas = clases.get_indexer(ajuste.predict(malla)).reshape(a.shape)
    figura = Figure(figsize=(7, 4), layout="constrained")
    eje = figura.subplots()
    colores = colormaps["viridis"](np.linspace(0.15, 0.9, len(clases)))
    eje.contourf(a, b, zonas, levels=np.arange(len(clases)+1)-0.5, cmap=ListedColormap(colores), alpha=0.6)
    for clase, color in zip(clases, colores):
        puntos = x.loc[y == clase]
        eje.scatter(*puntos.to_numpy().T, label=str(clase), color=color, edgecolor="white", s=18)
    eje.set(xlabel=x.columns[0], ylabel=x.columns[1], title=f"Regiones de decisión · {nombre}")
    eje.legend(facecolor="#292929", labelcolor="white")
    _exportar(figura, carpeta, "Superficie de decisión")


def reportar_comparacion(resultado, carpeta):
    """Exporta gráficos de prueba y diagnóstico del modelo elegido por CV."""
    nombre = max(resultado.ajustes, key=lambda n: resultado.ajustes[n].best_score_)
    mejor = resultado.ajustes[nombre]
    with threadpool_limits(limits=1):
        if resultado.tarea == "Clasificación":
            particion = resultado.particion
            _curvas_umbral(resultado.ajustes, particion.predictores_prueba, particion.objetivo_prueba, carpeta)
            predictores = particion.predictores_entrenamiento
            etiquetas = particion.objetivo_entrenamiento
            _aprendizaje(mejor, predictores, etiquetas, carpeta, nombre, resultado.semilla)
            _superficie(mejor, predictores, etiquetas, nombre, carpeta)
        else:
            for modelo, detalle in resultado.detalles.items():
                figura = figura_regresion(detalle, f"{resultado.objetivo} · {modelo}")
                _exportar(figura, carpeta, f"Regresión {modelo}")
