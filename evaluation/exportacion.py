"""Publicación de los resultados completos de cada ejecución."""

import json
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory, mkdtemp

import pandas as pd

from evaluation.reportes import guardar_configuracion, reportar_comparacion
from evaluation.visualizer import guardar_agrupamiento, guardar_evaluacion_cart


@contextmanager
def _preparar_exportacion(destino):
    """Prepara el reporte en el mismo sistema de archivos para publicarlo de una vez."""
    destino.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destino, prefix=".pendiente-") as temporal:
        yield Path(temporal)


def exportar_cart(resultado, lectura, destino):
    """Publica la figura, las métricas y los parámetros del árbol.

    Parameters
    ----------
    resultado : ResultadoCART
        Árbol ajustado, configuración y evaluación de prueba.
    lectura : Lectura
        Huella del archivo y opciones de lectura utilizadas.
    destino : pathlib.Path
        Directorio donde se crean los reportes de esta ejecución.

    Returns
    -------
    pathlib.Path
        Ruta del PNG del árbol en el reporte publicado.
    """
    with _preparar_exportacion(destino) as temporal:
        figura = guardar_evaluacion_cart(resultado.detalle, resultado.tarea, temporal)
        carpeta = figura.parent
        guardar_configuracion(carpeta, lectura, objetivo=resultado.objetivo, tarea=resultado.tarea,
                              profundidad=resultado.profundidad, semilla=resultado.semilla)
        pd.Series(resultado.metricas, name="Valor").to_csv(carpeta / "metricas.csv", index_label="Métrica")
        publicada = carpeta.rename(destino / carpeta.name)
    return publicada / figura.name


def exportar_comparacion(resultado, lectura, destino):
    """Publica juntas las curvas, las métricas y la selección de hiperparámetros.

    Parameters
    ----------
    resultado : Comparacion
        Modelos validados, métricas y partición efectiva.
    lectura : Lectura
        Huella del archivo y opciones de lectura utilizadas.
    destino : pathlib.Path
        Directorio donde se crean los reportes de esta ejecución.

    Returns
    -------
    pathlib.Path
        Directorio publicado con curvas, métricas y parámetros.
    """
    with _preparar_exportacion(destino) as temporal:
        carpeta = Path(mkdtemp(prefix="comparacion-", dir=temporal))
        reportar_comparacion(resultado, carpeta)
        guardar_configuracion(carpeta, lectura, objetivo=resultado.objetivo, tarea=resultado.tarea, semilla=resultado.semilla,
                              pliegues=resultado.pliegues, modelos=list(resultado.parametros))
        resultado.resumen.to_csv(carpeta / "metricas.csv", index=False)
        (carpeta / "parametros.json").write_text(json.dumps(resultado.parametros, indent=2), encoding="utf-8")
        return carpeta.rename(destino / carpeta.name)


def exportar_grupos(resultado, lectura, destino):
    """Publica todas las figuras, etiquetas y métricas del agrupamiento.

    Parameters
    ----------
    resultado : Agrupamiento
        Grupos elegidos, métricas por K y proyección de las observaciones.
    lectura : Lectura
        Huella del archivo y opciones de lectura utilizadas.
    destino : pathlib.Path
        Directorio donde se crean los reportes de esta ejecución.

    Returns
    -------
    dict[str, pathlib.Path]
        Rutas publicadas de las figuras de codo, siluetas y proyección.
    """
    with _preparar_exportacion(destino) as temporal:
        figuras = guardar_agrupamiento(resultado, temporal)
        carpeta = figuras["Codo"].parent
        guardar_configuracion(carpeta, lectura, tarea="Agrupamiento", k_maximo=resultado.k_maximo, k=resultado.k,
                              semilla=resultado.semilla)
        publicada = carpeta.rename(destino / carpeta.name)
    return {nombre: publicada / ruta.name for nombre, ruta in figuras.items()}
