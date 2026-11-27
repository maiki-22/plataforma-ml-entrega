"""Lectura de tablas y selección del flujo de aprendizaje."""

import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
from typing import NamedTuple

import numpy as np
import pandas as pd
import pyarrow as pa


NOMBRES_OBJETIVO = {
    "target", "objetivo", "clase", "class", "label", "etiqueta", "especie", "species", "respuesta", "outcome", "y",
}
AGRUPAMIENTO = "Agrupamiento (K-Means)"


class Lectura(NamedTuple):
    """Huella del archivo y opciones que determinan la tabla leída."""

    huella: str
    extension: str
    separador: str | None
    encabezado: bool


def _leer_texto(contenido, separador, encabezado):
    try:
        texto = contenido.decode("utf-8-sig")
    except UnicodeDecodeError:
        texto = contenido.decode("latin-1")
    if "\x00" in texto:
        raise ValueError("El archivo contiene bytes incompatibles con una tabla de texto.")
    if separador is None:
        try:
            separador = csv.Sniffer().sniff(texto[:8192], delimiters=",;\t|").delimiter
        except csv.Error:
            separador = ","
    lector = csv.reader(StringIO(texto, newline=""), delimiter=separador, strict=True)
    filas = (fila for fila in lector if fila)
    primera = next(filas, [])
    if encabezado:
        _validar_nombres(primera)
    for fila in filas:
        if len(fila) != len(primera):
            raise ValueError(f"La línea {lector.line_num} tiene un número distinto de campos.")
    tabla = pd.read_csv(StringIO(texto), sep=separador, header=0 if encabezado else None)
    if not encabezado:
        tabla.columns = [f"columna_{i + 1}" for i in range(len(tabla.columns))]
    return tabla


def _validar_nombres(nombres):
    if any(not nombre for nombre in nombres) or len(nombres) != len(set(nombres)):
        raise ValueError("Cada columna necesita un nombre único y no vacío.")


def cargar_tabla(nombre, contenido, separador=None, encabezado=True):
    """Lee un archivo y valida que contenga una tabla con datos.

    Parameters
    ----------
    nombre : str
        Nombre con extensión CSV, TSV o Parquet.
    contenido : bytes
        Archivo completo.
    separador : str, optional
        Delimitador explícito para texto. Por defecto se infiere en CSV.
    encabezado : bool
        Indica si la primera fila de texto contiene los nombres.

    Returns
    -------
    pandas.DataFrame
        Tabla con nombres de columna únicos.
    """
    extension = Path(nombre).suffix.lower()
    if not contenido:
        raise ValueError("El archivo está vacío.")
    try:
        if extension == ".parquet":
            tabla = pd.read_parquet(BytesIO(contenido))
        elif extension in {".csv", ".tsv"}:
            if extension == ".tsv" and separador is None:
                separador = "\t"
            tabla = _leer_texto(contenido, separador, encabezado)
        else:
            raise ValueError("Selecciona un archivo CSV, TSV o Parquet.")
    except (csv.Error, pd.errors.ParserError, pd.errors.EmptyDataError, pa.ArrowException, OSError) as error:
        raise ValueError("No se pudo leer la tabla. Revisa el formato y el delimitador.") from error
    if tabla.empty:
        raise ValueError("La tabla necesita al menos una columna y una fila de datos.")
    tabla.columns = [str(nombre).strip() for nombre in tabla.columns]
    _validar_nombres(tabla.columns)
    return tabla


@dataclass(frozen=True)
class Ruta:
    """Tarea seleccionada y motivo que se mostrará en la interfaz."""

    objetivo: str | None
    tarea: str
    motivo: str


def _parece_etiquetada(serie):
    """Comprueba que haya pocas clases y que todas se repitan."""
    frecuencias = serie.value_counts()
    frecuencias = frecuencias[frecuencias > 0]
    return 2 <= len(frecuencias) <= min(20, frecuencias.sum() // 3) and frecuencias.min() >= 2


def _detectar_objetivo(tabla):
    """Sugiere un objetivo por nombre o por la estructura de la última columna."""
    candidatas = [
        nombre for nombre in tabla
        if nombre.casefold() in NOMBRES_OBJETIVO or nombre.casefold().endswith(("_target", "_objetivo"))
    ]
    if len(candidatas) > 1:
        return None, "Hay varias candidatas: " + ", ".join(candidatas) + "."
    if candidatas:
        return candidatas[0], "Objetivo sugerido por el nombre de la columna."
    if tabla.shape[1] > 1:
        ultima = tabla.iloc[:, -1]
        categorica = (
            isinstance(ultima.dtype, pd.CategoricalDtype)
            or pd.api.types.infer_dtype(ultima, skipna=True) in {"string", "boolean"}
        )
        if categorica and _parece_etiquetada(ultima):
            return ultima.name, "Objetivo sugerido porque la última columna contiene pocas categorías repetidas."
    return None, "No se encontró una candidata clara a objetivo."


def seleccionar_ruta(tabla, objetivo=None, sin_objetivo=False):
    """Determina la tarea a partir del objetivo explícito o de una candidata.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Datos validados.
    objetivo : str, optional
        Columna elegida manualmente.
    sin_objetivo : bool
        Activa explícitamente el agrupamiento.

    Returns
    -------
    Ruta
        Objetivo, tipo de tarea sugerida y explicación.
    """
    if sin_objetivo:
        return Ruta(None, AGRUPAMIENTO, "Se eligió trabajar sin objetivo.")
    automatica = objetivo is None
    motivo = "Objetivo indicado manualmente."
    if automatica:
        objetivo, motivo = _detectar_objetivo(tabla)
        if objetivo is None:
            return Ruta(None, AGRUPAMIENTO, motivo)
    if objetivo not in tabla:
        raise ValueError(f"La columna objetivo «{objetivo}» no existe.")
    valores = tabla[objetivo].dropna()
    numerica = pd.api.types.is_numeric_dtype(valores) and not pd.api.types.is_bool_dtype(valores)
    finitos = np.isfinite(valores) if numerica else None
    if valores.empty or (numerica and not finitos.any()):
        mensaje = f"La columna objetivo «{objetivo}» está vacía o no contiene valores finitos."
        if automatica:
            return Ruta(None, AGRUPAMIENTO, mensaje)
        raise ValueError(mensaje + " Elige otra o selecciona «Sin objetivo».")
    if numerica and finitos.all() and (valores % 1 == 0).all() and _parece_etiquetada(valores):
        return Ruta(objetivo, "Clasificación", motivo + " Sus valores enteros se repiten como códigos de clase.")
    return Ruta(objetivo, "Regresión" if numerica else "Clasificación", motivo)
