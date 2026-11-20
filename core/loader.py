"""Lectura de tablas y selección del flujo de aprendizaje."""

import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path

import pandas as pd
import pyarrow as pa


NOMBRES_OBJETIVO = {"target", "objetivo", "clase", "class", "label", "etiqueta", "y"}


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
        return Ruta(None, "Agrupamiento (K-Means)", "Se eligió trabajar sin objetivo.")
    motivo = "Objetivo indicado manualmente."
    if objetivo is None:
        candidatas = [c for c in tabla if c.casefold() in NOMBRES_OBJETIVO]
        if len(candidatas) != 1:
            motivo = ("Hay varias candidatas: " + ", ".join(candidatas)
                      if candidatas else "Ningún nombre identifica una columna objetivo.")
            return Ruta(None, "Agrupamiento (K-Means)", motivo)
        objetivo = candidatas[0]
        motivo = "Objetivo detectado por el nombre de la columna."
    if objetivo not in tabla:
        raise ValueError(f"La columna objetivo «{objetivo}» no existe.")
    serie = tabla[objetivo]
    if serie.isna().all():
        raise ValueError("La columna objetivo está vacía. Elige otra o selecciona «Sin objetivo».")
    numerica = pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)
    return Ruta(objetivo, "Regresión" if numerica else "Clasificación", motivo)
