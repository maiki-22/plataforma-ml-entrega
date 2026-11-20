"""Controles para cargar la tabla activa y elegir la tarea."""

from dataclasses import replace
from hashlib import sha256
from pathlib import Path

import pandas as pd
import streamlit as st

from core.loader import Lectura, Ruta, cargar_tabla, seleccionar_ruta


DELIMITADORES = {"Automático": None, "Coma": ",", "Punto y coma": ";", "Tabulación": "\t", "Barra vertical": "|"}


def seleccionar_archivo(directorio: Path) -> tuple[str, bytes] | None:
    """Devuelve el nombre y los bytes del ejemplo o archivo seleccionado.

    Parameters
    ----------
    directorio : pathlib.Path
        Carpeta con los archivos de ejemplo.

    Returns
    -------
    tuple[str, bytes] o None
        Nombre y contenido seleccionados, o None mientras se espera una carga.
    """
    origen = st.radio("Origen de los datos", ["Ejemplo local", "Subir archivo"],
                      horizontal=True, key="origen", label_visibility="collapsed")
    if origen == "Subir archivo":
        archivo = st.file_uploader("Selecciona un archivo", type=["csv", "tsv", "parquet"])
        if archivo is None:
            return
        return archivo.name, archivo.getvalue()
    nombre = st.selectbox("Archivo de ejemplo", ["flores.csv", "mediciones.tsv", "iris_petalos.csv", "iris_medidas.tsv"], key="ejemplo")
    return nombre, (directorio / nombre).read_bytes()


def leer_archivo(nombre: str, contenido: bytes) -> tuple[pd.DataFrame, Lectura]:
    """Conserva una tabla por sesión, identificada por contenido y opciones.

    Parameters
    ----------
    nombre : str
        Nombre del archivo con su extensión.
    contenido : bytes
        Contenido completo del archivo.

    Returns
    -------
    tuple[pandas.DataFrame, Lectura]
        Tabla y clave de lectura utilizada en el estado de sesión.
    """
    clave = sha256(contenido).hexdigest()
    separador, encabezado = None, True
    extension = Path(nombre).suffix.lower()
    if extension != ".parquet":
        with st.expander("Opciones de lectura"):
            opcion = st.selectbox("Delimitador", list(DELIMITADORES), key=f"sep-{clave}")
            separador = DELIMITADORES[opcion]
            encabezado = st.checkbox("La primera fila contiene los nombres", True, key=f"cab-{clave}")
    lectura = Lectura(clave, extension, separador, encabezado)
    if st.session_state.get("lectura_tabla") != lectura:
        tabla = cargar_tabla(nombre, contenido, separador, encabezado)
        st.session_state["tabla_activa"] = tabla
        st.session_state["lectura_tabla"] = lectura
    return st.session_state["tabla_activa"], lectura


def mostrar_ruta(tabla: pd.DataFrame, clave: str) -> Ruta | None:
    """Explica la tarea seleccionada para la tabla.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Tabla de observaciones con nombres de columna únicos.
    clave : str
        Huella del archivo que identifica los controles.

    Returns
    -------
    Ruta o None
        Selección activa, o None si la columna elegida impide definir la tarea.
    """
    modo = st.radio("Variable objetivo", ["Automática", "Elegir columna", "Sin objetivo"],
                    horizontal=True, key=f"modo-{clave}")
    objetivo = st.selectbox("Columna objetivo", tabla.columns) if modo == "Elegir columna" else None
    try:
        ruta = seleccionar_ruta(tabla, objetivo, modo == "Sin objetivo")
    except ValueError as error:
        st.error(str(error))
        return

    if ruta.objetivo is None:
        mensaje = f"{ruta.motivo} Ruta seleccionada: {ruta.tarea}."
        if modo == "Automática":
            st.warning(mensaje)
        else:
            st.info(mensaje)
        return ruta

    st.text(f"{ruta.motivo} Columna: {ruta.objetivo}")
    tareas = ["Clasificación", "Regresión"]
    tarea = st.selectbox("Tarea supervisada", tareas, index=tareas.index(ruta.tarea),
                         key=f"tarea-{clave}-{ruta.objetivo}-{ruta.tarea}")
    st.info(f"Ruta seleccionada: aprendizaje supervisado · {tarea}.")
    return replace(ruta, tarea=tarea)
