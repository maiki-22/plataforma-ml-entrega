"""Entrada interactiva de la plataforma de aprendizaje automático."""

from hashlib import sha256
from pathlib import Path

import streamlit as st

from core.loader import cargar_tabla, seleccionar_ruta


DATOS = Path(__file__).resolve().parent / "data"
DELIMITADORES = {"Automático": None, "Coma": ",", "Punto y coma": ";", "Tabulación": "\t", "Barra vertical": "|"}


def mostrar_inicio() -> None:
    """Presenta la tabla y permite revisar el objetivo y la tarea."""
    st.set_page_config(page_title="Análisis de datos tabulares", page_icon="📊")
    st.title("Análisis de datos tabulares")
    origen = st.radio("Origen de los datos", ["Subir archivo", "Ejemplo local"], horizontal=True)
    if origen == "Subir archivo":
        archivo = st.file_uploader("Selecciona un archivo", type=["csv", "tsv", "parquet"])
        if archivo is None:
            return
        nombre, contenido = archivo.name, archivo.getvalue()
    else:
        nombre = st.selectbox("Archivo de ejemplo", ["flores.csv", "mediciones.tsv"])
        contenido = (DATOS / nombre).read_bytes()

    st.text(f"Archivo seleccionado: {nombre}")
    clave = sha256(contenido).hexdigest()
    separador, encabezado = None, True
    if Path(nombre).suffix.lower() != ".parquet":
        with st.expander("Opciones de lectura"):
            opcion = st.selectbox("Delimitador", list(DELIMITADORES), key=f"sep-{clave}")
            separador = DELIMITADORES[opcion]
            encabezado = st.checkbox("La primera fila contiene los nombres", True, key=f"cab-{clave}")

    try:
        tabla = cargar_tabla(nombre, contenido, separador, encabezado)
        st.subheader("Vista previa")
        st.caption(f"{len(tabla)} filas · {len(tabla.columns)} columnas · Primeras 10 filas")
        st.dataframe(tabla.head(10), hide_index=True)
        modo = st.radio("Variable objetivo", ["Automática", "Elegir columna", "Sin objetivo"],
                        horizontal=True, key=f"modo-{clave}")
        objetivo = st.selectbox("Columna objetivo", tabla.columns) if modo == "Elegir columna" else None
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
    else:
        st.text(f"{ruta.motivo} Columna: {ruta.objetivo}")
        tareas = ["Clasificación", "Regresión"]
        tarea = st.selectbox("Tarea supervisada", tareas, index=tareas.index(ruta.tarea),
                             key=f"tarea-{clave}-{ruta.objetivo}-{ruta.tarea}")
        st.info(f"Ruta seleccionada: aprendizaje supervisado · {tarea}.")


if __name__ == "__main__":
    mostrar_inicio()
