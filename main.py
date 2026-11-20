"""Entrada interactiva de la plataforma de aprendizaje automático."""

from hashlib import sha256
from pathlib import Path

import streamlit as st

from core.eda import UMBRAL_CORRELACION, analizar_tabla
from evaluation.visualizer import guardar_figuras_eda
from core.loader import cargar_tabla, seleccionar_ruta


DATOS = Path(__file__).resolve().parent / "data"
DELIMITADORES = {"Automático": None, "Coma": ",", "Punto y coma": ";", "Tabulación": "\t", "Barra vertical": "|"}


def mostrar_diagnostico(tabla, clave_lectura) -> None:
    """Muestra las tablas y reutiliza las figuras de la lectura actual."""
    st.subheader("Diagnóstico exploratorio")
    if st.session_state.get("lectura_eda") != clave_lectura:
        diagnostico = analizar_tabla(tabla)
        destino = Path(__file__).resolve().parent / "reports"
        try:
            figuras = guardar_figuras_eda(tabla, diagnostico, destino)
        except OSError as error:
            st.error(f"No se pudieron guardar los gráficos: {error}")
            return
        st.session_state["resultado_eda"] = diagnostico, figuras
        st.session_state["lectura_eda"] = clave_lectura
    diagnostico, figuras = st.session_state["resultado_eda"]
    st.dataframe(diagnostico.estructura)
    if diagnostico.descriptivos.empty:
        st.caption("La tabla contiene únicamente variables no numéricas.")
        return
    with st.expander("Distribución de las variables numéricas", expanded=True):
        st.dataframe(diagnostico.descriptivos)
        st.caption("Atípicos fuera de Q1 − 1,5 IQR o Q3 + 1,5 IQR. "
                   "Varianza baja entre 0 (excluido) y 10⁻⁶. Asimetría marcada si |g₁| > 1.")
    with st.expander("Covarianza y correlaciones"):
        for titulo, matriz in [("Covarianza", diagnostico.covarianza), ("Pearson", diagnostico.pearson),
                               ("Spearman", diagnostico.spearman)]:
            st.write(titulo)
            st.dataframe(matriz)
        st.write("Pares con correlación positiva alta")
        altas = diagnostico.pares[diagnostico.pares["Pearson"] > UMBRAL_CORRELACION]
        if altas.empty:
            st.caption("Ningún par supera el umbral de Pearson.")
        else:
            st.dataframe(altas, hide_index=True)
    if figuras:
        st.caption("Los mapas muestran hasta ocho variables con correlación definida. "
                   "Las matrices completas están disponibles arriba.")
        for titulo, ruta in figuras.items():
            st.image(str(ruta), caption=titulo)
        st.caption(f"Figuras a 300 DPI en reports/{next(iter(figuras.values())).parent.name}/")


def mostrar_ruta(tabla, clave) -> None:
    """Explica la tarea seleccionada para la tabla."""
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
    else:
        st.text(f"{ruta.motivo} Columna: {ruta.objetivo}")
        tareas = ["Clasificación", "Regresión"]
        tarea = st.selectbox("Tarea supervisada", tareas, index=tareas.index(ruta.tarea),
                             key=f"tarea-{clave}-{ruta.objetivo}-{ruta.tarea}")
        st.info(f"Ruta seleccionada: aprendizaje supervisado · {tarea}.")


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
    except ValueError as error:
        st.error(str(error))
        return

    st.subheader("Vista previa")
    st.caption(f"{len(tabla)} filas · {len(tabla.columns)} columnas · Primeras 10 filas")
    st.dataframe(tabla.head(10), hide_index=True)
    mostrar_ruta(tabla, clave)
    mostrar_diagnostico(tabla, (clave, Path(nombre).suffix.lower(), separador, encabezado))


if __name__ == "__main__":
    mostrar_inicio()
