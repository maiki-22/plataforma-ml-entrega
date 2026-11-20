"""Entrada del tablero de análisis de datos."""

from pathlib import Path

import streamlit as st

from interfaz.entrada import leer_archivo, mostrar_ruta, seleccionar_archivo
from interfaz.exploracion import (
    mostrar_metricas, mostrar_relaciones, mostrar_resumen, mostrar_variables, obtener_diagnostico,
)
from interfaz.modelos import mostrar_agrupamiento, mostrar_comparacion, mostrar_modelo


RAIZ = Path(__file__).resolve().parent
REPORTES = RAIZ / "reports"


def mostrar_inicio() -> None:
    """Organiza los controles de sesión y el tablero del archivo activo."""
    st.set_page_config(page_title="Cuaderno de datos", page_icon=":material/table_chart:", layout="wide")
    principal = st.container()
    with st.sidebar:
        st.header("Cuaderno de datos")
        st.subheader("Fuente de datos")
        archivo = seleccionar_archivo(RAIZ / "data")
        if archivo is None:
            principal.title("Abrir un conjunto de datos")
            principal.write("Selecciona un archivo CSV, TSV o Parquet en el panel lateral.")
            return
        nombre, contenido = archivo
        try:
            tabla, lectura = leer_archivo(nombre, contenido)
        except ValueError as error:
            st.error(str(error))
            return
        st.divider()
        st.subheader("Objetivo de aprendizaje")
        seleccion = mostrar_ruta(tabla, lectura.huella)

    st.title(nombre, anchor=False)
    resultado = obtener_diagnostico(tabla, lectura, REPORTES)
    if resultado is None:
        return
    diagnostico, figuras = resultado
    mostrar_metricas(tabla, diagnostico)

    titulos = ["Resumen", "Variables", "Relaciones"]
    if seleccion and seleccion.objetivo is not None:
        titulos.append("Modelo")
        st.session_state.pop("grupos", None)
    else:
        st.session_state.pop("resultado_cart", None)
        if seleccion:
            titulos.append("Agrupamiento")
    resumen, variables, relaciones, *modelo = st.tabs(titulos)
    with resumen:
        mostrar_resumen(tabla, diagnostico)
    with variables:
        mostrar_variables(diagnostico)
    with relaciones:
        mostrar_relaciones(diagnostico, figuras)
    if modelo:
        with modelo[0]:
            if seleccion.objetivo is None:
                mostrar_agrupamiento(tabla, lectura, REPORTES)
            else:
                modo = st.segmented_control("Análisis supervisado", ["Árbol CART", "Comparación"],
                                            default="Árbol CART", required=True)
                if modo == "Árbol CART":
                    mostrar_modelo(tabla, seleccion, lectura, REPORTES)
                else:
                    mostrar_comparacion(tabla, seleccion, lectura, REPORTES)


if __name__ == "__main__":
    mostrar_inicio()
