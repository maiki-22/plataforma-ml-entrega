"""Entrada interactiva de la plataforma de aprendizaje automático."""

import streamlit as st


def mostrar_inicio() -> None:
    """Presenta el selector de archivos tabulares."""
    st.set_page_config(page_title="Análisis de datos tabulares", page_icon="📊")
    st.title("Análisis de datos tabulares")
    archivo = st.file_uploader("Selecciona un archivo", type=["csv", "tsv", "parquet"])
    if archivo is not None:
        st.text(f"Archivo seleccionado: {archivo.name}")


if __name__ == "__main__":
    mostrar_inicio()
