"""Vistas del diagnóstico estructural, distribuciones y relaciones."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from core.eda import UMBRAL_CORRELACION, analizar_tabla
from evaluation.visualizer import guardar_figuras_eda


def obtener_diagnostico(tabla, clave_lectura, destino):
    """Calcula el diagnóstico y las figuras una vez por lectura en la sesión."""
    if st.session_state.get("lectura_eda") != clave_lectura:
        diagnostico = analizar_tabla(tabla)
        try:
            figuras = guardar_figuras_eda(tabla, diagnostico, destino)
        except OSError as error:
            st.error(f"No se pudieron guardar los gráficos: {error}")
            return
        st.session_state["resultado_eda"] = diagnostico, figuras
        st.session_state["lectura_eda"] = clave_lectura
    return st.session_state["resultado_eda"]


def mostrar_variables(diagnostico) -> None:
    """Presenta una tabla de diagnóstico por consulta."""
    st.header("Perfil de variables")
    vistas = ["Estructura y faltantes"]
    if not diagnostico.descriptivos.empty:
        vistas.append("Distribución numérica")
    vista = st.segmented_control("Consulta", vistas, default=vistas[0], required=True)
    if vista == vistas[0]:
        st.dataframe(diagnostico.estructura, column_config={
            "Faltantes (%)": st.column_config.ProgressColumn("Faltantes (%)", min_value=0, max_value=100, format="%.1f%%")
        })
    else:
        st.dataframe(diagnostico.descriptivos)
        st.caption("Atípicos fuera de Q1 − 1,5 IQR o Q3 + 1,5 IQR. "
                   "Varianza baja entre 0 (excluido) y 10⁻⁶. Asimetría marcada si |g₁| > 1.")


def mostrar_relaciones(diagnostico, figuras) -> None:
    """Relaciona cada figura con su matriz y las variables comparadas."""
    st.header("Relaciones entre variables")
    if diagnostico.descriptivos.empty:
        st.write("Se necesitan variables numéricas para calcular relaciones.")
        return
    matrices = {"Pearson": diagnostico.pearson, "Spearman": diagnostico.spearman,
                "Covarianza": diagnostico.covarianza}
    opciones = list(matrices)
    if "Dispersión" in figuras:
        opciones.append("Dispersión")
    metodo = st.segmented_control("Medida o gráfico", opciones, default="Pearson", required=True)
    if metodo in figuras:
        grafico, detalle = st.columns([1.4, 1], gap="large")
        grafico.image(str(figuras[metodo]), width="stretch", alt=f"Gráfico de {metodo} de las variables numéricas")
        detalle.download_button("Descargar figura", figuras[metodo].read_bytes,
                                file_name=figuras[metodo].name, mime="image/png",
                                icon=":material/download:", on_click="ignore")
        if metodo in matrices:
            detalle.caption("El mapa incluye hasta ocho variables con correlación definida.")
            with detalle.expander("Matriz completa", expanded=True):
                st.dataframe(matrices[metodo])
    else:
        st.dataframe(matrices[metodo])
    if metodo == "Pearson":
        with st.expander("Pares con correlación positiva alta"):
            altas = diagnostico.pares[diagnostico.pares["Pearson"] > UMBRAL_CORRELACION]
            if altas.empty:
                st.caption(f"Ningún par supera {UMBRAL_CORRELACION:g}.".replace(".", ",", 1))
            else:
                st.dataframe(altas, hide_index=True)
    if figuras:
        st.caption(f"Figuras a 300 DPI en reports/{next(iter(figuras.values())).parent.name}/")


def mostrar_distribucion(tabla, diagnostico):
    """Resume una variable en intervalos o en las categorías más frecuentes."""
    st.subheader("Distribución")
    nombre = st.selectbox("Variable a explorar", tabla.columns, key="distribucion")
    valores = tabla[nombre].dropna()
    if nombre in diagnostico.descriptivos.index:
        valores = valores[np.isfinite(valores)].astype(float)
        if valores.empty:
            st.caption("La variable carece de observaciones finitas.")
            return
        frecuencias, limites = np.histogram(valores, bins=min(12, valores.nunique()))
        datos = pd.DataFrame({"Desde": limites[:-1], "Hasta": limites[1:], "Filas": frecuencias})
        grafico = alt.Chart(datos).mark_bar().encode(
            x=alt.X("Desde:Q", bin="binned", title=nombre, scale=alt.Scale(zero=False)),
            x2="Hasta:Q", y="Filas:Q", tooltip=["Desde:Q", "Hasta:Q", "Filas:Q"])
    else:
        if valores.empty:
            st.caption("La variable carece de observaciones.")
            return
        datos = valores.astype(str).value_counts().head(10).rename_axis("Categoría").reset_index(name="Filas")
        grafico = alt.Chart(datos).mark_bar().encode(
            y=alt.Y("Categoría:N", sort="-x"), x="Filas:Q", tooltip=["Categoría:N", "Filas:Q"])
        st.caption("Hasta diez categorías por frecuencia.")
    st.altair_chart(grafico.properties(height=190), width="stretch", alt=f"Distribución de {nombre}")


def mostrar_metricas(tabla, diagnostico):
    """Presenta los indicadores del archivo activo."""
    filas, columnas, faltantes, atipicos = st.columns(4)
    filas.metric("Filas", len(tabla))
    columnas.metric("Columnas", len(tabla.columns))
    faltantes.metric("Celdas vacías", int(diagnostico.estructura["Faltantes"].sum()))
    cantidad = int(diagnostico.descriptivos["Atípicos (IQR)"].sum()) if not diagnostico.descriptivos.empty else 0
    atipicos.metric("Valores atípicos", cantidad, help="Suma de observaciones atípicas por variable numérica según IQR.")


def mostrar_resumen(tabla, diagnostico):
    """Muestra las primeras filas junto a la distribución seleccionada."""
    muestra, distribucion = st.columns([1.2, 1], gap="large")
    with muestra:
        st.subheader("Vista previa")
        st.caption(f"Primeras {min(10, len(tabla))} filas de {len(tabla)}")
        st.dataframe(tabla.head(10), height="auto", hide_index=True)
    with distribucion:
        mostrar_distribucion(tabla, diagnostico)
