"""Controles de entrenamiento y presentación de resultados por ruta."""

import numpy as np
import streamlit as st

from evaluation.exportacion import exportar_cart, exportar_comparacion, exportar_grupos
from flujo import ejecutar_cart
from interfaz.estado import EstadoVista, Publicacion
from models.supervised.comparacion import catalogo, comparar
from models.unsupervised.kmeans_engine import agrupar


def mostrar_modelo(tabla, seleccion, lectura, destino, estado: EstadoVista):
    """Ejecuta CART bajo demanda y conserva únicamente el resultado vigente.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Tabla de observaciones con nombres de columna únicos.
    seleccion : Ruta
        Columna objetivo y tarea elegidas en el tablero.
    lectura : Lectura
        Huella del archivo y opciones de lectura utilizadas.
    destino : pathlib.Path
        Directorio donde se crean los reportes de esta ejecución.
    estado : EstadoVista
        Parámetros y último resultado de esta vista en la sesión.
    """
    objetivo, tarea = seleccion.objetivo, seleccion.tarea
    st.header(f"Árbol CART de {tarea.lower()}")
    profundidad = st.number_input("Profundidad máxima", 1, 30, estado.parametros or 3, key="profundidad_cart")
    estado.configurar(profundidad)
    if st.button("Ejecutar análisis", type="primary"):
        with st.spinner("Entrenando el árbol y evaluando las filas reservadas…"):
            try:
                resultado = ejecutar_cart(tabla, objetivo, tarea, profundidad)
                figura = exportar_cart(resultado, lectura, destino)
            except (ValueError, OSError) as error:
                st.error(str(error))
                return
            estado.publicacion = Publicacion(resultado, figura)
    if estado.publicacion is None:
        return
    resultado, figura = estado.publicacion
    st.caption(f"Entrenamiento: {resultado.entrenamiento} filas · Prueba: {resultado.prueba} · "
               f"Sin objetivo: {resultado.omitidas} · Semilla: {resultado.semilla}")
    for columna, (nombre, valor) in zip(st.columns(len(resultado.metricas)), resultado.metricas.items()):
        columna.metric(nombre, f"{valor:.3f}")
    evaluacion, importancia = st.columns([1.4, 1])
    evaluacion.image(str(figura), alt="Evaluación del árbol sobre los datos de prueba")
    with evaluacion.expander("Valores de la evaluación"):
        st.dataframe(resultado.detalle)
    importancia.subheader("Importancia de variables")
    importancia.dataframe(resultado.modelo.importancias(), column_config={
        "Importancia": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f")})
    importancia.caption(f"Figura a 300 DPI en reports/{figura.parent.name}/{figura.name}")


def mostrar_comparacion(tabla, seleccion, lectura, destino, estado: EstadoVista):
    """Ajusta los candidatos elegidos y distingue resultados de CV y prueba.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Tabla de observaciones con nombres de columna únicos.
    seleccion : Ruta
        Columna objetivo y tarea elegidas en el tablero.
    lectura : Lectura
        Huella del archivo y opciones de lectura utilizadas.
    destino : pathlib.Path
        Directorio donde se crean los reportes de esta ejecución.
    estado : EstadoVista
        Parámetros y último resultado de esta vista en la sesión.
    """
    objetivo, tarea = seleccion.objetivo, seleccion.tarea
    nombres = list(catalogo(tarea))
    anteriores = nombres if estado.parametros is None else estado.parametros
    candidatos = st.multiselect("Modelos a comparar", nombres, default=anteriores, key=f"candidatos-{tarea}")
    estado.configurar(tuple(candidatos))
    if st.button("Comparar modelos", type="primary", disabled=not candidatos):
        with st.spinner("Validando hiperparámetros sobre entrenamiento…"):
            try:
                resultado = comparar(tabla, objetivo, tarea, candidatos)
                carpeta = exportar_comparacion(resultado, lectura, destino)
            except (ValueError, OSError, np.linalg.LinAlgError) as error:
                st.error(str(error))
                return
            estado.publicacion = Publicacion(resultado, carpeta)
    if estado.publicacion is not None:
        resultado, carpeta = estado.publicacion
        st.caption(f"{resultado.pliegues} pliegues · {resultado.entrenamiento} filas de entrenamiento · "
                   f"{resultado.prueba} filas reservadas para la prueba · Semilla: {resultado.semilla}")
        st.dataframe(resultado.resumen, hide_index=True)
        modelo = st.selectbox("Detalle del modelo", resultado.resumen["Modelo"])
        st.write("Hiperparámetros seleccionados")
        st.json(resultado.parametros[modelo])
        st.write("Matriz normalizada en prueba" if tarea == "Clasificación" else "Predicciones en prueba")
        st.dataframe(resultado.detalles[modelo])
        figuras = sorted(carpeta.glob("*.png"))
        if figuras:
            vista = st.selectbox("Reporte gráfico", figuras, format_func=lambda ruta: ruta.stem)
            st.image(str(vista), alt=vista.stem)
            st.download_button("Descargar reporte", vista.read_bytes, file_name=vista.name,
                               mime="image/png", on_click="ignore")
        st.caption(f"Resultados en reports/{carpeta.name}/")


def mostrar_agrupamiento(tabla, lectura, destino, estado: EstadoVista):
    """Compara grupos y presenta el diagnóstico del K elegido.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Tabla de observaciones con nombres de columna únicos.
    lectura : Lectura
        Huella del archivo y opciones de lectura utilizadas.
    destino : pathlib.Path
        Directorio donde se crean los reportes de esta ejecución.
    estado : EstadoVista
        Parámetros y último resultado de esta vista en la sesión.
    """
    st.header("Agrupamiento con K-Means")
    limite = st.number_input("K máximo a evaluar", 2, 20, estado.parametros or 8, key="limite_grupos")
    estado.configurar(limite)
    if st.button("Evaluar grupos", type="primary"):
        try:
            resultado = agrupar(tabla, limite)
            figuras = exportar_grupos(resultado, lectura, destino)
        except (ValueError, OSError) as error:
            st.error(str(error))
            return
        estado.publicacion = Publicacion(resultado, figuras)
    if estado.publicacion is not None:
        resultado, figuras = estado.publicacion
        st.metric("Grupos seleccionados", resultado.k)
        st.caption(resultado.motivo)
        vista = st.segmented_control("Diagnóstico de grupos", list(figuras), default="Codo", required=True)
        grafico, detalle = st.columns([1.3, 1])
        grafico.image(str(figuras[vista]), alt=vista)
        detalle.dataframe(resultado.metricas, hide_index=True)
        ruta = figuras[vista].parent / "asignaciones.csv"
        detalle.download_button("Descargar grupos por fila", ruta.read_bytes, file_name=ruta.name,
                                mime="text/csv", on_click="ignore")
