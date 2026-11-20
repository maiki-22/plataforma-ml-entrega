"""Entrada interactiva de la plataforma de aprendizaje automático."""

from hashlib import sha256
from pathlib import Path

import streamlit as st

from core.eda import UMBRAL_CORRELACION, analizar_tabla
from evaluation.visualizer import guardar_figuras_eda
from core.loader import cargar_tabla, seleccionar_ruta


DATOS = Path(__file__).resolve().parent / "data"
DELIMITADORES = {"Automático": None, "Coma": ",", "Punto y coma": ";", "Tabulación": "\t", "Barra vertical": "|"}


def obtener_diagnostico(tabla, clave_lectura):
    """Muestra las tablas y reutiliza las figuras de la lectura actual."""
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
    return st.session_state["resultado_eda"]


def mostrar_variables(diagnostico) -> None:
    """Presenta una tabla de diagnóstico por consulta."""
    st.header("Perfil de variables")
    vistas = ["Estructura y faltantes"]
    if not diagnostico.descriptivos.empty:
        vistas.append("Distribución numérica")
    vista = st.radio("Consulta", vistas, horizontal=True)
    if vista == vistas[0]:
        st.dataframe(diagnostico.estructura)
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
    metodo = st.radio("Medida o gráfico", opciones, horizontal=True)
    if metodo in figuras:
        st.image(str(figuras[metodo]), width=560, alt=f"Gráfico de {metodo} de las variables numéricas")
        if metodo in matrices:
            st.caption("El mapa incluye hasta ocho variables con correlación definida.")
            with st.expander("Matriz completa"):
                st.dataframe(matrices[metodo])
    else:
        st.dataframe(matrices[metodo])
    if metodo == "Pearson":
        with st.expander("Pares con correlación positiva alta"):
            altas = diagnostico.pares[diagnostico.pares["Pearson"] > UMBRAL_CORRELACION]
            if altas.empty:
                st.caption("Ningún par supera 0,85.")
            else:
                st.dataframe(altas, hide_index=True)
    if figuras:
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
    st.set_page_config(page_title="Cuaderno de datos", page_icon=":material/table_chart:")
    st.title("Cuaderno de datos")
    datos, objetivo, variables, relaciones = st.tabs(["Datos", "Objetivo", "Variables", "Relaciones"])
    with datos:
        st.header("Conjunto de datos")
        fuente, seleccion = st.columns([1, 1.4], vertical_alignment="top")
        origen = fuente.radio("Origen de los datos", ["Ejemplo local", "Subir archivo"], horizontal=True)
        if origen == "Subir archivo":
            archivo = seleccion.file_uploader("Selecciona un archivo", type=["csv", "tsv", "parquet"])
            if archivo is None:
                for panel in (objetivo, variables, relaciones):
                    panel.write("Selecciona un archivo en Datos para continuar.")
                return
            nombre, contenido = archivo.name, archivo.getvalue()
        else:
            nombre = seleccion.selectbox("Archivo de ejemplo", ["flores.csv", "mediciones.tsv"])
            contenido = (DATOS / nombre).read_bytes()

    clave = sha256(contenido).hexdigest()
    separador, encabezado = None, True
    if Path(nombre).suffix.lower() != ".parquet":
        with datos.expander("Opciones de lectura"):
            opcion = st.selectbox("Delimitador", list(DELIMITADORES), key=f"sep-{clave}")
            separador = DELIMITADORES[opcion]
            encabezado = st.checkbox("La primera fila contiene los nombres", True, key=f"cab-{clave}")

    try:
        tabla = cargar_tabla(nombre, contenido, separador, encabezado)
    except ValueError as error:
        datos.error(str(error))
        for panel in (objetivo, variables, relaciones):
            panel.write("Revisa el archivo y sus opciones de lectura en Datos.")
        return

    contexto = f"{nombre} · {len(tabla)} filas · {len(tabla.columns)} columnas"
    with datos:
        st.subheader("Vista previa")
        st.caption(f"{contexto} · Primeras 10 filas")
        st.dataframe(tabla.head(10), height=265, hide_index=True)
    with objetivo:
        st.header("Objetivo de aprendizaje")
        st.caption(contexto)
        mostrar_ruta(tabla, clave)
    resultado = obtener_diagnostico(tabla, (clave, Path(nombre).suffix.lower(), separador, encabezado))
    if resultado is not None:
        diagnostico, figuras = resultado
        with variables:
            st.caption(contexto)
            mostrar_variables(diagnostico)
        with relaciones:
            st.caption(contexto)
            mostrar_relaciones(diagnostico, figuras)


if __name__ == "__main__":
    mostrar_inicio()
