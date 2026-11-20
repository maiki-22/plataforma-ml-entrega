"""Pruebas del primer modelo supervisado y su integración con Streamlit."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import joblib
import numpy as np
import pandas as pd
from streamlit.testing.v1 import AppTest

from flujo import ejecutar_cart


RAIZ = Path(__file__).resolve().parents[1]


class PruebasCART(unittest.TestCase):
    def test_modelo_persistido_y_objetivo_excluido(self):
        tabla = pd.DataFrame({"color": ["azul", "rojo"] * 20, "objetivo": ["A", "B"] * 20})
        resultado = ejecutar_cart(tabla, "objetivo", "Clasificación")
        self.assertEqual(resultado.metricas["Exactitud"], 1)
        self.assertEqual(resultado.modelo.importancias().to_dict(), {"color": 1.0})
        np.testing.assert_array_equal(resultado.detalle.values, np.eye(2))
        with TemporaryDirectory() as carpeta:
            ruta = Path(carpeta) / "arbol.joblib"
            resultado.modelo.save(ruta)
            modelo = joblib.load(ruta)
            self.assertEqual(modelo.score(tabla[["color"]], tabla.objetivo), 1)

    def test_regresion_omite_solo_filas_sin_objetivo(self):
        tabla = pd.DataFrame({"x": range(11), "y": [2] * 10 + [None]})
        resultado = ejecutar_cart(tabla, "y", "Regresión")
        self.assertEqual((resultado.entrenamiento, resultado.prueba, resultado.omitidas), (8, 2, 1))
        self.assertEqual(resultado.metricas["MAE"], 0)

    def test_recorridos_supervisados(self):
        app = AppTest.from_file(str(RAIZ / "main.py"), default_timeout=15).run()
        app.number_input(key="profundidad_cart").set_value(4).run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        primero, figura = app.session_state["estado_analisis"].cart.publicacion
        self.assertIn("Exactitud", primero.metricas)
        app.selectbox(key="distribucion").set_value("clase").run()
        self.assertEqual(app.session_state["estado_analisis"].cart.publicacion.archivos, figura)
        modo = next(c for c in app.segmented_control if c.label == "Análisis supervisado")
        modo.set_value("Comparación").run()
        next(c for c in app.segmented_control if c.label == "Análisis supervisado").set_value("Árbol CART").run()
        self.assertEqual(app.number_input(key="profundidad_cart").value, 4)
        self.assertEqual(app.session_state["estado_analisis"].cart.publicacion.archivos, figura)
        app.selectbox(key="ejemplo").set_value("mediciones.tsv").run()
        self.assertNotIn("Modelo", [p.label for p in app.tabs])
        self.assertIsNone(app.session_state["estado_analisis"].cart.publicacion)
        next(c for c in app.radio if c.label == "Variable objetivo").set_value("Elegir columna").run()
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("RMSE", app.session_state["estado_analisis"].cart.publicacion.resultado.metricas)
        app.number_input[0].set_value(2).run()
        self.assertIsNone(app.session_state["estado_analisis"].cart.publicacion)
