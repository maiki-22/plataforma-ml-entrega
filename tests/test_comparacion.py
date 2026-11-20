"""Comprobación conjunta de los modelos sobre particiones comunes."""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from models.supervised.comparacion import catalogo, comparar
from models.supervised.prototype_models import LVQ1
from streamlit.testing.v1 import AppTest


RAIZ = Path(__file__).resolve().parents[1]


class PruebasComparacion(unittest.TestCase):
    def test_lvq_conserva_puntajes_y_desempata_por_orden_de_clase(self):
        modelo = LVQ1(epocas=0).fit(np.array([[0.], [4.]]), np.array(["A", "B"]))
        consultas = np.array([[2.], [1.], [3.]])
        distancia_relativa = np.exp(-8)
        probabilidades = np.array([[0.5, 0.5], [1, distancia_relativa], [distancia_relativa, 1]])
        probabilidades /= probabilidades.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(modelo.predict_proba(consultas), probabilidades)
        np.testing.assert_array_equal(modelo.predict(consultas), ["A", "A", "B"])

    def test_todos_los_modelos_sobre_ejemplos(self):
        for tarea, archivo, objetivo, sep in [
            ("Clasificación", "flores.csv", "clase", ","),
            ("Regresión", "mediciones.tsv", "temperatura_c", "\t"),
        ]:
            tabla = pd.read_csv(RAIZ / "data" / archivo, sep=sep)
            resultado = comparar(tabla, objetivo, tarea, list(catalogo(tarea)))
            self.assertTrue(np.isfinite(resultado.resumen.select_dtypes("number")).all().all())
            self.assertEqual(resultado.pliegues, 5)
            self.assertEqual((resultado.entrenamiento, resultado.prueba), (28, 8))
            self.assertEqual(set(resultado.detalles), set(catalogo(tarea)))

    def test_recorrido_comparacion_y_grupos(self):
        app = AppTest.from_file(str(RAIZ / "main.py"), default_timeout=30).run()
        next(c for c in app.segmented_control if c.label == "Análisis supervisado").set_value("Comparación").run()
        next(c for c in app.button if c.label == "Comparar modelos").click().run()
        self.assertFalse(app.exception)
        resultado, carpeta = app.session_state["comparacion"]
        self.assertEqual(len(resultado.resumen), 6)
        self.assertTrue((carpeta / "metricas.csv").is_file())
        areas = pd.read_csv(carpeta / "auc.csv")
        self.assertTrue(areas["AUC ROC"].between(0, 1).all())
        self.assertEqual(areas["Clase"].nunique(), 4)
        for nombre in ("Superficie de decisión.png", "Curvas de aprendizaje.png", "configuracion.json"):
            self.assertTrue((carpeta / nombre).is_file())
        app.selectbox(key="ejemplo").set_value("mediciones.tsv").run()
        next(c for c in app.button if c.label == "Evaluar grupos").click().run()
        self.assertFalse(app.exception)
        resultado, figuras = app.session_state["grupos"]
        self.assertEqual(len(resultado.filas), 36)
        self.assertTrue(all(ruta.is_file() for ruta in figuras.values()))
