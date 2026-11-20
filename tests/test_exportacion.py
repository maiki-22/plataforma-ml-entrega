"""Un fallo de escritura no debe publicar un reporte incompleto."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from sklearn.metrics import accuracy_score, roc_auc_score

from evaluation.exportacion import exportar_cart, exportar_comparacion
from flujo import ejecutar_cart
from models.supervised.comparacion import comparar


class PruebasExportacion(unittest.TestCase):
    def test_cart_exporta_la_configuracion_real(self):
        tabla = pd.DataFrame({"x": range(20), "y": range(20)}, index=range(30, 50))
        resultado = ejecutar_cart(tabla, "y", "Regresión", profundidad=2, semilla=7)
        with TemporaryDirectory() as temporal:
            figura = exportar_cart(resultado, ("abc", ".csv", None, True), Path(temporal))
            datos = json.loads(figura.with_name("configuracion.json").read_text())
        self.assertEqual((datos["semilla"], datos["profundidad"]), (7, 2))
        pd.testing.assert_index_equal(resultado.detalle.index, resultado.particion.objetivo_prueba.index)
        pd.testing.assert_index_equal(resultado.detalle.index, resultado.particion.predictores_prueba.index)

    def test_comparacion_exporta_curvas_de_las_filas_evaluadas(self):
        tabla = pd.DataFrame({"x": range(40), "clase": ["A", "B"] * 20}, index=[i // 2 for i in range(40)])
        tabla.iloc[0, 1] = None
        resultado = comparar(tabla, "clase", "Clasificación", ["CART"], semilla=7)
        prueba = resultado.particion.predictores_prueba
        reales = resultado.particion.objetivo_prueba
        ajuste = resultado.ajustes["CART"]
        self.assertEqual(ajuste.best_estimator_["modelo"].random_state, 7)
        self.assertEqual(resultado.omitidas, 1)
        self.assertEqual(resultado.resumen.at[0, "Exactitud"], accuracy_score(reales, ajuste.predict(prueba)))
        auc_esperada = roc_auc_score(reales == "B", ajuste.predict_proba(prueba)[:, 1])
        tabla["clase"] = "otra"
        with TemporaryDirectory() as temporal:
            carpeta = exportar_comparacion(resultado, ("abc", ".csv", None, True), Path(temporal))
            datos = json.loads((carpeta / "configuracion.json").read_text())
            areas = pd.read_csv(carpeta / "auc.csv").set_index("Clase")
        self.assertEqual(datos["semilla"], 7)
        self.assertAlmostEqual(areas.at["B", "AUC ROC"], auc_esperada)

    def test_error_de_escritura_permite_reintentar_sin_publicar_parciales(self):
        tabla = pd.DataFrame({"x": range(10), "objetivo": [0, 1] * 5})
        tarea = "Clasificación"
        resultado = ejecutar_cart(tabla, "objetivo", tarea)
        with TemporaryDirectory() as temporal:
            destino = Path(temporal)
            argumentos = (resultado, ("abc", ".csv", None, True), destino)
            with patch("evaluation.exportacion.guardar_configuracion", side_effect=OSError("Error de escritura")):
                with self.assertRaises(OSError):
                    exportar_cart(*argumentos)
            self.assertEqual(list(destino.iterdir()), [])
            figura = exportar_cart(*argumentos)
            self.assertEqual(set(p.name for p in figura.parent.iterdir()),
                             {"evaluacion.png", "configuracion.json", "metricas.csv"})
