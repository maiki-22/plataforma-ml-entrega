"""Verificaciones numéricas del diagnóstico exploratorio."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd
from PIL import Image

from core.eda import analizar_tabla
from evaluation.visualizer import guardar_figuras_eda


class PruebasDiagnostico(unittest.TestCase):
    def test_valores_conocidos(self):
        tabla = pd.DataFrame({"x": [0, 1, 2, 3, 4], "y": [0, 2, 4, 6, 8],
                              "constante": 7, "pequeña": [0, 0, 0, 0, 0.0001],
                              "cola": [1, 1, 1, 1, 30], "grupo": ["a", "a", None, "b", "b"]})
        resultado = analizar_tabla(tabla)
        self.assertEqual(resultado.covarianza.loc["x", "y"], 5)
        self.assertEqual(resultado.pearson.loc["x", "y"], 1)
        self.assertEqual(resultado.spearman.loc["x", "y"], 1)
        self.assertEqual(resultado.estructura.loc["grupo", "Faltantes (%)"], 20)
        self.assertEqual(resultado.estructura.loc["grupo", "Valores distintos (no numéricas)"], 2)
        self.assertTrue(resultado.descriptivos.loc["constante", "Varianza nula"])
        self.assertTrue(resultado.descriptivos.loc["pequeña", "Varianza baja"])
        self.assertTrue(resultado.descriptivos.loc["cola", "Asimetría marcada"])
        self.assertEqual(resultado.descriptivos.loc["cola", "Atípicos (IQR)"], 1)
        self.assertTrue(pd.isna(resultado.pearson.loc["x", "constante"]))
        with tempfile.TemporaryDirectory() as carpeta:
            figuras = guardar_figuras_eda(tabla, resultado, Path(carpeta))
            self.assertEqual(set(figuras), {"Pearson", "Spearman", "Dispersión"})
            for ruta in figuras.values():
                with Image.open(ruta) as imagen:
                    self.assertAlmostEqual(imagen.info["dpi"][0], 300, places=1)

    def test_solo_categorias(self):
        resultado = analizar_tabla(pd.DataFrame({"grupo": ["A", None, "B"]}))
        self.assertTrue(resultado.descriptivos.empty)
        self.assertTrue(resultado.pearson.empty)
        self.assertEqual(resultado.estructura.loc["grupo", "Faltantes"], 1)

    def test_correlacion_por_rangos_y_faltantes(self):
        tabla = pd.DataFrame({"x": [1, 2, 3, 4, None], "y": [1, 4, 9, 16, 100]})
        resultado = analizar_tabla(tabla)
        self.assertEqual(resultado.spearman.loc["x", "y"], 1)
        self.assertLess(resultado.pearson.loc["x", "y"], 1)
