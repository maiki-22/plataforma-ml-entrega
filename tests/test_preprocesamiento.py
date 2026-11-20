"""Transformaciones con faltantes y categorías exclusivas de prueba."""

import unittest

import numpy as np
import pandas as pd

from core.preprocessor import crear_preprocesador


class PruebasPreprocesamiento(unittest.TestCase):
    def test_estadisticas_solo_del_entrenamiento(self):
        datos = pd.DataFrame({"x": [2, None, 4], "grupo": ["A", None, "B"]})
        preparador = crear_preprocesador(datos, escalar=True).fit(datos)
        rama = preparador.named_transformers_["numericas"]
        prueba = pd.DataFrame({"x": [None, 1000], "grupo": ["C", None]})
        salida = preparador.transform(prueba)
        self.assertEqual(rama["imputar"].statistics_[0], 3)
        self.assertEqual(salida[0, 0], 0)
        self.assertEqual(sum(salida[0, 1:]), 0)
        self.assertEqual(sum(salida[1, 1:]), 1)
        prueba["x"] = [-300, 9000]
        preparador.transform(prueba)
        self.assertEqual(rama["escalar"].mean_[0], 3)

    def test_arbol_conserva_escala_y_columnas_vacias(self):
        tabla = pd.DataFrame({"x": [2, np.inf, 4], "vacia": [np.nan] * 3})
        salida = crear_preprocesador(tabla, escalar=False).fit_transform(tabla)
        np.testing.assert_array_equal(salida, [[2, 0], [3, 0], [4, 0]])
