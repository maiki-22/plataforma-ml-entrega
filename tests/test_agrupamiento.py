import unittest
from unittest.mock import patch

import pandas as pd

from models.unsupervised.kmeans_engine import agrupar


class PruebasGrupos(unittest.TestCase):
    def test_separa_nubes_y_compara_inicios(self):
        tabla = pd.DataFrame({"x": [0, 0.1, 0.2, 10, 10.1, 10.2, 20, 20.1, 20.2]})
        resultado = agrupar(tabla, 8)
        self.assertEqual(resultado.k, 3)
        self.assertEqual(resultado.filas["Grupo"].nunique(), 3)
        self.assertEqual(resultado.centroides.shape, (3, 2))
        self.assertEqual(len(resultado.metricas.dropna()), 14)

    def test_calculo_por_bloques_conserva_el_resultado(self):
        tabla = pd.DataFrame({"x": [0, 1, 2, 10, 11, 12], "y": [3, 4, 1, 8, 5, 9]})
        esperado = agrupar(tabla, 4)
        with patch("models.unsupervised.kmeans_engine.MAX_BYTES_DISTANCIAS", 0):
            por_bloques = agrupar(tabla, 4)
        self.assertEqual(esperado.k, por_bloques.k)
        pd.testing.assert_frame_equal(esperado.metricas, por_bloques.metricas)
        pd.testing.assert_frame_equal(esperado.filas, por_bloques.filas)

    def test_rechaza_tabla_constante(self):
        with self.assertRaises(ValueError):
            agrupar(pd.DataFrame({"x": [1, 1, 1]}))
