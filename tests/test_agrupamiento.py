import unittest

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

    def test_rechaza_tabla_constante(self):
        with self.assertRaises(ValueError):
            agrupar(pd.DataFrame({"x": [1, 1, 1]}))
