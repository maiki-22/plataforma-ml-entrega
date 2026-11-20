"""Un fallo de escritura no debe publicar un reporte incompleto."""

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from evaluation.exportacion import exportar_cart
from flujo import ejecutar_cart


class PruebasExportacion(unittest.TestCase):
    def test_error_de_escritura_permite_reintentar_sin_publicar_parciales(self):
        tabla = pd.DataFrame({"x": range(10), "objetivo": [0, 1] * 5})
        tarea = "Clasificación"
        resultado = ejecutar_cart(tabla, "objetivo", tarea)
        with TemporaryDirectory() as temporal:
            destino = Path(temporal)
            argumentos = (resultado, "objetivo", tarea, 3, ("abc", ".csv", None, True), destino)
            with patch("evaluation.exportacion.guardar_configuracion", side_effect=OSError("Error de escritura")):
                with self.assertRaises(OSError):
                    exportar_cart(*argumentos)
            self.assertEqual(list(destino.iterdir()), [])
            figura = exportar_cart(*argumentos)
            self.assertEqual(set(p.name for p in figura.parent.iterdir()),
                             {"evaluacion.png", "configuracion.json", "metricas.csv"})
