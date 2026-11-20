"""Pruebas de carga de archivos y selección de la tarea de aprendizaje."""

import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from core.loader import cargar_tabla, seleccionar_ruta


class PruebasIngesta(unittest.TestCase):
    def test_csv_latin1_y_comillas(self):
        contenido = 'año;clase\n2020;"uno;dos"\n'.encode("latin-1")
        tabla = cargar_tabla("datos.CSV", contenido)
        self.assertEqual(tabla.loc[0, "clase"], "uno;dos")
        self.assertEqual(seleccionar_ruta(tabla).tarea, "Clasificación")

    def test_tsv_sin_encabezado(self):
        tabla = cargar_tabla("datos.tsv", b"1\t2\n3\t4", encabezado=False)
        self.assertEqual(tabla["columna_1"].tolist(), [1, 3])

    def test_parquet_y_objetivo_manual(self):
        original = pd.DataFrame({"precio": [2.3, 4.6]})
        tabla = cargar_tabla("datos.parquet", original.to_parquet())
        pd.testing.assert_frame_equal(tabla, original)
        self.assertEqual(seleccionar_ruta(tabla, "precio").tarea, "Regresión")

    def test_archivos_invalidos(self):
        for nombre, contenido in [("a.csv", b""), ("a.csv", b"a,a\n1,2"),
                                  ("a.tsv", b"a\tb\n1"), ("a.parquet", b"incorrecto"),
                                  ("a.csv", b"a,b\n"), ("a.xls", b"x")]:
            with self.subTest(nombre=nombre, contenido=contenido), self.assertRaises(ValueError):
                cargar_tabla(nombre, contenido)

    def test_ambiguedad_y_eleccion_manual(self):
        tabla = cargar_tabla("a.csv", b"target,clase\n1,a\n")
        self.assertIsNone(seleccionar_ruta(tabla).objetivo)
        self.assertEqual(seleccionar_ruta(tabla, "clase").objetivo, "clase")
        self.assertIsNone(seleccionar_ruta(tabla, sin_objetivo=True).objetivo)
        with self.assertRaises(ValueError):
            seleccionar_ruta(tabla, "ausente")

    def test_recorrido_streamlit(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run()
        app.radio[0].set_value("Ejemplo local").run()
        self.assertIn("Clasificación", app.info[0].value)
        app.selectbox[0].set_value("mediciones.tsv").run()
        self.assertIn("K-Means", app.warning[0].value)
        app.radio[1].set_value("Elegir columna").run()
        self.assertIn("Regresión", app.info[0].value)
        app.selectbox[-1].set_value("Clasificación").run()
        self.assertIn("Clasificación", app.info[0].value)
        app.radio[0].set_value("Subir archivo").run()
        app.file_uploader[0].upload("error.parquet", b"mal").run()
        self.assertTrue(app.error)
        self.assertFalse(app.dataframe)
        app.file_uploader[0].upload("valido.csv", b"medida,clase\n2,A\n").run()
        self.assertIn("Clasificación", app.info[0].value)
        self.assertFalse(app.exception)
