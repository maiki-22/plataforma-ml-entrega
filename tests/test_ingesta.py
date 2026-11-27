"""Pruebas de carga de archivos y selección de la tarea de aprendizaje."""

import unittest
from pathlib import Path
from unittest.mock import patch

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

    def test_detecta_la_etiqueta_por_nombre_o_por_categorias_finales(self):
        tabla = pd.DataFrame({"medida": range(9), "grupo": ["A", "B", "C"] * 3})
        ruta = seleccionar_ruta(tabla)
        self.assertEqual((ruta.objetivo, ruta.tarea), ("grupo", "Clasificación"))
        self.assertIn("última columna", ruta.motivo)
        tabla = tabla.rename(columns={"grupo": "Especie"})[["Especie", "medida"]]
        self.assertEqual(seleccionar_ruta(tabla).objetivo, "Especie")
        tabla["identificador"] = [f"muestra-{i}" for i in range(9)]
        self.assertIsNone(seleccionar_ruta(tabla.drop(columns="Especie")).objetivo)
        self.assertIsNone(seleccionar_ruta(tabla.drop(columns=["Especie", "identificador"])).objetivo)

    def test_distingue_codigos_de_clase_y_objetivos_continuos(self):
        tabla = pd.DataFrame({"medida": range(9), "target": [0, 1, 2] * 3})
        ruta = seleccionar_ruta(tabla)
        self.assertEqual(ruta.tarea, "Clasificación")
        self.assertIn("códigos de clase", ruta.motivo)
        tabla = tabla.rename(columns={"target": "precio_objetivo"})
        tabla["precio_objetivo"] = [2.5, 3.125, 4.625] * 3
        self.assertEqual(seleccionar_ruta(tabla).tarea, "Regresión")

    def test_objetivo_invalido_en_modo_automatico_o_manual(self):
        for valor in (None, float("inf")):
            with self.subTest(valor=valor):
                tabla = pd.DataFrame({"medida": range(6), "target": [valor] * 6})
                ruta = seleccionar_ruta(tabla)
                self.assertIsNone(ruta.objetivo)
                self.assertIn("K-Means", ruta.tarea)
                self.assertIn("target", ruta.motivo)
                with self.assertRaisesRegex(ValueError, "Elige otra"):
                    seleccionar_ruta(tabla, "target")

    def test_streamlit_advierte_y_permite_agrupar_con_un_objetivo_vacio(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run()
        app.radio(key="origen").set_value("Subir archivo").run()
        app.file_uploader[0].upload("datos.csv", b"medida,target\n1,\n2,\n3,\n7,\n8,\n9,\n").run()
        self.assertIn("está vacía", app.warning[0].value)
        self.assertIn("K-Means", app.warning[0].value)
        next(control for control in app.button if control.label == "Evaluar grupos").click().run()
        resultado, _ = app.session_state["estado_analisis"].grupos.publicacion
        self.assertEqual(len(resultado.filas), 6)
        self.assertFalse(app.error)
        self.assertFalse(app.exception)

    def test_reutiliza_tabla_solo_si_coinciden_contenido_y_lectura(self):
        with patch("interfaz.entrada.cargar_tabla", wraps=cargar_tabla) as cargar:
            app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run()
            app.selectbox(key="distribucion").set_value("clase").run()
            self.assertEqual(cargar.call_count, 1)
            app.radio(key="origen").set_value("Subir archivo").run()
            app.file_uploader[0].upload("tabla.csv", b"a,b\n1,2\n").run()
            app.file_uploader[0].upload("tabla.csv", b"a,b\n3,4\n").run()
            self.assertEqual(cargar.call_count, 3)
            self.assertEqual(app.session_state["tabla_activa"].iloc[0, 0], 3)
            app.checkbox[0].uncheck().run()
            self.assertEqual(cargar.call_count, 4)
            self.assertEqual(app.session_state["tabla_activa"].shape, (2, 2))

    def test_recorrido_streamlit(self):
        app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "main.py")).run()
        self.assertEqual([pestana.label for pestana in app.tabs], ["Resumen", "Variables", "Relaciones", "Modelo"])
        self.assertEqual([metrica.value for metrica in app.metric[:2]], ["36", "3"])
        app.selectbox(key="distribucion").set_value("clase").run()
        self.assertTrue(any("categorías por frecuencia" in texto.value for texto in app.caption))
        self.assertIn("Clasificación", app.info[0].value)
        app.selectbox(key="ejemplo").set_value("mediciones.tsv").run()
        self.assertIn("K-Means", app.warning[0].value)
        next(control for control in app.radio if control.label == "Variable objetivo").set_value("Elegir columna").run()
        self.assertIn("Regresión", app.info[0].value)
        next(control for control in app.selectbox if control.label == "Tarea supervisada").set_value("Clasificación").run()
        self.assertIn("Clasificación", app.info[0].value)
        app.radio(key="origen").set_value("Subir archivo").run()
        app.file_uploader[0].upload("error.parquet", b"mal").run()
        self.assertTrue(app.error)
        self.assertFalse(app.dataframe)
        app.file_uploader[0].upload("valido.csv", b"medida,clase\n2,A\n").run()
        self.assertIn("Clasificación", app.info[0].value)
        self.assertFalse(app.exception)
