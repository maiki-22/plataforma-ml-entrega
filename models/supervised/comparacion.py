"""Selección de hiperparámetros dentro de entrenamiento y evaluación reservada."""

from dataclasses import dataclass

import pandas as pd
from scipy.sparse import issparse
from sklearn.model_selection import GridSearchCV, KFold, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from threadpoolctl import threadpool_limits

from core.particion import separar_datos
from core.preprocessor import crear_preprocesador
from evaluation.metrics import evaluar_supervisado
from models.base import ModeloBase
from models.supervised.linear_models import modelos_lineales
from models.supervised.prototype_models import modelos_vecinos
from models.supervised.spline_models import modelos_spline
from models.supervised.tree_models import modelos_de_arbol


def catalogo(tarea, max_vecinos=5):
    """Devuelve los estimadores aplicables a la tarea y sus rejillas."""
    modelos = modelos_lineales(tarea) | modelos_de_arbol(tarea) | modelos_vecinos(tarea, max_vecinos)
    if tarea == "Regresión":
        modelos.update(modelos_spline())
    return modelos


def _densa(datos):
    return datos.toarray() if issparse(datos) else datos


class ModeloValidado(ModeloBase):
    """Pipeline con preprocesamiento reajustado dentro de cada pliegue."""

    def __init__(self, nombre, estimador, parametros, pliegues, criterio):
        self.nombre = nombre
        self.estimador = estimador
        self.parametros = parametros
        self.pliegues = pliegues
        self.criterio = criterio

    def fit(self, predictores, objetivo):
        escalar = self.nombre not in {"CART", "Random Forest", "MARS", "GAM"}
        pasos = [("preparar", crear_preprocesador(predictores, escalar=escalar))]
        if self.nombre in {"LVQ1", "DANN", "MARS", "GAM"}:
            pasos.append(("densa", FunctionTransformer(_densa)))
        pasos.append(("modelo", self.estimador))
        rejilla = {f"modelo__{k}": v for k, v in self.parametros.items()}
        self.busqueda = GridSearchCV(Pipeline(pasos), rejilla, cv=self.pliegues,
                                     scoring=self.criterio, error_score="raise")
        self.busqueda.fit(predictores, objetivo)
        return self

    def predict(self, predictores):
        return self.busqueda.predict(predictores)

    def score(self, predictores, objetivo):
        return self.busqueda.best_estimator_.score(predictores, objetivo)


@dataclass
class Comparacion:
    resumen: pd.DataFrame
    detalles: dict
    parametros: dict
    pliegues: int
    entrenamiento: int
    prueba: int
    ajustes: dict


def comparar(tabla, objetivo, tarea, nombres):
    """Compara candidatos en particiones comunes con exactitud o MAE como criterio."""
    x, prueba, y, reales, _ = separar_datos(tabla, objetivo, tarea)
    clasificacion = tarea == "Clasificación"
    cantidad = min(5, int(y.value_counts().min()) if clasificacion else len(y))
    if cantidad < 2:
        raise ValueError("Se necesitan al menos dos observaciones de entrenamiento por clase para validación cruzada.")
    divisor = StratifiedKFold if clasificacion else KFold
    pliegues = list(divisor(cantidad, shuffle=True, random_state=42).split(x, y))
    candidatos = catalogo(tarea, min(len(indices) for indices, _ in pliegues))
    criterio = "accuracy" if clasificacion else "neg_mean_absolute_error"
    filas, detalles, parametros, ajustes = [], {}, {}, {}
    with threadpool_limits(limits=1):
        for nombre in nombres:
            estimador, rejilla = candidatos[nombre]
            modelo = ModeloValidado(nombre, estimador, rejilla, pliegues, criterio).fit(x, y)
            ajustes[nombre] = modelo.busqueda
            metricas, detalles[nombre] = evaluar_supervisado(reales, modelo.predict(prueba), tarea)
            cv = modelo.busqueda.best_score_
            columna = "Exactitud CV" if clasificacion else "MAE CV"
            filas.append({"Modelo": nombre, columna: cv if clasificacion else -cv, **metricas})
            parametros[nombre] = {k.removeprefix("modelo__"): v for k, v in modelo.busqueda.best_params_.items()}
    return Comparacion(pd.DataFrame(filas), detalles, parametros, cantidad, len(x), len(prueba), ajustes)
