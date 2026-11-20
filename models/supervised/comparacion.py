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
from evaluation.resultados import ResultadoSupervisado
from models.base import ModeloBase
from models.supervised.candidato import Candidato
from models.supervised.linear_models import modelos_lineales
from models.supervised.prototype_models import modelos_vecinos
from models.supervised.spline_models import modelos_spline
from models.supervised.tree_models import modelos_de_arbol


def catalogo(tarea, max_vecinos=5, semilla=42) -> dict[str, Candidato]:
    """Devuelve los estimadores aplicables a la tarea y sus rejillas."""
    modelos = modelos_lineales(tarea) | modelos_de_arbol(tarea, semilla) | modelos_vecinos(tarea, max_vecinos, semilla)
    if tarea == "Regresión":
        modelos.update(modelos_spline())
    return modelos


def _densa(datos):
    return datos.toarray() if issparse(datos) else datos


class ModeloValidado(ModeloBase):
    """Pipeline con preprocesamiento reajustado dentro de cada pliegue."""

    def __init__(self, candidato: Candidato, pliegues, criterio):
        self.candidato = candidato
        self.pliegues = pliegues
        self.criterio = criterio

    def fit(self, predictores, objetivo):
        pasos = [("preparar", crear_preprocesador(predictores, escalar=self.candidato.escalar))]
        if self.candidato.requiere_densa:
            pasos.append(("densa", FunctionTransformer(_densa)))
        pasos.append(("modelo", self.candidato.estimador))
        rejilla = {f"modelo__{k}": v for k, v in self.candidato.parametros.items()}
        self.busqueda = GridSearchCV(Pipeline(pasos), rejilla, cv=self.pliegues,
                                     scoring=self.criterio, error_score="raise")
        self.busqueda.fit(predictores, objetivo)
        return self

    def predict(self, predictores):
        return self.busqueda.predict(predictores)

    def score(self, predictores, objetivo):
        return self.busqueda.best_estimator_.score(predictores, objetivo)


@dataclass
class Comparacion(ResultadoSupervisado):
    """Ajustes, métricas y parámetros de los candidatos sobre las mismas filas."""

    resumen: pd.DataFrame
    detalles: dict
    parametros: dict
    pliegues: int
    ajustes: dict


def comparar(tabla, objetivo, tarea, nombres, semilla=42):
    """Compara candidatos en particiones comunes con exactitud o MAE como criterio."""
    particion = separar_datos(tabla, objetivo, tarea, semilla)
    predictores = particion.predictores_entrenamiento
    etiquetas = particion.objetivo_entrenamiento
    clasificacion = tarea == "Clasificación"
    cantidad = min(5, int(etiquetas.value_counts().min()) if clasificacion else len(etiquetas))
    if cantidad < 2:
        raise ValueError("Se necesitan al menos dos observaciones de entrenamiento por clase para validación cruzada.")
    divisor = StratifiedKFold if clasificacion else KFold
    pliegues = list(divisor(cantidad, shuffle=True, random_state=semilla).split(predictores, etiquetas))
    candidatos = catalogo(tarea, min(len(indices) for indices, _ in pliegues), semilla=semilla)
    criterio = "accuracy" if clasificacion else "neg_mean_absolute_error"
    filas, detalles, parametros, ajustes = [], {}, {}, {}
    with threadpool_limits(limits=1):
        for nombre in nombres:
            modelo = ModeloValidado(candidatos[nombre], pliegues, criterio)
            modelo.fit(predictores, etiquetas)
            ajustes[nombre] = modelo.busqueda
            predicciones = modelo.predict(particion.predictores_prueba)
            metricas, detalles[nombre] = evaluar_supervisado(particion.objetivo_prueba, predicciones, tarea)
            cv = modelo.busqueda.best_score_
            columna = "Exactitud CV" if clasificacion else "MAE CV"
            filas.append({"Modelo": nombre, columna: cv if clasificacion else -cv, **metricas})
            parametros[nombre] = {k.removeprefix("modelo__"): v for k, v in modelo.busqueda.best_params_.items()}
    return Comparacion(objetivo=objetivo, tarea=tarea, semilla=semilla, particion=particion,
                       resumen=pd.DataFrame(filas), detalles=detalles, parametros=parametros,
                       pliegues=cantidad, ajustes=ajustes)
