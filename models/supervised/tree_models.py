"""Árboles CART de clasificación y regresión."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.preprocessor import crear_preprocesador
from models.base import ModeloBase
from models.supervised.candidato import Candidato


class ModeloCART(ModeloBase):
    """Árbol de decisión con preparación de predictores incorporada.

    Parameters
    ----------
    tarea : str
        Clasificación o Regresión.
    profundidad : int
        Profundidad máxima del árbol.
    semilla : int
        Semilla para resolver empates de forma reproducible.
    """

    def __init__(self, tarea, profundidad=3, semilla=42):
        estimadores = {"Clasificación": DecisionTreeClassifier, "Regresión": DecisionTreeRegressor}
        if tarea not in estimadores:
            raise ValueError("CART requiere una tarea de clasificación o regresión.")
        self.arbol = estimadores[tarea](max_depth=profundidad, random_state=semilla)

    def fit(self, predictores, objetivo):
        self.pipeline = Pipeline([
            ("preparar", crear_preprocesador(predictores, escalar=False)),
            ("arbol", self.arbol),
        ])
        self.pipeline.fit(predictores, objetivo)
        return self

    def predict(self, predictores):
        return self.pipeline.predict(predictores)

    def score(self, predictores, objetivo):
        return self.pipeline.score(predictores, objetivo)

    def importancias(self):
        """Suma las importancias de los indicadores de cada variable original.

        Returns
        -------
        pandas.Series
            Importancias sumadas por predictor original y ordenadas de mayor a menor.
        """
        preparador = self.pipeline.named_steps["preparar"]
        nombres = []
        for tipo, transformacion, columnas in preparador.transformers_:
            if tipo == "numericas":
                nombres.extend(columnas)
            elif columnas:
                for columna, categorias in zip(columnas, transformacion["codificar"].categories_, strict=True):
                    nombres.extend([columna] * len(categorias))
        valores = pd.Series(self.arbol.feature_importances_, index=nombres, name="Importancia")
        return valores.groupby(level=0, sort=False).sum().sort_values(ascending=False)


def modelos_de_arbol(tarea, semilla=42):
    """Candidatos de profundidad para árboles y bosques reproducibles.

    Parameters
    ----------
    tarea : str
        Tarea supervisada, «Clasificación» o «Regresión».
    semilla : int
        Semilla de la ejecución para las operaciones aleatorias.

    Returns
    -------
    dict[str, Candidato]
        Candidatos identificados por su nombre visible.
    """
    if tarea == "Clasificación":
        arbol, bosque = DecisionTreeClassifier, RandomForestClassifier
    else:
        arbol, bosque = DecisionTreeRegressor, RandomForestRegressor
    return {
        "CART": Candidato(arbol(random_state=semilla), {"max_depth": [2, 4, None]}, escalar=False),
        "Random Forest": Candidato(bosque(n_estimators=60, random_state=semilla),
                                   {"max_depth": [3, None]}, escalar=False),
    }
