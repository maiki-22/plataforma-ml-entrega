"""Árboles CART de clasificación y regresión."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from core.preprocessor import crear_preprocesador
from models.base import ModeloBase


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
        """Ajusta el preprocesador y el árbol con las mismas filas."""
        self.pipeline = Pipeline([
            ("preparar", crear_preprocesador(predictores, escalar=False)),
            ("arbol", self.arbol),
        ])
        self.pipeline.fit(predictores, objetivo)
        return self

    def predict(self, predictores):
        """Aplica las transformaciones aprendidas y devuelve predicciones."""
        return self.pipeline.predict(predictores)

    def score(self, predictores, objetivo):
        """Devuelve exactitud en clasificación o R² en regresión."""
        return self.pipeline.score(predictores, objetivo)

    def importancias(self):
        """Suma las importancias de los indicadores de cada variable original."""
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


def modelos_de_arbol(tarea):
    """Candidatos de profundidad para árboles y bosques reproducibles."""
    if tarea == "Clasificación":
        arbol, bosque = DecisionTreeClassifier, RandomForestClassifier
    else:
        arbol, bosque = DecisionTreeRegressor, RandomForestRegressor
    return {"CART": (arbol(random_state=42), {"max_depth": [2, 4, None]}),
            "Random Forest": (bosque(n_estimators=60, random_state=42), {"max_depth": [3, None]})}
