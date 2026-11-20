"""Estimadores lineales y candidatos de regularización."""

from sklearn.linear_model import Lasso, LogisticRegression, Ridge


def modelos_lineales(tarea):
    """Elige la familia lineal compatible con el tipo de objetivo."""
    if tarea == "Clasificación":
        return {"Logística": (LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]})}
    return {"Ridge": (Ridge(), {"alpha": [0.1, 1, 10]}),
            "Lasso": (Lasso(max_iter=5000), {"alpha": [0.01, 0.1, 1]})}
