"""Estimadores lineales y candidatos de regularización."""

from sklearn.linear_model import Lasso, LogisticRegression, Ridge

from models.supervised.candidato import Candidato


def modelos_lineales(tarea):
    """Elige la familia lineal compatible con el tipo de objetivo.

    Parameters
    ----------
    tarea : str
        Tarea supervisada, «Clasificación» o «Regresión».

    Returns
    -------
    dict[str, Candidato]
        Candidatos identificados por su nombre visible.
    """
    if tarea == "Clasificación":
        return {"Logística": Candidato(LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}, escalar=True)}
    return {
        "Ridge": Candidato(Ridge(), {"alpha": [0.1, 1, 10]}, escalar=True),
        "Lasso": Candidato(Lasso(max_iter=5000), {"alpha": [0.01, 0.1, 1]}, escalar=True),
    }
