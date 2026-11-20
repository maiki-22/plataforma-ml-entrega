"""Contrato de un modelo que puede participar en la comparación supervisada."""

from dataclasses import KW_ONLY, dataclass

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class Candidato:
    """Define la entrada de una familia en el catálogo supervisado.

    Parameters
    ----------
    estimador : BaseEstimator
        Modelo sin ajustar, compatible con los pipelines de scikit-learn.
    parametros : dict
        Hiperparámetros del estimador y valores que se probarán en cada pliegue.
    escalar : bool
        Estandariza los predictores numéricos con las estadísticas del entrenamiento.
    requiere_densa : bool, optional
        Convierte la salida del preprocesador en una matriz densa antes del ajuste.
    """

    estimador: BaseEstimator
    parametros: dict[str, list]
    _: KW_ONLY
    escalar: bool
    requiere_densa: bool = False
