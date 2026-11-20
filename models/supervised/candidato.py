"""Contrato de un modelo que puede participar en la comparación supervisada."""

from dataclasses import KW_ONLY, dataclass

from sklearn.base import BaseEstimator


@dataclass(frozen=True)
class Candidato:
    """Define el estimador, su rejilla y la representación que necesita para entrenar."""

    estimador: BaseEstimator
    parametros: dict[str, list]
    _: KW_ONLY
    escalar: bool
    requiere_densa: bool = False
