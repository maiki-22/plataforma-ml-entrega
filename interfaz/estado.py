"""Resultados de las vistas para un mismo archivo, objetivo y tarea."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import streamlit as st

from core.loader import Lectura, Ruta
from evaluation.resultados import ResultadoSupervisado
from models.unsupervised.kmeans_engine import Agrupamiento


class Publicacion(NamedTuple):
    """Resultado entrenado y archivos disponibles para la vista."""
    resultado: ResultadoSupervisado | Agrupamiento
    archivos: Path | dict[str, Path]


@dataclass
class EstadoVista:
    """Conserva los parámetros elegidos aunque el control deje de estar visible."""

    parametros: int | tuple[str, ...] | None = None
    publicacion: Publicacion | None = None

    def configurar(self, parametros: int | tuple[str, ...]) -> None:
        """Descarta la ejecución anterior cuando cambian las opciones del modelo.

        Parameters
        ----------
        parametros : int o tuple[str, ...]
            Profundidad de CART, K máximo o nombres de los candidatos seleccionados.
        """
        if self.parametros != parametros:
            self.parametros = parametros
            self.publicacion = None


@dataclass
class EstadoAnalisis:
    """Mantiene una ejecución independiente por vista dentro del contexto activo."""

    contexto: tuple[Lectura | None, str | None, str | None]
    cart: EstadoVista = field(default_factory=EstadoVista)
    comparacion: EstadoVista = field(default_factory=EstadoVista)
    grupos: EstadoVista = field(default_factory=EstadoVista)


def estado_del_analisis(lectura: Lectura | None = None, ruta: Ruta | None = None) -> EstadoAnalisis:
    """Recupera las vistas del contexto actual o empieza una sesión de análisis vacía.

    Parameters
    ----------
    lectura : Lectura, optional
        Identidad de la tabla activa, si se pudo cargar.
    ruta : Ruta, optional
        Objetivo y tarea seleccionados para la tabla.

    Returns
    -------
    EstadoAnalisis
        Estado de las tres vistas para el contexto actual.
    """
    contexto = (lectura, ruta.objetivo, ruta.tarea) if ruta else (lectura, None, None)
    estado = st.session_state.get("estado_analisis")
    if estado is None or estado.contexto != contexto:
        estado = EstadoAnalisis(contexto)
        st.session_state["estado_analisis"] = estado
    return estado
