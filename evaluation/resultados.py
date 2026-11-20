"""Datos compartidos por los resultados del aprendizaje supervisado."""

from dataclasses import dataclass

from core.particion import Particion


@dataclass(kw_only=True)
class ResultadoSupervisado:
    """Identifica qué se entrenó y conserva la reserva usada para evaluarlo."""

    objetivo: str
    tarea: str
    semilla: int
    particion: Particion

    @property
    def entrenamiento(self) -> int:
        """Cantidad de filas utilizadas para ajustar el modelo."""
        return len(self.particion.objetivo_entrenamiento)

    @property
    def prueba(self) -> int:
        """Cantidad de filas reservadas para evaluación."""
        return len(self.particion.objetivo_prueba)

    @property
    def omitidas(self) -> int:
        """Cantidad de filas descartadas por ausencia de objetivo."""
        return self.particion.filas_omitidas
