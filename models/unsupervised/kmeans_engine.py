"""Comparación de inicializaciones y elección reproducible de K."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, pairwise_distances, silhouette_samples
from threadpoolctl import threadpool_limits

from core.preprocessor import crear_preprocesador


@dataclass
class Agrupamiento:
    """Resultados calculados en el espacio transformado y su proyección."""
    k: int
    motivo: str
    metricas: pd.DataFrame
    filas: pd.DataFrame
    centroides: np.ndarray
    k_maximo: int
    semilla: int


# Por encima de este presupuesto se delega el cálculo por bloques a sklearn.
MAX_BYTES_DISTANCIAS = 32 * 1024**2


def _evaluar_candidatos(datos, limite, semilla):
    """Compara ambas inicializaciones con las mismas distancias euclidianas."""
    registros, ganadores = [], {}
    # Una sola hebra evita sobrecarga de OpenMP con las tablas de la demo.
    with threadpool_limits(limits=1):
        if len(datos) ** 2 * datos.dtype.itemsize <= MAX_BYTES_DISTANCIAS:
            entrada_silueta = pairwise_distances(datos, metric="euclidean")
            metrica = "precomputed"
        else:
            # sklearn calcula distancias por bloques cuando no se conserva la matriz.
            entrada_silueta = datos
            metrica = "euclidean"
        for k in range(2, limite + 1):
            for inicio in ("k-means++", "random"):
                modelo = KMeans(k, init=inicio, n_init=5, random_state=semilla).fit(datos)
                grupos = modelo.labels_
                siluetas = silhouette_samples(entrada_silueta, grupos, metric=metrica)
                registros.append({"K": k, "Inicio": inicio, "Inercia": modelo.inertia_,
                                  "Silueta": siluetas.mean(), "CHI": calinski_harabasz_score(datos, grupos),
                                  "DBI": davies_bouldin_score(datos, grupos)})
                if k not in ganadores or modelo.inertia_ < ganadores[k][0].inertia_:
                    ganadores[k] = modelo, siluetas
    return registros, ganadores


def _elegir_k(inercias, ganadores):
    """Localiza el codo o resuelve una curva plana con la silueta."""
    ks = np.arange(1, len(inercias) + 1)
    # Distancia perpendicular a la cuerda entre los extremos normalizados.
    k_normalizada = (ks - 1) / (ks[-1] - 1)
    inercia_normalizada = (inercias - inercias[-1]) / (inercias[0] - inercias[-1])
    distancias = (1 - k_normalizada - inercia_normalizada) / np.sqrt(2)
    k = int(ks[np.argmax(distancias)])
    motivo = "Mayor distancia a la recta entre extremos de la curva de inercia."
    if distancias.max() < 0.05:
        k = max(ganadores, key=lambda valor: ganadores[valor][1].mean())
        motivo = "Codo poco definido (distancia inferior a 0,05). Se eligió la mayor silueta."
    return k, motivo


def _proyectar(datos, modelo, siluetas, indice, semilla):
    """Proyecta las filas y sus centroides sobre los mismos ejes."""
    pca = PCA(n_components=min(2, datos.shape[1]), random_state=semilla)
    puntos = pca.fit_transform(datos)
    centros = pca.transform(modelo.cluster_centers_)
    if puntos.shape[1] == 1:
        puntos = np.column_stack([puntos, np.zeros(len(datos))])
        centros = np.column_stack([centros, np.zeros(modelo.n_clusters)])
    filas = pd.DataFrame(puntos, columns=["CP1", "CP2"], index=indice)
    filas["Grupo"] = modelo.labels_ + 1
    filas["Silueta"] = siluetas
    return filas, centros


def agrupar(tabla: pd.DataFrame, k_maximo: int = 8, semilla: int = 42) -> Agrupamiento:
    """Evalúa de 2 a K grupos y usa el codo normalizado o la silueta.

    Parameters
    ----------
    tabla : pandas.DataFrame
        Tabla de observaciones con nombres de columna únicos.
    k_maximo : int
        Límite de grupos solicitado, acotado por las observaciones distintas.
    semilla : int
        Semilla de la ejecución para las operaciones aleatorias.

    Returns
    -------
    Agrupamiento
        K seleccionado, métricas por inicio, asignaciones y proyección sobre los ejes de PCA.
    """
    datos = crear_preprocesador(tabla, escalar=True).fit_transform(tabla)
    if issparse(datos):
        datos = datos.toarray()
    limite = min(k_maximo, len(datos) - 1, len(np.unique(datos, axis=0)))
    if limite < 2:
        raise ValueError("Se necesitan tres filas y dos observaciones distintas como mínimo.")
    registros, ganadores = _evaluar_candidatos(datos, limite, semilla)
    inercia_inicial = np.square(datos - datos.mean(axis=0)).sum()
    inercias = np.array([inercia_inicial] + [modelo.inertia_ for modelo, _ in ganadores.values()])
    k, motivo = _elegir_k(inercias, ganadores)
    modelo, siluetas = ganadores[k]
    filas, centros = _proyectar(datos, modelo, siluetas, tabla.index, semilla)
    registros.insert(0, {"K": 1, "Inicio": "Referencia", "Inercia": inercias[0]})
    return Agrupamiento(k, motivo, pd.DataFrame(registros), filas, centros, k_maximo, semilla)
