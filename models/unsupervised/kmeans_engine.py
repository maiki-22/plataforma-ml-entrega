"""Comparación de inicializaciones y elección reproducible de K."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_samples
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


def agrupar(tabla, k_maximo=8):
    """Evalúa de 2 a K grupos y usa el codo normalizado o la silueta."""
    datos = crear_preprocesador(tabla, escalar=True).fit_transform(tabla)
    if issparse(datos):
        datos = datos.toarray()
    limite = min(k_maximo, len(datos) - 1, len(np.unique(datos, axis=0)))
    if limite < 2:
        raise ValueError("Se necesitan tres filas y dos observaciones distintas como mínimo.")
    registros, ganadores = [], {}
    # Una sola hebra evita sobrecarga de OpenMP con las tablas de la demo.
    with threadpool_limits(limits=1):
        for k in range(2, limite + 1):
            for inicio in ("k-means++", "random"):
                modelo = KMeans(k, init=inicio, n_init=5, random_state=42).fit(datos)
                grupos = modelo.labels_
                siluetas = silhouette_samples(datos, grupos)
                registros.append({"K": k, "Inicio": inicio, "Inercia": modelo.inertia_,
                                  "Silueta": siluetas.mean(), "CHI": calinski_harabasz_score(datos, grupos),
                                  "DBI": davies_bouldin_score(datos, grupos)})
                if k not in ganadores or modelo.inertia_ < ganadores[k][0].inertia_:
                    ganadores[k] = modelo, siluetas
    ks = np.arange(1, limite + 1)
    inercias = np.array([np.square(datos - datos.mean(axis=0)).sum()] +
                        [ganadores[k][0].inertia_ for k in ks[1:]])
    # Distancia perpendicular a la cuerda entre los extremos normalizados.
    distancias = (1 - (ks - 1) / (limite - 1) - (inercias - inercias[-1]) /
                  (inercias[0] - inercias[-1])) / np.sqrt(2)
    k = int(ks[np.argmax(distancias)])
    motivo = "Mayor distancia a la recta entre extremos de la curva de inercia."
    if distancias.max() < 0.05:
        k = max(ganadores, key=lambda valor: ganadores[valor][1].mean())
        motivo = "Codo poco definido (distancia inferior a 0,05). Se eligió la mayor silueta."
    modelo, siluetas = ganadores[k]
    pca = PCA(n_components=min(2, datos.shape[1]), random_state=42)
    puntos = pca.fit_transform(datos)
    centros = pca.transform(modelo.cluster_centers_)
    if puntos.shape[1] == 1:
        puntos = np.column_stack([puntos, np.zeros(len(datos))])
        centros = np.column_stack([centros, np.zeros(k)])
    filas = pd.DataFrame(puntos, columns=["CP1", "CP2"], index=tabla.index)
    filas["Grupo"] = modelo.labels_ + 1
    filas["Silueta"] = siluetas
    registros.insert(0, {"K": 1, "Inicio": "Referencia", "Inercia": inercias[0]})
    return Agrupamiento(k, motivo, pd.DataFrame(registros), filas, centros)
