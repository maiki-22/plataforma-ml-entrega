# Sistema modular y escalable de Machine Learning

Aplicación en Python y Streamlit para cargar datos tabulares, realizar análisis exploratorio y ejecutar modelos supervisados o agrupamiento con K-Means. Autor: Miguel Cortés.

## Informe técnico

[Descargar el informe técnico (PDF, 15 páginas)](Informe_tecnico.pdf). Incluye arquitectura y UML, metodología, resultados de defectos de acero y sismos, verificación auxiliar de regresión y referencias.

## Presentación de defensa

[Descargar la presentación (PDF, 9 diapositivas)](Defensa.pdf). Exportada desde Penpot en formato panorámico 16:9 para acompañar la defensa y las demostraciones en vivo.

## Instalación y ejecución

Requiere Python 3.12. Desde una terminal:

```bash
git clone --branch entrega https://github.com/maiki-22/plataforma-ml-entrega.git
cd plataforma-ml-entrega
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

Abrir la dirección local indicada por Streamlit. En Windows, crear el entorno con `py -3.12 -m venv .venv` y activarlo en PowerShell con `.\.venv\Scripts\Activate.ps1`; después ejecutar los mismos comandos de instalación y arranque.

## Demostración con los datasets de entrega

Los archivos están incluidos en `data/` y se seleccionan desde **Ejemplo local**. Su uso no requiere descargar datos adicionales.

### Clasificación de defectos de acero

1. Abrir `defectos_acero.csv` y mantener **Variable objetivo → Automática**. Se reconoce `clase` y se propone clasificación.
2. Revisar **Variables** y **Relaciones** para inspeccionar calidad y asociaciones.
3. En **Modelo → Árbol CART**, elegir profundidad 3 y pulsar **Ejecutar análisis**.
4. Interpretar la matriz de confusión normalizada y las importancias. Para semilla 42, se reservan 104 de las 518 filas; el árbol acierta 99 de las 104, con exactitud aproximada de 95.2 % y F1 macro de 0.907.
5. En **Comparación**, elegir candidatos y pulsar **Comparar modelos**. Revisar métricas de validación y prueba, parámetros, curvas y superficie de decisión.

El análisis distingue tres tipos de defectos previamente detectados usando área logarítmica y luminosidad. No evalúa la detección de placas sanas frente a defectuosas.

### Agrupamiento de sismos

1. Abrir `sismos_profundos_pacifico.tsv` y mantener **Automática**. La aplicación informa que no encuentra un objetivo claro y selecciona agrupamiento.
2. Revisar los cuatro atributos: latitud, longitud relativa al meridiano de 180°, profundidad y magnitud.
3. En **Agrupamiento**, mantener K máximo 8 y pulsar **Evaluar grupos**.
4. Revisar codo, siluetas y proyección PCA. Con semilla 42, el codo selecciona K=5 y la silueta media es aproximadamente 0.392; K=6 tiene una silueta ligeramente mayor, de 0.402.
5. Descargar las asignaciones por fila para interpretar perfiles en las variables originales.

Los grupos describen similitudes entre los 250 eventos del recorte; no constituyen pronósticos sísmicos ni una evaluación del riesgo.

## Cargar otros archivos

**Subir archivo** acepta CSV, TSV y Parquet. Para texto se admite UTF-8 y Latin-1, delimitador automático o explícito y archivos con o sin encabezado. Se validan nombres de columnas y estructura de filas.

El objetivo se sugiere por nombres conocidos, sufijos o una última columna categórica con pocas categorías repetidas. La detección es heurística: se puede corregir con **Elegir columna** y cambiar entre clasificación y regresión. **Sin objetivo** fuerza agrupamiento utilizando las columnas recibidas. Seleccionar una ruta no inicia por sí solo el entrenamiento.

Para probar regresión, abrir `mediciones.tsv`, elegir manualmente `temperatura_c` y seleccionar **Regresión**. Los ejemplos pequeños `flores.csv`, `mediciones.tsv` y los dos archivos Iris se conservan también para las pruebas y recorridos de la aplicación.

## Arquitectura

| Ubicación | Responsabilidad |
|---|---|
| `main.py` | Entrada interactiva y navegación |
| `interfaz/` | Controles, vistas y estado de sesión |
| `core/loader.py` | Ingesta, validación y sugerencia de ruta |
| `core/eda.py` | Faltantes, cardinalidad, varianza, asimetría, IQR, covarianza, Pearson y Spearman |
| `core/particion.py` | Separación de predictores/objetivo y reserva de prueba |
| `core/preprocessor.py` | Imputación, codificación y escalado |
| `models/base.py` | Contrato `fit`, `predict`, `score` y `save` |
| `models/supervised/` | Familias, candidatos y validación de hiperparámetros |
| `models/unsupervised/kmeans_engine.py` | Comparación de K e inicializaciones y proyección |
| `flujo.py` | Ajuste y evaluación del CART interactivo |
| `evaluation/` | Métricas, figuras y exportación |
| `data/`, `reports/`, `tests/` | Datos incluidos, resultados y pruebas |

## Modelos y preparación

| Tarea | Modelos disponibles |
|---|---|
| Clasificación | Logística, CART, Random Forest, k-NN, LVQ1 y DANN |
| Regresión | Ridge, Lasso, CART, Random Forest, k-NN, MARS y GAM |
| Agrupamiento | K-Means con inicialización aleatoria y K-Means++ |

Las numéricas completan faltantes e infinitos con la mediana aprendida; una columna numérica vacía en entrenamiento se conserva con cero. Las categóricas utilizan One-Hot, con tratamiento de ausencias y categorías nuevas. El escalado se activa para modelos sensibles a escala y se omite para CART y Random Forest.

En supervisado, prueba se reserva antes de aprender transformaciones. Comparación ajusta el pipeline completo dentro de hasta cinco pliegues comunes, estratificados para clasificación. Se maximiza exactitud en clasificación y se minimiza MAE en regresión. Cada configuración ganadora se reajusta con todo entrenamiento y se evalúa sobre la misma reserva de prueba. El CART interactivo usa directamente la profundidad elegida, sin búsqueda.

K-Means prepara la tabla explorada completa y compara cinco reinicios por cada inicialización y K. Conserva la menor inercia por K y detecta el codo mediante distancia a la cuerda entre extremos normalizados. Si la distancia máxima es inferior a 0.05, selecciona por silueta entre las soluciones conservadas. Calcula también Calinski-Harabasz y Davies-Bouldin. PCA se aplica después del agrupamiento para visualizar puntos y centroides.

## Reportes

Se generan carpetas por ejecución dentro de `reports/`, con PNG a 300 DPI, métricas CSV y configuración JSON según la ruta. Los resultados de ejecución no se incluyen en Git.

- Clasificación: exactitud, F1 macro y matriz normalizada; Comparación exporta además ROC/AUC y Precision-Recall por clase y microagregadas, y curvas de aprendizaje del candidato destacado por CV.
- Regresión: MAE, RMSE, R², reales frente a predichos y residuales. La banda corresponde al IC puntual del 95 % de la media de una recta de calibración, no a un intervalo predictivo de nuevas observaciones.
- Agrupamiento: inercia/codo, siluetas individuales, índices, PCA y asignaciones por fila.

La superficie de decisión de clasificación requiere exactamente dos predictores numéricos originales con variación. Para reproducir una ejecución, conservar archivo, opciones de lectura, semilla, parámetros y versiones. Los flujos `ejecutar_cart`, `comparar` y `agrupar` aceptan `semilla` desde Python.

## Pruebas

Con el entorno activado, desde la raíz:

```bash
python -m unittest discover -s tests -v
```

La suite cubre carga, enrutamiento, EDA, preparación sin fuga, modelos, exportación y recorridos de Streamlit.

## Procedencia de los datos

- **Acero:** Buscema, Terzi y Tastle (2010), [Steel Plates Faults, UCI](https://doi.org/10.24432/C5J88N), CC BY 4.0. Se conservan 518 observaciones de `K_Scatch`, `Stains` y `Dirtiness`; sus indicadores se combinan en `clase`. `LogOfAreas` y `Luminosity_Index` se renombran como predictores. [Copia tabular utilizada](https://raw.githubusercontent.com/iamMPAI/Steel-plates-faults/main/stell-faults.csv).
- **Sismos:** [USGS, ANSS ComCat](https://doi.org/10.5066/F7MS3QZH), mediante la [copia del catálogo profundo de Nathan Ridge / Live Atlas](https://raw.githubusercontent.com/liveatlasearth/earthquake-catalogs/main/data/deep_earthquakes_m4_5_plus_2016_2025.csv), versión 1.0, CC BY 4.0. Recorte de 250 eventos de 2024 UTC, magnitud ≥4.5 y profundidad ≥300 km, entre 32°S y 12°S y hasta 15° a cada lado de 180°. La longitud se centra en ese meridiano para evitar el salto entre 179°E y 179°O.
- **Iris:** versión distribuida por scikit-learn del conjunto de Fisher (1936), con encabezados en español. `iris_petalos.csv` conserva dos medidas y la especie; `iris_medidas.tsv` conserva cuatro medidas sin etiqueta. [Descripción](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html).
- **Flores y mediciones:** ejemplos sintéticos pequeños para verificar los recorridos de clasificación, regresión y agrupamiento.
