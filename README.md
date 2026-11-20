# Cuaderno de datos

Aplicación local en Python y Streamlit para cargar tablas, explorar sus distribuciones y relaciones, revisar una muestra y seleccionar la ruta de aprendizaje.

## Ejecutar en local

Desde la carpeta del repositorio, con Python 3.12 instalado, ejecutar en Linux o macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run main.py
```

En Windows, crear el entorno con `py -3.12 -m venv .venv` y activarlo desde PowerShell con `.\.venv\Scripts\Activate.ps1`. Los comandos de instalación y arranque son los mismos.

## Comprobar la carga y el enrutamiento

1. Abrir la dirección local que indica Streamlit. El panel lateral comienza con un ejemplo local. En `flores.csv` se muestran 36 filas, tres columnas y la ruta de clasificación por la columna `clase`.
2. Cambiar a `mediciones.tsv`. Las dos variables son mediciones sin etiqueta. La aplicación explica la ausencia de objetivo y selecciona agrupamiento con K-Means.
3. Elegir **Subir archivo** para probar CSV, TSV o Parquet. Para texto se admite UTF-8 o Latin-1. En **Opciones de lectura** se puede corregir el delimitador y desmarcar el encabezado para asignar nombres como `columna_1`.
4. En **Objetivo de aprendizaje**, usar **Elegir columna** para indicar el objetivo. Si un código numérico representa categorías, seleccionar **Clasificación**. **Sin objetivo** permite elegir agrupamiento explícitamente.

Los dos ejemplos son datos sintéticos creados para comprobar estos recorridos. El primero simula medidas de flores en centímetros y tres clases ficticias. El segundo simula temperatura y humedad en tres condiciones ambientales.

El tablero muestra filas, columnas, celdas vacías y valores atípicos del archivo activo. En **Resumen**, la vista previa se puede consultar junto a la distribución de una variable. Seleccionar otra columna actualiza el histograma o las frecuencias por categoría.

## Diagnóstico exploratorio

En **Variables**, la aplicación resume por columna el tipo, los faltantes y la cardinalidad de las variables no numéricas. Para las numéricas muestra cuartiles, varianza muestral, asimetría y recuento de atípicos según la regla de 1,5 rangos intercuartílicos. Como criterios exploratorios, señala varianza positiva de hasta 10⁻⁶ en la escala original y asimetría absoluta superior a 1. La varianza depende de las unidades de medida.

En **Relaciones**, elegir Pearson, Spearman, Covarianza o, cuando existan pares válidos, Dispersión. Junto a cada mapa están la matriz completa y la descarga del PNG. Los cálculos bivariados usan las filas con valores finitos en ambas columnas. Una correlación indefinida aparece vacía. La tabla de pares destacados usa Pearson positivo mayor que 0,85. La figura de dispersión selecciona el par con mayor correlación absoluta, que también puede ser negativa.

Los mapas anotados incluyen hasta ocho variables para facilitar la lectura. Las figuras se guardan como PNG a 300 DPI en una carpeta propia dentro de `reports/`. Cambiar el objetivo conserva el diagnóstico de la misma tabla.

## Organización

| Ubicación | Responsabilidad |
|---|---|
| main.py | Entrada de Streamlit y navegación del tablero |
| interfaz/ | Controles de archivo y objetivo, vistas de exploración y ejecución de modelos |
| core/ | Carga de tablas, análisis exploratorio, partición de entrenamiento y prueba y preprocesamiento |
| flujo.py | Entrenamiento y evaluación del árbol CART |
| models/ | Contrato común y familias supervisadas y no supervisadas |
| evaluation/ | Métricas, figuras y exportación de resultados por ejecución |
| data/ y reports/ | Ejemplos locales y salidas por ejecución |
| tests/ | Pruebas de ingesta, diagnóstico, modelos, exportación y recorridos en Streamlit |

Con el entorno activado, ejecutar las pruebas con `python -m unittest discover -s tests -v`.

## Preparación de predictores

Las variables numéricas usan la mediana de entrenamiento para completar faltantes y valores infinitos. Una columna numérica enteramente vacía en entrenamiento se conserva con valor cero. Los modelos sensibles a escala pueden activar la estandarización posterior. Las categorías se representan con indicadores binarios, incluida la categoría de ausencia cuando se observa durante el ajuste. Una categoría nueva en prueba produce ceros en los indicadores de esa columna.

Para comprobar el tratamiento de faltantes, categorías nuevas y estadísticas de entrenamiento, ejecutar `python -m unittest discover -s tests -p test_preprocesamiento.py -v`.

## Predicción con CART

Con una columna objetivo seleccionada, abrir **Modelo**, elegir la profundidad máxima y pulsar **Ejecutar análisis**. El tablero usa la semilla 42 por defecto. Se reserva el 20 % de las filas con objetivo para prueba, redondeando hacia arriba, con un mínimo de dos filas y, para clasificación, al menos tantas como clases. La partición de clasificación es estratificada. El árbol trabaja con las variables numéricas en su escala original.

- **Clasificación:** abrir `flores.csv` con la detección automática. Revisar la exactitud y la matriz de confusión, normalizada por clase real.
- **Regresión:** abrir `mediciones.tsv`, elegir manualmente `temperatura_c` como objetivo y mantener **Regresión**. Revisar MAE y RMSE en grados Celsius, R² y el gráfico de valores reales frente a predichos.

La tabla de importancias suma los indicadores categóricos correspondientes a una misma variable de origen. La figura a 300 DPI y el archivo `metricas.csv` se guardan en una carpeta por ejecución de `reports/`. Cambiar de vista conserva el resultado. Al volver a CART o a Comparación se recuperan las opciones seleccionadas. Cambiar la profundidad, los candidatos o el K máximo descarta el resultado de esa vista. Cambiar de archivo, de opciones de lectura, de objetivo o de tarea inicia un nuevo análisis.

## Agrupamiento

Abrir `mediciones.tsv` y pulsar **Agrupamiento → Evaluar grupos**. Se prueban inicializaciones aleatoria y K-Means++ (cinco reinicios cada una) sobre variables preparadas y numéricas estandarizadas. La curva usa la menor inercia por K y parte de K=1 como referencia. Se selecciona la mayor distancia perpendicular a la cuerda normalizada. Por debajo de 0,05 se usa la mayor silueta. CHI y DBI complementan la comparación. Las etiquetas, la silueta individual y las coordenadas PCA de cada fila se exportan junto con las figuras. Para una única dimensión la segunda coordenada se dibuja en cero.

## Comparar modelos supervisados

En **Modelo → Comparación**, seleccionar las familias y pulsar **Comparar modelos**. Se usa la misma reserva que en CART. Cada candidato ajusta el preprocesamiento exclusivamente en la parte de entrenamiento de cada pliegue. Una búsqueda en rejilla compara hasta cinco pliegues reproducibles, estratificados en clasificación. Se optimiza la exactitud en clasificación y el MAE en regresión. El mejor ajuste de cada familia se reentrena con todo el entrenamiento y se evalúa en las mismas filas de prueba. La columna CV corresponde a validación y las demás métricas a prueba.

| Familia | Candidatos de ajuste |
|---|---|
| Logística, Ridge y Lasso | Regularización mediante C o alpha |
| CART y Random Forest | Profundidad máxima |
| k-NN | Vecinos euclidianos con voto o media ponderados por distancia |
| LVQ1 | Tasa inicial de atracción o repulsión de prototipos por clase |
| DANN | Número de vecinos bajo la métrica discriminante local regularizada |
| MARS | Límite de términos aditivos de bisagra con construcción progresiva y poda |
| GAM | Penalización de suavidad de los splines aditivos |

Cada familia registra su estimador y su rejilla de hiperparámetros en un `Candidato`. El campo `escalar` indica si se estandarizan las variables numéricas y `requiere_densa` indica si el estimador necesita una matriz densa. Para incorporar un estimador a una familia existente, se añade su entrada en ese catálogo. El entrenamiento lee ambas propiedades en cada pliegue.

LVQ1, DANN y logística se ofrecen para clasificación. Ridge, Lasso, MARS y GAM se ofrecen para regresión. CART, Random Forest y k-NN admiten ambas tareas. Los datos sintéticos sirven para verificar los recorridos y su tamaño limita la interpretación de diferencias de rendimiento.

La comparación exporta métricas y parámetros elegidos dentro de `reports/comparacion-*/`. Para revisar el fundamento de los métodos especializados, consultar el [artículo de DANN](https://hastie.su.domains/Papers/dann_IEEE.pdf), la [implementación de MARS](https://pypi.org/project/mars-earth/) y la [referencia de GAM](https://pygam.readthedocs.io/en/latest/reference/_autosummary/pygam.pygam.LinearGAM.html).

## Interpretar y reproducir los reportes

**Modelo → Comparación** exporta curvas ROC y precisión-recall por clase (uno contra el resto) y microagregadas. El AUC se guarda en `auc.csv` y los puntos en `curvas_por_clase.csv`. Las curvas usan puntajes por clase: en DANN son proporciones de votos y en LVQ1 son cercanías relativas a los prototipos. El modelo con mayor puntuación de validación aporta las curvas de aprendizaje y, si existen exactamente dos predictores numéricos con variación, la superficie de decisión en unidades originales. La curva de aprendizaje muestra los tamaños factibles y mantiene fijos los hiperparámetros seleccionados.

En regresión, las figuras incluyen la distribución de residuales y la calibración lineal de las predicciones sobre los valores reales de prueba. Con al menos tres observaciones y variación del objetivo, el sombreado representa el intervalo puntual de confianza del 95 % para la media de esa recta bajo el supuesto de errores independientes, homocedásticos y aproximadamente normales. Para repetir una ejecución, comprobar en `configuracion.json` la huella SHA-256 del archivo, las opciones de lectura, la tarea, la semilla y la configuración del modelo.

Desde Python, `ejecutar_cart`, `comparar` y `agrupar` aceptan el argumento `semilla`. Los resultados supervisados conservan las filas efectivas en `particion`, con los índices y el orden que se usaron para entrenar y evaluar. Las métricas y las curvas de prueba utilizan esa misma reserva. El JSON de cada reporte registra la semilla efectiva.

```python
import pandas as pd

from flujo import ejecutar_cart

tabla = pd.read_csv("data/iris_petalos.csv")
resultado = ejecutar_cart(tabla, objetivo="clase", tarea="Clasificación", semilla=7)
print(resultado.semilla, resultado.metricas)
print(resultado.particion.objetivo_prueba.head())
```

## Informe y demostración

El [informe técnico](docs/entrega/Informe_tecnico.pdf) documenta el diseño, el diagrama de clases, la evaluación y sus límites. La [presentación en PDF](docs/entrega/Defensa.pdf) acompaña las dos ejecuciones en vivo.

1. Abrir **iris_petalos.csv** y mantener el objetivo automático `clase`. En **Modelo**, ejecutar CART con profundidad 3. Examinar la matriz normalizada y después abrir **Comparación** para consultar las curvas.
2. Abrir **iris_medidas.tsv**. En **Agrupamiento**, evaluar hasta K=8 y comparar la curva del codo con las siluetas. El archivo contiene las cuatro medidas de las mismas flores y omite la especie.

Ambos archivos derivan de Iris (R. A. Fisher, 1936), en la versión de scikit-learn incluida con el proyecto mediante sus dependencias. Los encabezados están en español y las medidas en centímetros. La vista etiquetada selecciona largo y ancho del pétalo; la vista sin etiquetas conserva también largo y ancho del sépalo. [Descripción y procedencia en scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_iris.html) · [Datos originales en UCI](https://archive.ics.uci.edu/dataset/53/iris).
