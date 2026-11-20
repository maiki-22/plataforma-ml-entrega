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

| Ubicación | Responsabilidad prevista |
|---|---|
| main.py | Interfaz en español |
| core/ | Carga de tablas, análisis exploratorio y preprocesamiento |
| models/ | Contrato común y familias supervisadas y no supervisadas |
| evaluation/ | Métricas y figuras |
| data/ y reports/ | Ejemplos locales y salidas por ejecución |
| tests/ | Pruebas de ingesta, selección de tarea y diagnóstico |

Con el entorno activado, ejecutar las pruebas con `python -m unittest discover -s tests -v`.

## Preparación de predictores

Las variables numéricas usan la mediana de entrenamiento para completar faltantes y valores infinitos. Una columna numérica enteramente vacía en entrenamiento se conserva con valor cero. Los modelos sensibles a escala pueden activar la estandarización posterior. Las categorías se representan con indicadores binarios, incluida la categoría de ausencia cuando se observa durante el ajuste. Una categoría nueva en prueba produce ceros en los indicadores de esa columna.

Para comprobar el tratamiento de faltantes, categorías nuevas y estadísticas de entrenamiento, ejecutar `python -m unittest discover -s tests -p test_preprocesamiento.py -v`.

## Predicción con CART

Con una columna objetivo seleccionada, abrir **Modelo**, elegir la profundidad máxima y pulsar **Ejecutar análisis**. La semilla es 42. Se reserva el 20 % de las filas con objetivo para prueba, redondeando hacia arriba, con un mínimo de dos filas y, para clasificación, al menos tantas como clases. La partición de clasificación es estratificada. El árbol trabaja con las variables numéricas en su escala original.

- **Clasificación:** abrir `flores.csv` con la detección automática. Revisar la exactitud y la matriz de confusión, normalizada por clase real.
- **Regresión:** abrir `mediciones.tsv`, elegir manualmente `temperatura_c` como objetivo y mantener **Regresión**. Revisar MAE y RMSE en grados Celsius, R² y el gráfico de valores reales frente a predichos.

La tabla de importancias suma los indicadores categóricos correspondientes a una misma variable de origen. La figura a 300 DPI y el archivo `metricas.csv` se guardan en una carpeta por ejecución de `reports/`. Cambiar de vista conserva el resultado. Modificar la configuración del modelo requiere volver a ejecutar el análisis.

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

LVQ1, DANN y logística se ofrecen para clasificación. Ridge, Lasso, MARS y GAM se ofrecen para regresión. CART, Random Forest y k-NN admiten ambas tareas. Los datos sintéticos sirven para verificar los recorridos y su tamaño limita la interpretación de diferencias de rendimiento.

La comparación exporta métricas y parámetros elegidos dentro de `reports/comparacion-*/`. Para revisar el fundamento de los métodos especializados, consultar el [artículo de DANN](https://hastie.su.domains/Papers/dann_IEEE.pdf), la [implementación de MARS](https://pypi.org/project/mars-earth/) y la [referencia de GAM](https://pygam.readthedocs.io/en/latest/reference/_autosummary/pygam.pygam.LinearGAM.html).

## Interpretar y reproducir los reportes

**Modelo → Comparación** exporta curvas ROC y precisión-recall por clase (uno contra el resto) y microagregadas. El AUC se guarda en `auc.csv` y los puntos en `curvas_por_clase.csv`. Las curvas usan puntajes por clase: en DANN son proporciones de votos y en LVQ1 son cercanías relativas a los prototipos. El modelo con mayor puntuación de validación aporta las curvas de aprendizaje y, si existen exactamente dos predictores numéricos con variación, la superficie de decisión en unidades originales. La curva de aprendizaje muestra los tamaños factibles y mantiene fijos los hiperparámetros seleccionados.

En regresión, las figuras incluyen la distribución de residuales y la calibración lineal de las predicciones sobre los valores reales de prueba. Con al menos tres observaciones y variación del objetivo, el sombreado representa el intervalo puntual de confianza del 95 % para la media de esa recta bajo el supuesto de errores independientes, homocedásticos y aproximadamente normales. Para repetir una ejecución, comprobar en `configuracion.json` la huella SHA-256 del archivo, las opciones de lectura, la tarea, la semilla y la configuración del modelo.
