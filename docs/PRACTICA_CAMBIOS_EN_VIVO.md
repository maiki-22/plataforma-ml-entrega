# Práctica de código y preguntas para la defensa de Machine Learning

La defensa y la modificación en vivo representan el 50 % de la pauta. La preparación debe concentrarse en localizar una responsabilidad, cambiarla sin romper el flujo y demostrar el efecto con un ejemplo pequeño. Los ejercicios siguientes usan los nombres y el comportamiento del código disponible en este repositorio.

Las prioridades son una estimación basada en la pauta, no una predicción de las preguntas del docente. Ninguna lista cubre todos los cambios posibles: dominar los recorridos permite resolver variantes.

Complementa esta práctica con [la guía del recorrido del código](GUIA_DEFENSA.md) y [los ejemplos contextualizados](FLUJOS_Y_RESULTADOS_DATASETS_REALES.md). Aquí las soluciones son propuestas para ensayar; no están aplicadas al programa.

## 1 Cómo practicar una modificación en cinco minutos

1. **Primeros 30 segundos:** reformula el pedido y define un resultado observable. Por ejemplo: «Agregaré F1 macro a la evaluación; deberá aparecer también en el CSV».
2. **Hasta el minuto 1:** localiza la función responsable y sus consumidores. Busca el símbolo, no memorices números de línea.
3. **Minutos 1 a 3:** implementa el cambio mínimo completo. Si agregas una opción, sigue su recorrido hasta el estimador y el reporte.
4. **Minuto 3 a 4:** ejecuta un ejemplo cuyo resultado puedas anticipar o inspecciona el parámetro efectivo.
5. **Minuto 4 a 5:** muestra el efecto, explica por qué es correcto y menciona una condición de borde pertinente.

Durante cada ensayo responde en voz alta: **qué entra, qué sale, qué se aprende y en qué conjunto se aprende**. Si el cambio mejora una métrica, explica también qué evidencia permite atribuirle esa mejora.

Desde la raíz, activa el entorno y busca responsabilidades:

```bash
source .venv/bin/activate
rg -n 'def separar_datos|def evaluar_supervisado|def comparar|def _elegir_k' core models evaluation
python -m streamlit run main.py
```

Practica sobre una copia de trabajo. Este repositorio ya puede contener cambios propios: evita restauraciones globales de Git. Guarda tus ejercicios por separado y vuelve a una base conocida entre ellos.

## 2 Qué practicar primero

| Prioridad | Pedido posible del docente | Tiempo de práctica | Responsabilidad principal |
|---|---|---|---|
| Alta | Agregar F1 macro, recall o exactitud balanceada | 2–3 min | `evaluation/metrics.py::evaluar_supervisado` |
| Alta | Cambiar la rejilla de profundidad o vecinos | 2–3 min | Catálogos en `models/supervised/` |
| Alta | Cambiar el porcentaje de prueba conservando estratificación | 3–5 min | `core/particion.py::separar_datos` |
| Alta | Incorporar un nombre de objetivo o corregir una detección | 2–4 min | `core/loader.py` |
| Alta | Elegir K por silueta en lugar del codo | 2–4 min | `kmeans_engine.py::_elegir_k` |
| Alta | Explicar o reescribir el cálculo cuadrático de DANN | 3–5 min | `prototype_models.py::DANN._votos` |
| Alta | Mostrar profundidad efectiva o cantidad de hojas | 2 min | `interfaz/modelos.py::mostrar_modelo` |
| Media | Detectar también correlaciones negativas fuertes | 2–3 min | `interfaz/exploracion.py::mostrar_relaciones` |
| Media | Pasar de imputación por mediana a media | 2–3 min | `core/preprocessor.py::crear_preprocesador` |
| Media | Agregar un clasificador al catálogo | 3–5 min | `linear_models.py` o una nueva familia |
| Media | Guardar y recuperar el pipeline entrenado | 3–5 min | `ModeloBase.save` o estimador de la búsqueda |
| Media | Exponer un parámetro nuevo desde Streamlit | 5–10 min | Interfaz, estado, flujo, modelo y exportación |
| Avanzada | Seleccionar modelos por F1 macro y adaptar las curvas | 5–10 min | Comparación y `evaluation/reportes.py` |
| Avanzada | Permitir varios prototipos LVQ por clase | 10–20 min | `LVQ1.fit`, `predict` y `predict_proba` |
| Avanzada | Quitar variables según entrenamiento dentro de CV | 10–20 min | Transformador ajustable dentro del pipeline |

Los tiempos son metas para ensayar, no garantías. Las tareas avanzadas sirven para practicar cómo explicar el alcance antes de editar.

## 3 Mapa para encontrar el lugar del cambio

| Si cambia… | Empieza por… | Luego comprueba… |
|---|---|---|
| La lectura de bytes, formatos o delimitadores | `core/loader.py::cargar_tabla`, `_leer_texto` | `interfaz/entrada.py`, `tests/test_ingesta.py` |
| La decisión supervisado o agrupamiento | `seleccionar_ruta`, `_detectar_objetivo` | Mensaje y elección manual de tarea en `mostrar_ruta` |
| El diagnóstico exploratorio | `core/eda.py::analizar_tabla` | Tablas en `interfaz/exploracion.py` y figuras en `evaluation/visualizer.py` |
| Las filas de entrenamiento y prueba | `core/particion.py::separar_datos` | Alineación X/y y viabilidad de CV |
| Imputación, codificación o escala | `core/preprocessor.py::crear_preprocesador` | Ajuste dentro del pipeline y categorías nuevas |
| CART interactivo | `flujo.py::ejecutar_cart`, `ModeloCART` | Widget, estado, resultado y `exportar_cart` |
| La búsqueda de un modelo | Función de catálogo que construye `Candidato` | `ModeloValidado.fit`, `best_params_`, `cv_results_` |
| La métrica de selección | `models/supervised/comparacion.py::comparar` | Nombre de columna CV y curvas de aprendizaje |
| Las métricas de prueba | `evaluation/metrics.py::evaluar_supervisado` | Tarjetas, resumen y CSV |
| K y sus índices | `models/unsupervised/kmeans_engine.py` | `resultado.k`, motivo, métricas y asignaciones |
| Un gráfico o su resolución | `evaluation/visualizer.py`, `evaluation/reportes.py` | Archivo nuevo, ejes, leyenda y DPI |
| Un resultado que no se actualiza | `interfaz/estado.py::EstadoVista.configurar` | Todos los parámetros nuevos dentro de la identidad de la vista |

**Dos caminos distintos:** la profundidad de CART interactivo pasa por `ModeloCART`; la profundidad de CART en Comparación sale del catálogo `modelos_de_arbol`. Cambiar uno no cambia automáticamente el otro.

## 4 Ejercicios resueltos para ensayar

Oculta las soluciones, intenta resolver el enunciado con cronómetro y luego compara. Haz cada ejercicio por separado.

### Ejercicio 1 Agregar F1 macro a las métricas

**Pedido:** «La exactitud puede ocultar errores en clases minoritarias. Agrega F1 macro».

En `evaluation/metrics.py`, incorpora `f1_score` al import de `sklearn.metrics` y cambia el retorno de clasificación:

```python
return {
    "Exactitud": accuracy_score(reales, predicciones),
    "F1 macro": f1_score(reales, predicciones, average="macro", zero_division=0),
}, detalle
```

**Comprobación mínima:** para reales `['a', 'a', 'a', 'b']` y predicciones `['a', 'a', 'a', 'a']`, la exactitud es `0.75` y F1 macro es aproximadamente `0.428571`. Ejecuta CART y Comparación: ambas rutas consumen este diccionario y exportan sus métricas.

**Explicación:** F1 combina precisión y recall de cada clase; macro promedia los resultados dando igual peso a cada clase. `zero_division=0` define el valor cuando no se puede calcular la razón correspondiente. Agregar esta métrica al reporte conserva `accuracy` como criterio de la búsqueda actual.

**Variante:** agrega `balanced_accuracy_score`. En el mismo ejemplo debe dar `0.5`: promedia recall por clase.

### Ejercicio 2 Ampliar los hiperparámetros de CART

**Pedido:** «Prueba profundidad 3 y un mínimo de cinco observaciones por hoja».

En `models/supervised/tree_models.py::modelos_de_arbol`, reemplaza la rejilla del candidato CART:

```python
{
    "max_depth": [2, 3, 4, None],
    "min_samples_leaf": [1, 5],
}
```

**Comprobación:** ejecuta Comparación solo con CART e inspecciona:

```python
busqueda = resultado.ajustes["CART"]
print(busqueda.best_params_)
print(len(busqueda.cv_results_["params"]))  # 8 combinaciones
```

Con cinco pliegues son 40 ajustes de validación más el reajuste final de la combinación ganadora. `ModeloValidado` antepone `modelo__` para dirigir los parámetros al estimador dentro del pipeline.

**Cuidado:** el gráfico actual de hiperparámetros usa la primera clave de la rejilla. Con dos parámetros mezcla combinaciones que comparten profundidad; para interpretarlo correctamente habría que separar curvas por `min_samples_leaf` o usar un mapa de calor.

### Ejercicio 3 Cambiar la reserva al treinta por ciento

**Pedido:** «Usa 30 % de prueba y mantén la estratificación».

Para un cambio fijo, sustituye `0.2` por `0.3` en `core/particion.py::separar_datos`. Conserva el mínimo de dos filas, el mínimo por número de clases y `stratify=estratos`.

**Comprobación:** con `iris_petalos.csv`, sin objetivos ausentes, deben quedar 105 filas de entrenamiento y 45 de prueba. Revisa tamaños, recuentos por clase y alineación de índices de cada X con su y.

**Explicación:** primero se reserva prueba; mediana, escalado y categorías se aprenden después con entrenamiento. Cambiar la reserva modifica también los pliegues disponibles y la evidencia para comparar modelos.

**Variante más completa:** agregar `fraccion_prueba=0.2` a la firma, validar `0 < fraccion_prueba < 1` y propagarlo desde los flujos. Además comprueba que el tamaño calculado deje entrenamiento suficiente: en clasificación, al menos una fila por clase para dividir y más para hacer CV. El código actual no expone esta fracción en la interfaz.

### Ejercicio 4 Reconocer un objetivo llamado diagnostico

**Pedido:** «El dataset nuevo usa `diagnostico` como etiqueta».

Agrega `"diagnostico"` a `NOMBRES_OBJETIVO` en `core/loader.py`.

**Comprobación:** construye una tabla con columnas `x`, `diagnostico`, `otra_medida`; usa valores de texto en el objetivo. Debe seleccionar `diagnostico` aunque no sea la última columna. Si agregas también una columna `clase`, debe explicar la ambigüedad y enrutar a agrupamiento.

**Explicación:** el nombre es una heurística. El sistema no puede deducir con certeza el propósito predictivo de cualquier columna. El modo manual permite corregir la elección y el modo sin objetivo fuerza agrupamiento.

**Estado actual que debes conocer:** también se reconocen sufijos `_target` y `_objetivo`; si no hay coincidencias por nombre, se examina la última columna categórica con pocas categorías repetidas. Los objetivos numéricos enteros repetidos pueden sugerir clasificación. No implementes de nuevo algo que ya existe.

### Ejercicio 5 Incluir correlaciones negativas fuertes

**Pedido:** «Una correlación de −0.95 también puede indicar redundancia. Muéstrala».

En `interfaz/exploracion.py::mostrar_relaciones`:

```python
altas = diagnostico.pares[
    diagnostico.pares["Pearson"].abs() > UMBRAL_CORRELACION
]
```

Cambia también el texto «Pares con correlación positiva alta» por «Pares con correlación alta en valor absoluto». El diagnóstico ya conserva correlaciones negativas y ordena por magnitud; el filtro de la vista es el que actualmente las excluye.

**Comprobación:** `x=[1,2,3,4]`, `z=[-1,-2,-3,-4]` produce Pearson `−1` y debe aparecer. No tomes valor absoluto de la matriz completa: conservar el signo ayuda a interpretar la relación.

**Explicación:** la correlación por pares detecta algunas redundancias, pero no todas las dependencias multivariadas. Una correlación baja tampoco prueba independencia.

### Ejercicio 6 Cambiar imputación sin provocar fuga

**Pedido:** «Usa la media en vez de la mediana para las numéricas».

En `crear_preprocesador`, cambia únicamente `SimpleImputer(strategy="median", ...)` por `SimpleImputer(strategy="mean", ...)`.

**Comprobación:** con entrenamiento `x=[1,2,100,None]`, el faltante debe imputarse con `103/3`, no con `2`. Transformar prueba con valores extremos no debe alterar `statistics_` del imputador ajustado.

**Explicación:** media y mediana se aprenden en `fit`. La mediana resiste mejor valores extremos. El lugar del imputador dentro del pipeline es lo que conserva la separación entre ajuste y evaluación.

**Variante:** sustituye `StandardScaler` por `RobustScaler` para explorar un escalado basado en mediana y rango intercuartílico. Compara por CV; no prometas que mejora todos los modelos.

### Ejercicio 7 Seleccionar K mediante silueta

**Pedido:** «Ignora el codo y elige siempre la mayor silueta».

Para practicar una sustitución mínima, conserva la firma de `_elegir_k` y reemplaza su cuerpo por:

```python
k = max(ganadores, key=lambda valor: ganadores[valor][1].mean())
return k, "Mayor silueta media entre las soluciones conservadas por K."
```

**Comprobación:** ejecuta `agrupar` y compara `resultado.k` con el máximo de la silueta entre las soluciones ganadoras por K. `ganadores[k]` contiene el modelo de menor inercia entre inicializaciones y sus siluetas; no incluye todas las filas de `resultado.metricas`.

**Explicación:** esta variante elige K entre los ganadores por inercia. Si el pedido fuera maximizar silueta sobre todas las parejas `(K, inicio)`, habría que conservar y seleccionar esas soluciones también. K=1 no participa porque la silueta requiere al menos dos grupos.

**Condición de borde:** si hay empate, `max` conserva el primer máximo encontrado; en la construcción actual los K se insertan de menor a mayor.

### Ejercicio 8 Mostrar la complejidad efectiva del árbol

**Pedido:** «Muestra profundidad real y número de hojas».

En `interfaz/modelos.py::mostrar_modelo`, después de recuperar `resultado, figura`:

```python
st.caption(
    f"Profundidad efectiva: {resultado.modelo.arbol.get_depth()} · "
    f"Hojas: {resultado.modelo.arbol.get_n_leaves()}"
)
```

**Comprobación:** entrena de nuevo y compara con `resultado.profundidad`. La profundidad efectiva puede ser menor que el máximo permitido. Este cambio solo informa sobre el ajuste; no necesita cambiar el entrenamiento.

### Ejercicio 9 Incorporar un clasificador al catálogo

**Pedido:** «Agrega un clasificador lineal por descenso de gradiente».

En `linear_models.py`, importa `SGDClassifier` y agrega esta entrada al diccionario de clasificación junto a Logística:

```python
"SGD logística": Candidato(
    SGDClassifier(loss="log_loss", random_state=42, max_iter=2000),
    {"alpha": [0.001, 0.01]},
    escalar=True,
),
```

**Comprobación:** debe aparecer en el multiselector; ejecuta solo ese candidato y verifica tanto métricas como exportación ROC/PR. La configuración `loss="log_loss"` permite `predict_proba`, que utiliza el reportador actual. Comprueba también posibles avisos de convergencia.

**Explicación:** `Candidato` declara estimador, rejilla y necesidades de preparación; `ModeloValidado` aporta el pipeline y la búsqueda comunes. Para respetar una semilla configurable habría que pasar `semilla` a `modelos_lineales` desde `catalogo`, en lugar del valor fijo del ejemplo.

**Pregunta asociada:** «¿Podrías agregar `LinearSVC` igual de rápido?» El entrenamiento sí puede integrarse, pero la exportación actual llama `predict_proba`. Habría que adaptar el reportador a `decision_function` o incorporar calibración dentro del entrenamiento.

### Ejercicio 10 Guardar el modelo y verificar predicciones

**Pedido:** «Entrena, guarda el modelo y úsalo sin reentrenar».

Desde la raíz, con el entorno activo:

```python
from pathlib import Path
from tempfile import TemporaryDirectory
import joblib
import numpy as np
import pandas as pd
from flujo import ejecutar_cart

tabla = pd.read_csv("data/iris_petalos.csv")
resultado = ejecutar_cart(tabla, "clase", "Clasificación")
x = resultado.particion.predictores_prueba
with TemporaryDirectory() as temporal:
    ruta = Path(temporal) / "modelo.joblib"
    resultado.modelo.save(ruta)
    recuperado = joblib.load(ruta)
    np.testing.assert_array_equal(
        resultado.modelo.predict(x), recuperado.predict(x)
    )
```

**Explicación:** guardar el objeto CART conserva también el pipeline ajustado. Guardar solo `arbol` perdería la mediana, las categorías y el orden transformado de columnas. Para Comparación puede guardarse `resultado.ajustes[nombre].best_estimator_`, que ya contiene su preparación. Recupera archivos de confianza: cargar objetos con joblib puede ejecutar código.

### Ejercicio 11 Llevar un parámetro de la pantalla al árbol

**Pedido:** «Permite elegir `min_samples_leaf` en CART interactivo».

Este ejercicio mide integración. Sigue esta secuencia:

1. Añade el parámetro con valor predeterminado `1` a `ModeloCART.__init__`, preferentemente después de los argumentos existentes o como argumento solo por nombre. Pásalo al constructor del árbol.
2. Añádelo a `ejecutar_cart` y transmite el valor por nombre. Conserva las llamadas actuales con semilla posicional.
3. Añádelo al resultado para conservar qué configuración produjo las métricas.
4. Incorpora el widget en `mostrar_modelo`; conserva sus dos valores en `estado.configurar((profundidad, minimo))` y adapta la recuperación del estado anterior.
5. Amplía la anotación de tipos de `EstadoVista` para aceptar la tupla de enteros.
6. Registra el valor en `configuracion.json` mediante `exportar_cart`.

**Comprobación:** `resultado.modelo.arbol.get_params()["min_samples_leaf"]` debe ser el valor elegido. Cambiarlo debe invalidar la publicación anterior. Tras entrenar, el JSON debe coincidir con el estimador efectivo.

La [sección 9 de la guía de defensa](GUIA_DEFENSA.md#9-cambios-para-practicar) contiene un desarrollo de esta variante. Si solo te solicitan cambiar el valor fijo, modificar el constructor es suficiente; si piden control desde la pantalla, recorre toda la cadena.

### Ejercicio 12 Seleccionar por F1 macro de forma coherente

**Pedido:** «Usa F1 macro para elegir hiperparámetros y adapta los reportes».

El cambio abarca más que una cadena de texto:

1. En `comparar`, usa `criterio="f1_macro"` para clasificación.
2. Cambia «Exactitud CV» por «F1 macro CV» en el resumen.
3. Añade F1 macro de prueba a `evaluar_supervisado` como en el ejercicio 1.
4. Revisa `evaluation/reportes.py::_aprendizaje`: `modelo.score` sigue devolviendo exactitud. Para medir el mismo criterio usa `get_scorer(ajuste.scoring)` sobre entrenamiento y validación, y etiqueta las curvas como `1 − F1 macro` si graficas su complemento.
5. `1 - resultados["mean_test_score"]` pasa a representar el complemento de F1 macro; actualiza ese eje. No lo llames proporción de observaciones mal clasificadas.

**Comprobación:** muestra `busqueda.scoring`, `best_score_`, la columna de CV y los ejes. Usa datos desbalanceados para comprobar que el criterio puede cambiar el ajuste elegido; no necesitas que cambie para demostrar que se aplicó.

**Explicación:** selección por validación y evaluación de prueba son operaciones distintas. El reportador elige el mejor candidato por `best_score_`, por lo que seguirá el nuevo criterio. No elijas la familia ganadora por la métrica de prueba.

## 5 DANN y LVQ para explicar en la pizarra

### La forma cuadrática de DANN

El código actual usa:

```python
distancias = np.sum((diferencias @ metrica) * diferencias, axis=1)
```

Con `n` observaciones de entrenamiento y `p` predictores transformados:

| Expresión | Forma | Significado |
|---|---|---|
| `x` | `(p,)` | Observación consultada |
| `self.datos_` | `(n,p)` | Entrenamiento almacenado |
| `diferencias` | `(n,p)` | Cada fila es `x_i − x` |
| `metrica` | `(p,p)` | Matriz local que pondera direcciones |
| `diferencias @ metrica` | `(n,p)` | Diferencias transformadas |
| Producto elemento a elemento | `(n,p)` | Contribuciones de cada coordenada |
| Suma por `axis=1` | `(n,)` | Una distancia cuadrática por observación |

Para cada fila:

```text
d_i² = (x_i − x)ᵀ M(x) (x_i − x)
     = Σ_j Σ_k diferencias[i,j] M[j,k] diferencias[i,k]
```

La expresión equivalente de la pauta es:

```python
np.einsum("ij,jk,ik->i", diferencias, metrica, diferencias)
```

`i` identifica la fila y queda en la salida. `j` y `k` recorren coordenadas y se suman. No se está sumando entre observaciones. Aunque el cálculo matricial intermedio puede escribirse como `(d M)·d`, el escalar coincide con la forma cuadrática.

**Ejemplo para calcular a mano:** `d=[1,2]`, `M=[[2,0],[0,0.5]]` da `d²=2×1²+0.5×2²=4`. Con `M=I`, se recupera la distancia euclidiana al cuadrado.

**Ejercicio en vivo:** reemplaza temporalmente la expresión por un bucle y verifica equivalencia:

```python
distancias_lentas = np.array([
    diferencia @ metrica @ diferencia
    for diferencia in diferencias
])
np.testing.assert_allclose(distancias, distancias_lentas)
```

No hace falta extraer raíz cuadrada para ordenar vecinos: es una transformación monótona de distancias no negativas.

### De dónde sale la métrica local

`DANN._votos` primero obtiene un entorno por distancia euclidiana. En ese entorno estima:

```text
W = Σ_c (1/m) Σ_{i en c} (x_i − μ_c)(x_i − μ_c)ᵀ
B = Σ_c (m_c/m) (μ_c − μ)(μ_c − μ)ᵀ
A = (W + 10⁻⁵ I)⁻¹
M = A B A + epsilon A
```

`m` es el tamaño del entorno y `m_c` el tamaño de la clase dentro de él. `W` mide dispersión interna y `B` separación de medias. La regularización de `W` permite resolver el sistema cuando hay columnas constantes o redundantes. `epsilon` agrega un componente definido positivo si es positivo y el resto conserva las propiedades esperadas de covarianza.

No llames a `M` simplemente «la covarianza». El programa construye una **métrica discriminante local** a partir de matrices de dispersión. La notación `Sigma(x_0)` de la pauta debe relacionarse con la fórmula concreta que se está usando.

**Preguntas para practicar:** ¿por qué se recalcula para cada consulta?, ¿qué pasa si el entorno solo tiene una clase?, ¿qué ocurre con `epsilon=0`?, ¿por qué `entorno` y `vecinos` son parámetros diferentes? Si `B=0`, el término `epsilon A` mantiene distancias útiles cuando `epsilon>0`.

### Actualización LVQ1

El código aprende un prototipo por clase, inicializado en su media. Para cada observación selecciona el prototipo más cercano y aplica:

```text
w_nuevo = w + signo × α_t × (x − w)
signo = +1 si coincide la clase; −1 si no coincide
α_t = tasa × (1 − epoca/epocas)
```

Con `w=[0,0]`, `x=[2,4]`, `α=0.1`: atracción produce `[0.2,0.4]` y repulsión `[-0.2,-0.4]`. La tasa disminuye durante las épocas ejecutadas; con el rango actual no llega exactamente a cero.

**Variante avanzada:** varios prototipos por clase exige guardar `clase_de_prototipo`. Hoy el índice del prototipo coincide con el índice de clase. Al romper esa equivalencia deben cambiar tanto la regla de atracción como la conversión de ganadores a etiquetas y la agregación de puntajes por clase.

## 6 Preguntas orales y respuestas que debes poder demostrar

| Pregunta probable | Núcleo de una respuesta correcta |
|---|---|
| ¿Dónde está la POO? | `ModeloBase` define el contrato abstracto; `ModeloCART` y `ModeloValidado` lo implementan. Los adaptadores LVQ1, DANN y GAM siguen el contrato de estimadores de scikit-learn. |
| ¿Todos los modelos heredan de tu clase base? | No. Los estimadores del catálogo se componen dentro de `ModeloValidado`. K-Means tiene otro flujo y resultado. Explica la arquitectura real. |
| ¿Qué evita duplicar entrenamiento por familia? | `Candidato` declara diferencias y `ModeloValidado` comparte preparación, búsqueda y predicción. |
| ¿Por qué hay funciones y clases? | Las funciones calculan o coordinan tareas concretas; los estimadores conservan parámetros aprendidos y exponen operaciones comunes. |
| ¿Dónde se elimina el target de X? | En `separar_datos`, antes de crear el pipeline. Compruébalo con las columnas de `predictores_entrenamiento`. |
| ¿Cómo evitas fuga de datos? | Prueba se separa primero. Cada ajuste de CV aprende su propia preparación. Muestra el pipeline y las estadísticas aprendidas. |
| ¿El EDA completo produce fuga por sí solo? | Describir la tabla no ajusta el modelo. Usar el test observado para decidir variables o hiperparámetros contamina la evaluación; las reglas aprendidas deben validarse dentro de entrenamiento. |
| ¿Qué diferencia hay entre fit y transform? | `fit` aprende parámetros; `transform` aplica los ya aprendidos. En prueba se transforma, no se reajusta. |
| ¿Por qué escalar k-NN y Ridge? | La escala afecta distancias y penalizaciones. CART usa cortes y normalmente no necesita estandarización. |
| ¿Qué haces con categorías nuevas? | `OneHotEncoder(handle_unknown="ignore")` genera ceros en los indicadores de esa columna. No aprende categorías de prueba. |
| ¿Por qué una etiqueta entera puede ser clasificación? | Los números pueden codificar clases. La heurística observa repetición y cardinalidad; la elección manual corrige casos ambiguos. |
| ¿Puedes detectar perfectamente cualquier target? | No. El propósito de una columna requiere contexto. Se usan reglas explícitas, mensajes y corrección manual. |
| ¿Por qué CV estratificada? | Conserva aproximadamente proporciones de clases. El número de pliegues está acotado por la clase menos frecuente de entrenamiento. |
| ¿Usas estratificación en regresión? | La ruta actual usa `KFold` con mezcla. En clasificación usa `StratifiedKFold`. |
| ¿Por qué se maximiza un MAE negativo? | La búsqueda maximiza puntajes. Maximizar `−MAE` equivale a minimizar MAE; el resumen cambia el signo para mostrar error positivo. |
| ¿score siempre es la métrica de búsqueda? | No. `.score()` del estimador es exactitud o R²; `GridSearchCV` usa el criterio `scoring` indicado. |
| ¿Por qué accuracy puede engañar? | Predecir siempre la clase mayoritaria puede dar gran exactitud y recall cero en otra clase. Muestra el ejemplo del ejercicio 1. |
| ¿Cómo se normaliza la matriz de confusión? | Por clase real, `normalize="true"`. Sus filas con soporte suman uno; la diagonal da recall por clase. |
| ¿ROC usa las clases predichas? | Usa puntajes por clase y varía umbrales. El código calcula uno contra el resto y microagregación. |
| ¿Qué diferencia hay entre macro y micro? | Macro da igual peso a cada clase; micro reúne decisiones antes de calcular. En ROC/PR multiclase se agregan indicadores uno contra el resto. |
| ¿Los puntajes LVQ son probabilidades calibradas? | No. Son cercanías normalizadas. DANN devuelve proporciones de votos. Sirven como puntajes, pero calibración es otra propiedad. |
| ¿R² puede ser negativo? | Sí, puede rendir peor que predecir la media del objetivo evaluado. No está limitado inferiormente por cero. |
| ¿RMSE y MAE tienen unidades? | Ambas tienen las unidades del objetivo. RMSE penaliza más los errores grandes. R² es adimensional. |
| ¿La banda de regresión cubre futuras observaciones? | La figura actual muestra un IC puntual de la media de una recta de calibración, bajo sus supuestos. No es un intervalo predictivo del modelo ML. |
| ¿Una importancia alta demuestra causalidad? | No. Resume contribución a divisiones del árbol. Correlación entre predictores y cardinalidad pueden afectar su interpretación. |
| ¿Ridge y Lasso qué penalizan? | Ridge usa suma de coeficientes al cuadrado; Lasso usa suma de valores absolutos y puede llevar coeficientes a cero. |
| ¿MARS y GAM son lo mismo? | MARS construye funciones de bisagra adaptativas y poda; GAM suma funciones suaves. Aquí MARS usa `max_degree=1` y GAM es de regresión con splines. |
| ¿GAM está disponible para clasificación? | No en este catálogo. Se usa `LinearGAM` para regresión; un GAM clasificador necesitaría otro adaptador. |
| ¿Por qué disminuye la inercia al aumentar K? | El óptimo con más centroides no puede empeorar el óptimo de la suma de cuadrados. Ajustes con reinicios finitos pueden no alcanzar esos óptimos. |
| ¿Cómo seleccionas K realmente? | Codo por distancia a la cuerda normalizada, umbral 0.05 y respaldo por silueta; para cada K se conserva la inicialización con menor inercia. |
| ¿Qué índices se maximizan? | Silueta y CHI favorecen valores altos; DBI favorece valores bajos. La inercia disminuye con complejidad y no basta para elegir K sola. |
| ¿Qué significa silueta negativa? | En promedio, el punto está más cerca de otro grupo que de los demás puntos del suyo según esa medida. No demuestra un error de etiqueta verdadero. |
| ¿PCA decide los grupos? | No. K-Means ajusta en el espacio preparado completo; PCA proyecta después observaciones y centroides sobre los mismos ejes. |
| ¿Los grupos son clases descubiertas? | Son particiones según variables, transformación y distancia. Su utilidad requiere interpretación del dominio. |
| ¿Por qué K-Means ajusta preparación con toda la tabla? | El flujo describe esa tabla sin reserva supervisada. Si se pretende evaluar asignación futura habría que separar datos y conservar el preparador y el modelo. |
| ¿Qué ocurre cuando cambia un widget? | Streamlit reejecuta; el estado conserva resultados solo mientras su contexto y parámetros coinciden. El entrenamiento está detrás del botón. |
| ¿Qué garantiza la semilla? | Controla las operaciones aleatorias donde se propaga. Versiones, datos y entorno también afectan reproducibilidad; la huella y configuración ayudan a identificar una ejecución. |

## 7 Puntos de la pauta que merecen una respuesta precisa

Estos puntos son buenas oportunidades de preguntas porque distinguen el requisito general de la implementación concreta:

- **Automático tiene dos niveles.** La carga sugiere la ruta sin pedir target; el ajuste se activa al pulsar el botón. Si te piden entrenamiento inmediato al cargar, estarían pidiendo cambiar la interacción y controlar costos y reejecuciones.
- **El EDA actual sí exporta covarianza.** `guardar_figuras_eda` incluye Pearson, Spearman y covarianza. La guía anterior describe una versión donde covarianza era solo tabla; comprueba el código actual antes de exponer.
- **Correlación negativa.** El diagnóstico la conserva y la dispersión considera magnitud, pero la lista destacada de la interfaz actual filtra solo positivas mayores que 0.85. El ejercicio 5 amplía ese criterio.
- **Reglas de target.** La versión actual contempla nombres, sufijos, última categórica y códigos enteros repetidos. Una temperatura entera repetida podría sugerir clasificación; el contexto y la selección manual son necesarios.
- **Curvas de aprendizaje.** El reportador actual las genera para clasificación y con hiperparámetros ya seleccionados. Sirven como diagnóstico; no constituyen una evaluación anidada independiente de la selección.
- **Superficie de decisión.** Se dibuja solo con exactamente dos predictores numéricos originales con variación. Para más dimensiones habría que fijar las restantes o definir otra proyección con interpretación clara.
- **Datos arbitrarios y escalabilidad.** La silueta puede requerir trabajo cuadrático en filas. El motor evita conservar la matriz de distancias si supera 32 MiB, pero eso no elimina el costo de cálculo; también densifica los datos de K-Means después del preprocesamiento. Muchas categorías pueden elevar memoria.
- **Una fila o una sola clase no sirven para todo.** Cargar una tabla y poder entrenar son validaciones distintas. Agrupamiento requiere al menos tres filas y dos observaciones distintas; clasificación y CV exigen soporte por clase.
- **CLI e interfaz.** La entrada actual es Streamlit y los flujos también se invocan desde Python. La pauta admite un punto de entrada interactivo; no afirmes que existe un parser CLI de entrenamiento si no lo has implementado.

## 8 Más desafíos sin solución inmediata

Úsalos cuando completes los ejercicios anteriores sin mirar código de apoyo.

| Pedido | Qué debe demostrar tu solución |
|---|---|
| Hacer configurable la semilla en la pantalla | La misma semilla llega a partición, CV, estimadores aplicables y JSON; cambiarla invalida resultados. |
| Cambiar el multiplicador IQR de 1.5 a 3 | Actualizar ambos límites y explicar qué cuenta la tarjeta de atípicos; no eliminar filas silenciosamente. |
| Exportar también las figuras en PDF | Mantener PNG, generar PDF desde la misma figura y conservar rutas coherentes. |
| Filtrar predictores de varianza nula | Aprender el filtro en entrenamiento dentro del pipeline y aplicarlo igual en prueba y cada pliegue. |
| Retirar una columna identificadora elegida por la persona | Excluirla antes del ajuste sin perder el objetivo ni el identificador necesario para vincular resultados. |
| Usar distancias Manhattan en k-NN | Cambiar métrica, conservar preprocesamiento y explicar la geometría L1 frente a L2. |
| Hacer configurable epsilon en DANN | Validar positividad y llevarlo a la rejilla; mostrar qué pasa con datos colineales. |
| Añadir decaimiento configurable a LVQ1 | Comprobar primer y último paso, conservar el signo y reproducir la inicialización. |
| Elegir K por DBI | Minimizar DBI, conservar la solución correspondiente y actualizar motivo y gráficos. |
| Controlar clases insuficientes para CV | Mensaje claro antes de entrenar; no bajar a un pliegue ni eliminar clases sin explicación. |
| Añadir curvas de aprendizaje para regresión | Usar KFold y MAE/RMSE; el helper actual estratifica y supone clasificación, así que no basta invocarlo tal cual. |
| Permitir carga Excel | Añadir lector, dependencia necesaria, opción de interfaz y validaciones comunes; explicar el alcance frente a los formatos exigidos. |

## 9 Comprobaciones rápidas sin abrir la interfaz

Ejecuta este bloque desde la raíz con el entorno activo. Comprueba el comportamiento base y sirve como punto de partida para adaptar los ejercicios.

```bash
python - <<'PY'
import numpy as np
import pandas as pd
from core.loader import seleccionar_ruta
from flujo import ejecutar_cart
from models.supervised.comparacion import comparar
from models.unsupervised.kmeans_engine import agrupar

tabla = pd.read_csv("data/iris_petalos.csv")
ruta = seleccionar_ruta(tabla)
print("Ruta:", ruta)
resultado = ejecutar_cart(tabla, ruta.objetivo, ruta.tarea)
p = resultado.particion
assert p.predictores_entrenamiento.index.equals(p.objetivo_entrenamiento.index)
assert p.predictores_prueba.index.equals(p.objetivo_prueba.index)
assert ruta.objetivo not in p.predictores_entrenamiento
print("Filas:", resultado.entrenamiento, resultado.prueba)
print("Métricas:", resultado.metricas)

comparacion = comparar(tabla, ruta.objetivo, ruta.tarea, ["CART"])
busqueda = comparacion.ajustes["CART"]
print("Criterio:", busqueda.scoring)
print("Rejilla probada:", busqueda.cv_results_["params"])
print("Ganador:", busqueda.best_params_)

sin_etiqueta = pd.read_csv("data/iris_medidas.tsv", sep="\t")
assert seleccionar_ruta(sin_etiqueta).objetivo is None
grupos = agrupar(sin_etiqueta, k_maximo=8)
assert len(grupos.filas) == len(sin_etiqueta)
assert np.isfinite(grupos.filas[["CP1", "CP2", "Silueta"]]).all().all()
print("K:", grupos.k, grupos.motivo)
PY
```

Pruebas existentes según el módulo que cambies:

```bash
python -m unittest discover -s tests -p 'test_ingesta.py' -v
python -m unittest discover -s tests -p 'test_preprocesamiento.py' -v
python -m unittest discover -s tests -p 'test_cart.py' -v
python -m unittest discover -s tests -p 'test_comparacion.py' -v
python -m unittest discover -s tests -p 'test_agrupamiento.py' -v
```

Una prueba puede fijar el comportamiento anterior que acabas de cambiar intencionalmente. Revisa su expectativa y justifica la nueva; no borres pruebas para ocultar un fallo. Un cambio visual se comprueba además en una ejecución nueva: la sesión puede conservar resultados y figuras previas.

## 10 Simulacros y criterio de preparación

Realiza cada simulacro con diez minutos de preguntas: cinco para el cambio y cinco para explicarlo y responder variantes.

| Simulacro | Cambio principal | Preguntas de seguimiento |
|---|---|---|
| A | Añadir F1 macro | ¿Por qué difiere de accuracy? ¿Cambió el criterio de CV? ¿Dónde aparece en el CSV? |
| B | Ampliar rejilla CART | ¿Cuántos ajustes se ejecutan? ¿Dónde se entrena el preprocesador? ¿El gráfico sigue siendo interpretable? |
| C | Elegir K por silueta | ¿Qué soluciones compara? ¿Por qué K=1 no sirve? ¿Mayor silueta demuestra clases verdaderas? |
| D | Reescribir distancia DANN | ¿Cuáles son las dimensiones? ¿Dónde se suman índices? ¿Qué asegura regularización? |
| E | Agregar un objetivo nuevo | ¿Qué ocurre con dos candidatas? ¿Con códigos enteros? ¿Con una etiqueta vacía? |
| F | Parametrizar hojas desde la interfaz | ¿Se invalidó el resultado previo? ¿Llegó al estimador? ¿Quedó en el reporte? |

Evalúa cada intento de 0 a 2 en cinco aspectos: **localización**, **comprensión matemática**, **cambio completo**, **verificación** y **explicación**. Usa 0 si dependiste de la solución, 1 si necesitaste una pista y 2 si lo resolviste y justificaste por tu cuenta. Esta es una escala de práctica, no la rúbrica oficial.

Empieza por ejercicios 1, 2, 4, 7 y la derivación de DANN. Después incorpora 3, 5, 6 y 8. Termina con integración de parámetros, métricas de selección y los desafíos avanzados. Si exponen varias personas, cada una debe recorrer ambas rutas y deducir la actualización LVQ y la forma cuadrática DANN sin apoyo.
