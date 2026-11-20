# Plataforma de aprendizaje automático

Aplicación local en Python y Streamlit para cargar tablas, revisar una muestra y seleccionar la ruta de aprendizaje.

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

1. Abrir la dirección local que indica Streamlit y elegir **Ejemplo local**. En `flores.csv` se muestran 36 filas, tres columnas y la ruta de clasificación por la columna `clase`.
2. Cambiar a `mediciones.tsv`. Las dos variables son mediciones sin etiqueta. La aplicación explica la ausencia de objetivo y selecciona agrupamiento con K-Means.
3. Elegir **Subir archivo** para probar CSV, TSV o Parquet. Para texto se admite UTF-8 o Latin-1. En **Opciones de lectura** se puede corregir el delimitador y desmarcar el encabezado para asignar nombres como `columna_1`.
4. Usar **Elegir columna** para indicar el objetivo. Si un código numérico representa categorías, seleccionar **Clasificación**. **Sin objetivo** permite elegir agrupamiento explícitamente.

Los dos ejemplos son datos sintéticos creados para comprobar estos recorridos. El primero simula medidas de flores en centímetros y tres clases ficticias. El segundo simula temperatura y humedad en tres condiciones ambientales.

## Organización

| Ubicación | Responsabilidad prevista |
|---|---|
| main.py | Interfaz en español |
| core/ | Carga de tablas, análisis exploratorio y preprocesamiento |
| models/ | Contrato común y familias supervisadas y no supervisadas |
| evaluation/ | Métricas y figuras |
| data/ y reports/ | Ejemplos locales y salidas por ejecución |
| tests/ | Pruebas de carga y selección de tarea |

Con el entorno activado, ejecutar las pruebas con `python -m unittest discover -s tests -v`.
