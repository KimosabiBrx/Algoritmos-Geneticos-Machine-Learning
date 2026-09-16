# Algoritmos Genéticos aplicados a Machine Learning — Dataset de Museos

Este repositorio contiene tres implementaciones de **Algoritmos Genéticos (AG)** aplicados a distintos problemas de Machine Learning sobre un dataset de visitantes a museos en Perú. Cada script aborda una etapa distinta del pipeline de modelado: selección de características, optimización de hiperparámetros y neuroevolución de arquitecturas.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `A Feature Selection - AG.py` | Selección de características mediante AG para predecir el departamento (`COD_DPTO`) usando un Random Forest. |
| `B Hyperparameter optimization - AG.py` | Optimización de hiperparámetros de un Random Forest mediante AG para clasificar visitantes nacionales vs. extranjeros (`COD_TIPO`). |
| `C Neuroevolution - AG.py` | Neuroevolución: búsqueda de arquitectura y función de activación de una red neuronal (`MLPRegressor`) para predecir el total de visitantes (`TOTAL`). |
| `*.ipynb` | Versiones equivalentes de cada script, listas para ejecutarse en Google Colab. |

## Dataset

Los tres scripts utilizan el archivo `base de datos museos_5.csv` (separador `;`, codificación `latin-1`)  y debe colocarse en la misma carpeta que los scripts antes de ejecutarlos.

## Requisitos

```bash
pip install numpy pandas scikit-learn
```

## Descripción de cada algoritmo genético

### A. Feature Selection (`A Feature Selection - AG.py`)
- **Cromosoma:** vector binario donde cada gen indica si una característica se usa (1) o no (0).
- **Fitness:** accuracy promedio (validación cruzada 3-fold) de un Random Forest entrenado solo con las características activas, con una leve penalización por cantidad de características.
- **Selección:** torneo (k=3). **Cruce:** un punto (prob. 0.8). **Mutación:** bit-flip (prob. 0.03 por gen).
- **Objetivo:** predecir el departamento (`COD_DPTO`) reduciendo el número de características sin perder precisión.

### B. Hyperparameter Optimization (`B Hyperparameter optimization - AG.py`)
- **Cromosoma:** vector de 5 genes (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `max_features`) que codifica los hiperparámetros de un Random Forest.
- **Fitness:** accuracy promedio en validación cruzada estratificada (3-fold) sobre un conjunto de búsqueda, reservando un 20% de los datos como holdout final.
- **Selección:** torneo (k=3). **Cruce:** aritmético con ruido para genes numéricos + herencia uniforme para el gen categórico (prob. 0.85). **Mutación:** reinicio aleatorio por gen (prob. 0.20).
- **Objetivo:** clasificar visitantes nacionales vs. extranjeros (`COD_TIPO`), comparando el resultado final contra los hiperparámetros por defecto de scikit-learn.

### C. Neuroevolution (`C Neuroevolution - AG.py`)
- **Cromosoma:** par de genes que codifican la arquitectura de capas ocultas y la función de activación de una red neuronal (`MLPRegressor`).
- **Fitness:** inverso del error cuadrático medio (MSE) sobre un conjunto de prueba.
- **Selección:** torneo (k=3). **Cruce:** intercambio de genes entre padres. **Mutación:** reinicio aleatorio por gen (prob. 0.10).
- **Objetivo:** encontrar la arquitectura óptima para predecir el total de visitantes (`TOTAL`), tras eliminar columnas redundantes y aplicar One-Hot Encoding.

## Ejecución

```bash
python "A Feature Selection - AG.py"
python "B Hyperparameter optimization - AG.py"
python "C Neuroevolution - AG.py"
```

Cada script imprime en consola el progreso generación por generación (mejor fitness y promedio de la población) y finaliza con un resumen del mejor resultado encontrado, comparándolo contra una línea base (todas las características o hiperparámetros por defecto, según el caso).

## Estructura común de los AG

Todos los algoritmos siguen el mismo esquema clásico:

1. Representación del cromosoma
2. Inicialización de la población
3. Función de aptitud (fitness)
4. Selección por torneo
5. Cruzamiento
6. Mutación
7. Criterio de terminación (número fijo de generaciones) + elitismo
