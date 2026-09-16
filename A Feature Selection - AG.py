import random
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

random.seed(11)
np.random.seed(11)

ARCHIVO_DATOS = "base de datos museos_5.csv"

FEATURE_COLS = [
    "ANIO", "COD_DPTO", "COD_MES",
    "ADU_BOLESPPAGANTES", "EST_BOLESPPAGANTES", "NIN_BOLESPPAGANTES",
    "MIL_BOLESPPAGANTES", "ADM_BOLESPPAGANTES",
    "ADU_PAGANTES", "EST_PAGANTES", "NIN_PAGANTES", "TOTAL_PAGANTES",
    "ADU_BOLESPNOPAGANTES", "EST_BOLESPNOPAGANTES", "NIN_BOLESPNOPAGANTES",
    "MIL_BOLESPNOPAGANTES", "ADM_BOLESPNOPAGANTES",
    "ADU_NOPAGANTES", "EST_NOPAGANTES", "NIN_NOPAGANTES", "TOTAL_NOPAGANTES",
    "TOTAL",
]
TARGET_COL = "COD_TIPO"  # 0 = Nacionales, 1 = Extranjeros


def cargar_datos():
    tabla = pd.read_csv(ARCHIVO_DATOS, sep=";", encoding="latin-1")
    X = tabla[FEATURE_COLS].to_numpy(dtype=float)
    y = tabla[TARGET_COL].to_numpy()
    return X, y


X_DATOS, Y_DATOS = cargar_datos()

# Separamos un conjunto de PRUEBA final (20%) que el AG jamas vera
# durante la busqueda. Solo se usa al final, para confirmar el resultado.
X_BUSQUEDA, X_PRUEBA, Y_BUSQUEDA, Y_PRUEBA = train_test_split(
    X_DATOS, Y_DATOS, test_size=0.2, stratify=Y_DATOS, random_state=11
)

PARTICION_CV = StratifiedKFold(n_splits=3, shuffle=True, random_state=11)

print(f"Registros totales: {X_DATOS.shape[0]}")
print(f"  -> usados para la busqueda del AG (con CV interno): {X_BUSQUEDA.shape[0]}")
print(f"  -> reservados como prueba final (holdout): {X_PRUEBA.shape[0]}\n")

# 1. REPRESENTACION DEL CROMOSOMA
# Ejemplo de cromosoma: [120, 14, 4, 2, 0]  ->
#   120 arboles, profundidad maxima 14, min_samples_split=4,
#   min_samples_leaf=2, max_features='sqrt' (indice 0)
RANGO_N_ESTIMATORS = (10, 60)       # numero de arboles
RANGO_MAX_DEPTH = (2, 15)           # profundidad maxima
RANGO_MIN_SAMPLES_SPLIT = (2, 20)
RANGO_MIN_SAMPLES_LEAF = (1, 10)
OPCIONES_MAX_FEATURES = ["sqrt", "log2"]  # se referencian por indice 0,1


def cromosoma_aleatorio():
    return [
        random.randint(*RANGO_N_ESTIMATORS),
        random.randint(*RANGO_MAX_DEPTH),
        random.randint(*RANGO_MIN_SAMPLES_SPLIT),
        random.randint(*RANGO_MIN_SAMPLES_LEAF),
        random.randint(0, len(OPCIONES_MAX_FEATURES) - 1),
    ]


def cromosoma_a_hiperparametros(cromosoma):
    n_estimators, max_depth, min_split, min_leaf, idx_max_features = cromosoma
    return {
        "n_estimators": int(n_estimators),
        "max_depth": int(max_depth),
        "min_samples_split": int(min_split),
        "min_samples_leaf": int(min_leaf),
        "max_features": OPCIONES_MAX_FEATURES[int(idx_max_features)],
    }

# 2. INICIALIZACION DE LA POBLACION
TAMANO_POBLACION = 10

def poblacion_inicial():
    return [cromosoma_aleatorio() for _ in range(TAMANO_POBLACION)]


# 3. FUNCION DE APTITUD (FITNESS)

def calcular_fitness(cromosoma):
    hiperparametros = cromosoma_a_hiperparametros(cromosoma)
    modelo = RandomForestClassifier(
        **hiperparametros, random_state=11, n_jobs=1
    )
    exactitudes = cross_val_score(
        modelo, X_BUSQUEDA, Y_BUSQUEDA, cv=PARTICION_CV,
        scoring="accuracy", n_jobs=-1
    )
    return exactitudes.mean()


# 4. SELECCION -> TORNEO
def seleccion_torneo(poblacion, lista_fitness, k=3):
    seleccionados = []
    for _ in range(len(poblacion)):
        participantes = random.sample(range(len(poblacion)), k)
        ganador = max(participantes, key=lambda i: lista_fitness[i])
        seleccionados.append(poblacion[ganador][:])
    return seleccionados


# 5. CRUZAMIENTO -> ARITMETICO / UNIFORME SEGUN EL GEN

PROBABILIDAD_CRUCE = 0.85

RANGOS_GENES_NUMERICOS = [
    RANGO_N_ESTIMATORS, RANGO_MAX_DEPTH,
    RANGO_MIN_SAMPLES_SPLIT, RANGO_MIN_SAMPLES_LEAF,
]

def cruzamiento(padre1, padre2):
    if random.random() > PROBABILIDAD_CRUCE:
        return padre1[:], padre2[:]

    hijo1, hijo2 = [], []

    # Genes numericos (los primeros 4): cruce aritmetico
    for i in range(4):
        minimo, maximo = RANGOS_GENES_NUMERICOS[i]
        promedio = (padre1[i] + padre2[i]) / 2
        # Pequena variacion aleatoria alrededor del promedio para no
        # perder diversidad (si no, todos los hijos serian identicos)
        ruido = random.uniform(-0.15, 0.15) * (maximo - minimo)
        valor1 = int(np.clip(round(promedio + ruido), minimo, maximo))
        valor2 = int(np.clip(round(promedio - ruido), minimo, maximo))
        hijo1.append(valor1)
        hijo2.append(valor2)

    # Gen categorico (max_features): herencia uniforme
    if random.random() < 0.5:
        hijo1.append(padre1[4])
        hijo2.append(padre2[4])
    else:
        hijo1.append(padre2[4])
        hijo2.append(padre1[4])

    return hijo1, hijo2


# 6. MUTACION -> REINICIO ALEATORIO DENTRO DEL RANGO
PROBABILIDAD_MUTACION = 0.20  # por gen

def mutar(cromosoma):
    nuevo = cromosoma[:]

    if random.random() < PROBABILIDAD_MUTACION:
        nuevo[0] = random.randint(*RANGO_N_ESTIMATORS)
    if random.random() < PROBABILIDAD_MUTACION:
        nuevo[1] = random.randint(*RANGO_MAX_DEPTH)
    if random.random() < PROBABILIDAD_MUTACION:
        nuevo[2] = random.randint(*RANGO_MIN_SAMPLES_SPLIT)
    if random.random() < PROBABILIDAD_MUTACION:
        nuevo[3] = random.randint(*RANGO_MIN_SAMPLES_LEAF)
    if random.random() < PROBABILIDAD_MUTACION:
        nuevo[4] = random.randint(0, len(OPCIONES_MAX_FEATURES) - 1)

    return nuevo


# 7. CRITERIO DE TERMINACION
GENERACIONES = 10


# CICLO PRINCIPAL DEL ALGORITMO GENETICO
def correr_algoritmo_genetico():
    poblacion = poblacion_inicial()
    mejor_cromosoma_hist = None
    mejor_fitness_hist = -np.inf

    for gen in range(1, GENERACIONES + 1):
        fitness_poblacion = [calcular_fitness(c) for c in poblacion]

        idx_mejor = int(np.argmax(fitness_poblacion))
        if fitness_poblacion[idx_mejor] > mejor_fitness_hist:
            mejor_fitness_hist = fitness_poblacion[idx_mejor]
            mejor_cromosoma_hist = poblacion[idx_mejor][:]

        hp_mejor = cromosoma_a_hiperparametros(poblacion[idx_mejor])
        print(f"Gen {gen:2d} | mejor fitness: {fitness_poblacion[idx_mejor]:.4f} "
              f"| promedio: {np.mean(fitness_poblacion):.4f} "
              f"| hiperparametros: {hp_mejor}")

        # Seleccion
        padres = seleccion_torneo(poblacion, fitness_poblacion)

        # Cruzamiento
        siguiente_generacion = []
        for i in range(0, len(padres) - 1, 2):
            hijo1, hijo2 = cruzamiento(padres[i], padres[i + 1])
            siguiente_generacion += [hijo1, hijo2]
        if len(padres) % 2 == 1:
            siguiente_generacion.append(padres[-1][:])

        # Mutacion
        siguiente_generacion = [mutar(c) for c in siguiente_generacion]

        # Elitismo: el mejor cromosoma encontrado siempre sobrevive
        siguiente_generacion[0] = mejor_cromosoma_hist[:]

        poblacion = siguiente_generacion

    return mejor_cromosoma_hist, mejor_fitness_hist


# EJECUCION Y REPORTE DE RESULTADOS
if __name__ == "__main__":
    print("=" * 72)
    print(" ALGORITMO GENETICO - HYPERPARAMETER OPTIMIZATION")
    print(" MODELO: RANDOM FOREST | TAREA: NACIONAL vs EXTRANJERO")
    print("=" * 72 + "\n")

    mejor_cromosoma, mejor_fitness = correr_algoritmo_genetico()
    mejores_hiperparametros = cromosoma_a_hiperparametros(mejor_cromosoma)

    print("\n" + "=" * 72)
    print(" RESULTADO")
    print("=" * 72)
    print(f"Mejor fitness (accuracy CV en busqueda): {mejor_fitness:.4f}")
    print("Mejores hiperparametros encontrados:")
    for nombre, valor in mejores_hiperparametros.items():
        print(f"   - {nombre}: {valor}")

    # -- Comparacion contra hiperparametros por defecto de sklearn --
    modelo_defecto = RandomForestClassifier(random_state=11, n_jobs=-1)
    modelo_defecto.fit(X_BUSQUEDA, Y_BUSQUEDA)
    acc_defecto = modelo_defecto.score(X_PRUEBA, Y_PRUEBA)

    # -- Evaluacion final del mejor modelo encontrado por el AG --
    # (entrenado con TODO el conjunto de busqueda, probado en el holdout)
    modelo_ag = RandomForestClassifier(
        **mejores_hiperparametros, random_state=11, n_jobs=-1
    )
    modelo_ag.fit(X_BUSQUEDA, Y_BUSQUEDA)
    acc_ag = modelo_ag.score(X_PRUEBA, Y_PRUEBA)

    print("\n" + "=" * 72)
    print(" COMPARACION FINAL (conjunto de prueba, nunca visto por el AG)")
    print("=" * 72)
    print(f"Accuracy con hiperparametros POR DEFECTO de sklearn: {acc_defecto:.4f}")
    print(f"Accuracy con hiperparametros ENCONTRADOS POR EL AG:  {acc_ag:.4f}")