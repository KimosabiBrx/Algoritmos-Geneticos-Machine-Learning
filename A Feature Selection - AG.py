import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

np.random.seed(42)

# 0. DATOS
RUTA_CSV = "base de datos museos_5.csv"
df = pd.read_csv(RUTA_CSV, sep=";", encoding="latin-1")

# Aqui NO incluimos COD_DPTO como caracteristica (es el propio target)
FEATURE_COLS = [
    "ANIO", "COD_MES", "COD_TIPO",
    "ADU_BOLESPPAGANTES", "EST_BOLESPPAGANTES", "NIN_BOLESPPAGANTES",
    "MIL_BOLESPPAGANTES", "ADM_BOLESPPAGANTES",
    "ADU_PAGANTES", "EST_PAGANTES", "NIN_PAGANTES", "TOTAL_PAGANTES",
    "ADU_BOLESPNOPAGANTES", "EST_BOLESPNOPAGANTES", "NIN_BOLESPNOPAGANTES",
    "MIL_BOLESPNOPAGANTES", "ADM_BOLESPNOPAGANTES",
    "ADU_NOPAGANTES", "EST_NOPAGANTES", "NIN_NOPAGANTES", "TOTAL_NOPAGANTES",
    "TOTAL",
]
TARGET_COL = "COD_DPTO"  # 20 departamentos distintos

X = df[FEATURE_COLS].values.astype(float)
y = df[TARGET_COL].values
n_clases = len(np.unique(y))

N_FEATURES = X.shape[1]
print(f"Dataset: {X.shape[0]} registros, {N_FEATURES} caracteristicas candidatas")
print(f"Caracteristicas: {FEATURE_COLS}")
print(f"Target: {TARGET_COL} ({n_clases} departamentos distintos)\n")

# 1. REPRESENTACION DEL CROMOSOMA
# Vector binario de longitud N_FEATURES.
# gen[i] = 1 -> se usa la caracteristica i ; gen[i] = 0 -> no se usa.

def crear_cromosoma():
    cromosoma = np.random.randint(0, 2, size=N_FEATURES)
    if cromosoma.sum() == 0:
        cromosoma[np.random.randint(0, N_FEATURES)] = 1
    return cromosoma

# 2. INICIALIZACION DE LA POBLACION
TAM_POBLACION = 20

def inicializar_poblacion(tam=TAM_POBLACION):
    return [crear_cromosoma() for _ in range(tam)]


# 3. FUNCION DE APTITUD (FITNESS)
# Accuracy promedio (3-fold CV) de un Random Forest pequeno, usando solo
# las caracteristicas activas del cromosoma, con penalizacion leve por
# usar muchas caracteristicas.

def fitness(cromosoma):
    indices = np.where(cromosoma == 1)[0]
    if len(indices) == 0:
        return 0.0

    X_sub = X[:, indices]
    modelo = RandomForestClassifier(
        n_estimators=25, max_depth=10, random_state=42, n_jobs=-1
    )
    scores = cross_val_score(modelo, X_sub, y, cv=3, scoring="accuracy")
    accuracy = scores.mean()

    penalizacion = 0.001 * len(indices)
    return accuracy - penalizacion

# 4. SELECCION (torneo)
def seleccion_torneo(poblacion, fitnesses, k=3):
    seleccionados = []
    for _ in range(len(poblacion)):
        participantes_idx = np.random.choice(len(poblacion), size=k, replace=False)
        mejor_idx = max(participantes_idx, key=lambda i: fitnesses[i])
        seleccionados.append(poblacion[mejor_idx].copy())
    return seleccionados


# 5. CRUZAMIENTO (un punto)
PROB_CRUCE = 0.8

def cruzamiento(padre1, padre2):
    if np.random.rand() > PROB_CRUCE:
        return padre1.copy(), padre2.copy()
    punto = np.random.randint(1, N_FEATURES)
    hijo1 = np.concatenate([padre1[:punto], padre2[punto:]])
    hijo2 = np.concatenate([padre2[:punto], padre1[punto:]])
    return hijo1, hijo2


# 6. MUTACION (bit-flip)
PROB_MUTACION = 0.03

def mutacion(cromosoma):
    cromosoma = cromosoma.copy()
    for i in range(len(cromosoma)):
        if np.random.rand() < PROB_MUTACION:
            cromosoma[i] = 1 - cromosoma[i]
    if cromosoma.sum() == 0:
        cromosoma[np.random.randint(0, N_FEATURES)] = 1
    return cromosoma

# 7. CRITERIO DE TERMINACION
NUM_GENERACIONES = 10

# CICLO PRINCIPAL DEL ALGORITMO GENETICO
def ejecutar_ag():
    poblacion = inicializar_poblacion()
    mejor_global = None
    mejor_fitness_global = -np.inf

    for gen in range(NUM_GENERACIONES):
        fitnesses = [fitness(ind) for ind in poblacion]

        idx_mejor = int(np.argmax(fitnesses))
        if fitnesses[idx_mejor] > mejor_fitness_global:
            mejor_fitness_global = fitnesses[idx_mejor]
            mejor_global = poblacion[idx_mejor].copy()

        print(f"Generacion {gen+1:2d} | Mejor fitness: {fitnesses[idx_mejor]:.4f} "
              f"| Promedio: {np.mean(fitnesses):.4f} "
              f"| N features (mejor): {poblacion[idx_mejor].sum()}")

        padres = seleccion_torneo(poblacion, fitnesses)

        nueva_poblacion = []
        for i in range(0, len(padres) - 1, 2):
            hijo1, hijo2 = cruzamiento(padres[i], padres[i + 1])
            nueva_poblacion.extend([hijo1, hijo2])
        if len(padres) % 2 == 1:
            nueva_poblacion.append(padres[-1])

        nueva_poblacion = [mutacion(ind) for ind in nueva_poblacion]
        nueva_poblacion[0] = mejor_global.copy()  # elitismo

        poblacion = nueva_poblacion

    return mejor_global, mejor_fitness_global


if __name__ == "__main__":
    print("=" * 70)
    print("INICIANDO ALGORITMO GENETICO - FEATURE SELECTION (Departamento)")
    print("=" * 70)

    mejor_cromosoma, mejor_fitness = ejecutar_ag()

    print("\n" + "=" * 70)
    print("RESULTADO FINAL")
    print("=" * 70)
    indices_elegidos = np.where(mejor_cromosoma == 1)[0]
    print(f"Mejor fitness (accuracy - penalizacion): {mejor_fitness:.4f}")
    print(f"Caracteristicas seleccionadas: {len(indices_elegidos)} de {N_FEATURES}")
    for i in indices_elegidos:
        print(f"  - {FEATURE_COLS[i]}")

    modelo_todas = RandomForestClassifier(n_estimators=25, max_depth=10, random_state=42, n_jobs=-1)
    acc_todas = cross_val_score(modelo_todas, X, y, cv=3, scoring="accuracy").mean()
    print(f"\nAccuracy usando TODAS las caracteristicas ({N_FEATURES}): {acc_todas:.4f}")

    modelo_sub = RandomForestClassifier(n_estimators=25, max_depth=10, random_state=42, n_jobs=-1)
    acc_sub = cross_val_score(modelo_sub, X[:, indices_elegidos], y, cv=3, scoring="accuracy").mean()
    print(f"Accuracy usando SOLO las seleccionadas ({len(indices_elegidos)}): {acc_sub:.4f}")