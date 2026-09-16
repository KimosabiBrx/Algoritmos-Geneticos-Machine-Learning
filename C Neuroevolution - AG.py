import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import random
import warnings
from sklearn.exceptions import ConvergenceWarning

# Ignorar advertencias de convergencia para mantener la consola limpia en la presentación
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# ==========================================
# FASE 0: PREPROCESAMIENTO DE DATOS
# ==========================================
print("[CONTROL] 1. Iniciando carga de datos...")
# 1. Cargar el dataset
df = pd.read_csv("base de datos museos_5.csv", sep=";", encoding="latin-1")
print(f"[CONTROL] Dataset cargado exitosamente. Tamaño original: {df.shape}")

print("[CONTROL] 2. Limpiando redundancias y evitando fuga de datos...")
# 2. Limpieza de redundancias y fuga de datos
columnas_a_eliminar = ['FECHA_CORTE', 'NOM_DPTO', 'NOM_MES', 'NOM_TIPO']
columnas_subtotales = ['ADU_BOLESPPAGANTES', 'EST_BOLESPPAGANTES', 'NIN_BOLESPPAGANTES', 
                       'TOTAL_PAGANTES', 'ADU_NOPAGANTES', 'TOTAL_NOPAGANTES'] 

# Usamos errors='ignore' para prevenir caídas si alguna columna ya no existe
df = df.drop(columns=columnas_a_eliminar + columnas_subtotales, errors='ignore')

print("[CONTROL] 3. Aplicando One-Hot Encoding...")
# 3. Codificación One-Hot
df = pd.get_dummies(df, columns=['COD_DPTO', 'COD_MES', 'COD_TIPO', 'NOM_MUSEO', 'TIPO_cat_cod'], drop_first=True)
print(f"[CONTROL] Dimensiones tras One-Hot: {df.shape}")

print("[CONTROL] 4. Separando y escalando datos...")
# 4. Separar Features (X) y Target (y)
X = df.drop(columns=['TOTAL'])
y = df['TOTAL']

# 5. División Train/Test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Escalado 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("[CONTROL] Preprocesamiento finalizado con éxito.\n")


# FASES DEL ALGORITMO GENÉTICO (Neuroevolución)

# Fase 1: Representación
opciones_capas = [(16,), (32,), (16, 8), (32, 16)] 
opciones_activacion = ['relu', 'tanh', 'logistic']

# Fase 2: Inicialización
def inicializar_poblacion(tamano):
    poblacion = []
    for _ in range(tamano):
        poblacion.append([random.randint(0, len(opciones_capas) - 1), 
                          random.randint(0, len(opciones_activacion) - 1)])
    return poblacion

# Fase 3: Función de Aptitud
def calcular_fitness(cromosoma):
    capas = opciones_capas[cromosoma[0]]
    activacion = opciones_activacion[cromosoma[1]]
    modelo = MLPRegressor(hidden_layer_sizes=capas, activation=activacion, max_iter=50, random_state=42)
    try:
        modelo.fit(X_train_scaled, y_train)
        predicciones = modelo.predict(X_test_scaled)
        mse = mean_squared_error(y_test, predicciones)
        return 1.0 / (mse + 1e-6)
    except:
        return 0 

# Fase 4: Selección (Torneo)
def seleccion_torneo(poblacion, fitness_pob, k=3):
    seleccionados = []
    for _ in range(len(poblacion)):
        participantes = random.sample(list(zip(poblacion, fitness_pob)), k)
        ganador = max(participantes, key=lambda x: x[1])[0]
        seleccionados.append(ganador)
    return seleccionados

# Fase 5: Cruzamiento
def cruzamiento(padre1, padre2):
    return [padre1[0], padre2[1]], [padre2[0], padre1[1]]

# Fase 6: Mutación
def mutacion(cromosoma, tasa_mutacion=0.1):
    if random.random() < tasa_mutacion:
        cromosoma[0] = random.randint(0, len(opciones_capas) - 1)
    if random.random() < tasa_mutacion:
        cromosoma[1] = random.randint(0, len(opciones_activacion) - 1)
    return cromosoma


# FASE 7: TERMINACIÓN (Ejecución Principal)
TAMANO_POBLACION = 6  # Reducido para mayor velocidad en la exposición
GENERACIONES = 4

print("==========================================")
print("[AG] INICIANDO EVOLUCIÓN DE RED NEURONAL")
print("==========================================")
poblacion = inicializar_poblacion(TAMANO_POBLACION)
mejor_cromosoma_global = None
mejor_fitness_global = 0

for generacion in range(GENERACIONES):
    print(f"\n[AG] ---> Generación {generacion + 1} de {GENERACIONES} <---")
    
    fitness_pob = []
    for i, ind in enumerate(poblacion):
        print(f"      Evaluando red {i+1}/{TAMANO_POBLACION} (Capas: {opciones_capas[ind[0]]}, Act: {opciones_activacion[ind[1]]})...", end="")
        fit = calcular_fitness(ind)
        fitness_pob.append(fit)
        print(" ¡Listo!")
    
    mejor_idx = np.argmax(fitness_pob)
    if fitness_pob[mejor_idx] > mejor_fitness_global:
        mejor_fitness_global = fitness_pob[mejor_idx]
        mejor_cromosoma_global = poblacion[mejor_idx]
        print(f"      [!] NUEVO MEJOR FITNESS: {mejor_fitness_global:.8f}")
    
    # Procesos evolutivos
    print("      Aplicando Selección, Cruzamiento y Mutación...")
    padres = seleccion_torneo(poblacion, fitness_pob)
    
    nueva_poblacion = []
    for i in range(0, len(padres), 2):
        p1 = padres[i]
        p2 = padres[i+1] if i+1 < len(padres) else padres[0] 
        h1, h2 = cruzamiento(p1, p2)
        nueva_poblacion.extend([mutacion(h1), mutacion(h2)])
        
    poblacion = nueva_poblacion[:TAMANO_POBLACION]

print("\n==========================================")
print("[AG] BÚSQUEDA FINALIZADA")
print("==========================================")
arquitectura_final = opciones_capas[mejor_cromosoma_global[0]]
activacion_final = opciones_activacion[mejor_cromosoma_global[1]]
print(f"La MEJOR arquitectura encontrada para predecir visitantes es:")
print(f"- Capas Ocultas y Neuronas: {arquitectura_final}")
print(f"- Función de Activación: {activacion_final}")