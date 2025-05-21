# DEPENDENCIAS
import sys
import time
import numpy as np
from scipy.stats import norm

# FUNCIONES AUXILIARES

# Calculo de la integral
def funcion_ej62(x):
    
    # Coordenadas del punto
    x1, x2, x3, x4, x5 = x

    # Calculo de la funcion
    return x1 * (x2 ** 2) * (x3 ** 3) * (x4 ** 4) * (x5 ** 5)

# Calculo del intervalo de confianza con método normal
def intervalo_normal(estimador, desviacion_estimador, d):

    # Calculos previos para simplificar formula
    z = norm.ppf(1 - d/2)

    # Calculo de cota inferior y superior
    w1 = estimador - z * desviacion_estimador
    w2 = estimador + z * desviacion_estimador
    return w1, w2

# FUNCIONES PRINCIPALES

# Version estratificada del método
def montecarlo_estratificado(n, f, d, semilla=42, proporcional=False):

    # Inicializar contadores
    tic = time.time()

    # Iniciar semilla
    np.random.seed(semilla)

    # Definir estratos
    estratos = [(0.00, 0.72), (0.72, 0.83), (0.83, 0.90), (0.90, 0.95), (0.95, 1.00)]
    largo_estratos = len(estratos) # 5 en este caso, se deduce de estratos por si la variable cambia

    # Calcular cantidad de muestras de cada estrato de manera proporcional
    if proporcional:
        
        # Definir pesos en intervalo para cada estrato
        pesos = [b - a for (a, b) in estratos]
        
        # Generar lista de tamaños de muestra para cada estrato
        n_estrato = [int(n * peso) for peso in pesos]

    
    # Calcular cantidad de muestras de cada estrato de manera uniforme
    else:
        
        # Dividir cantidad de muestras total por cantidad de estratos (10^6 / 5)
        n_base = n // largo_estratos
        
        # Generar lista siguiendo estructura de la lista anterior para reutilizar
        n_estrato = [n_base for _ in range(largo_estratos)]

    # Inicializar acumuladores globales
    estimadores = []
    varianzas = []

    # Iterar en cada estrato
    for i in range(largo_estratos):
        
        # Inicializar valores auxiliares
        acumulador = 0.0
        acumulador_previo = 0.0
        varianza_i = 0.0

        # Inicializar valores del estrato
        a, b = estratos[i] # Comienzo y fin del estrato
        ni = n_estrato[i] # Cantidad de muestras del estrato
        pi = b - a # peso del estrato

        # Iterar segun cantidad de muestras del estrato
        for j in range(1, ni + 1):
            
            # Sortear x1, x2, x3 y x4 en [0,1], y x5 en [a,b]
            x1 = np.random.uniform(0, 1)
            x2 = np.random.uniform(0, 1)
            x3 = np.random.uniform(0, 1)
            x4 = np.random.uniform(0, 1)
            x5 = np.random.uniform(a, b)

            # Calcular valor de la funcion y acumular
            valor = f((x1, x2, x3, x4, x5))
            acumulador += valor

            # Calcular valor de la varianza y acumular
            if j > 1:
                varianza_i += (1 - 1/j) * (valor - (acumulador_previo / (j - 1))) ** 2

            acumulador_previo = acumulador

        # Calcular media y varianza, agregarlas a las listas de cada estrato
        media_i = acumulador / ni
        varianza_i = varianza_i / (ni - 1)
        estimadores.append(pi * media_i)
        varianzas.append((pi ** 2) * varianza_i / ni)

    # Estimador total
    estimador = sum(estimadores)
    varianza_total = sum(varianzas)
    desviacion = np.sqrt(varianza_total)
    w1, w2 = intervalo_normal(estimador, desviacion, d)

    # Apagar contador y calcular tiempo de ejecución
    toc = time.time()
    tiempo_de_ejecucion = toc - tic

    # Retornar estimadores para flujo principal
    return {
        "estimador": estimador,
        "varianza": varianza_total,
        "desviacion": desviacion,
        "intervalo": (w1, w2),
        "tiempo": tiempo_de_ejecucion
    }

# MÉTODO PRINCIPAL

# Leer parametros
n = 1000000 if len(sys.argv) <= 1 else float(sys.argv[1]) # Cantidad de iteraciones (n)
d = 0.05 if len(sys.argv) <= 2 else float(sys.argv[2]) # Nivel de confianza (1-d)

# Ejecutar ambas versiones del algoritmo
res_uniforme = montecarlo_estratificado(n, funcion_ej62, d, 42, proporcional=False)
res_proporcional = montecarlo_estratificado(n, funcion_ej62, d, 42, proporcional=True)

# Mostrar resultados
print()
print("--- Estratificación con asignación UNIFORME ---")
print(f"Estimador: {res_uniforme['estimador']}")
print(f"Varianza: {res_uniforme['varianza']}")
print(f"Desviación: {res_uniforme['desviacion']}")
print(f"Intervalo de confianza: {float(res_uniforme['intervalo'][0])}, {float(res_uniforme['intervalo'][1])}")
print(f"Tiempo: {round(res_uniforme['tiempo'], 3)} segundos")
print()
print("--- Estratificación con asignación PROPORCIONAL ---")
print(f"Estimador: {res_proporcional['estimador']}")
print(f"Varianza: {res_proporcional['varianza']}")
print(f"Desviación: {res_proporcional['desviacion']}")
print(f"Intervalo de confianza: {float(res_uniforme['intervalo'][0])}, {float(res_uniforme['intervalo'][1])}")
print(f"Tiempo: {round(res_proporcional['tiempo'], 3)} segundos")
