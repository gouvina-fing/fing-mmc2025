# DEPENDENCIAS
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FUNCIONES PRINCIPALES

# Version iterativa del metodo, generando valores aleatorios y acumulando en cada iteracion
def montecarlo_iterativo(n: int, modo: str):

  # Inicializar contador
  tic = time.time()

  # Inicializar valores de la hiperspefera
  c1 = 0.45
  c2 = 0.5
  c3 = 0.6
  c4 = 0.6
  c5 = 0.5
  c6 = 0.45
  r = 0.35

  # Inicializar estimadores
  X = 0
  V = 0

  # Iterar segun el tamaño de muestra elegido
  for _ in range(0, n):
      
      # Sortear coordenadas del punto a revisar
      x1 = np.random.uniform(0, 1)
      x2 = np.random.uniform(0, 1)
      x3 = np.random.uniform(0, 1)
      x4 = np.random.uniform(0, 1)
      x5 = np.random.uniform(0, 1)
      x6 = np.random.uniform(0, 1)
      
      # Comprobar si el punto se encuentra dentro de la hiperesfera
      # Si no es asi, saltar a la siguiente iteración
      if not (((x1-c1) ** 2) + ((x2-c2) ** 2) + ((x3-c3) ** 2) + ((x4-c4) ** 2) + ((x5-c5) ** 2) + ((x6-c6) ** 2) <= r ** 2):
        continue

      # Comprobar si el punto cumple las restricciones, solo en caso de que el parámetro modo indique necesario esto
      # Si no es así, saltar a la siguiente iteración
      if modo == "con_restricciones":
        if ((3*x1 + 7*x4) > 5) or ((x3 + x4) > 1) or ((x1 - x2 - x5 + x6) < 0):
          continue

      # Acumular valor final en el caso de cumplir todas las restricciones
      X += 1

  # Calcular estimadores luego de acumulados los valores de la muestra
  X = X / n
  V = math.sqrt((X * (1-X)) / (n-1))

  # Apagar contador y calcular tiempo de ejecución
  toc = time.time()
  tiempo_de_ejecucion = toc - tic

  # Retornar estimadores para flujo principal
  return X, V, tiempo_de_ejecucion

# MÉTODO PRINCIPAL

# Iniciar semilla
np.random.seed(42)

# Leer parametros
n = int(sys.argv[1]) # Cantidad de iteraciones
modo = "con_restricciones" if len(sys.argv) <= 2 else sys.argv[2] # Modo con restricciones o sin restricciones (parte b)
tipo_de_ejecucion = "simple" if len(sys.argv) <= 3 else sys.argv[3] # Tipo de ejecucion simple o busqueda por tiempos

# Calcular volumen exacto para parte b
r = 0.35
volumen_exacto = ((math.pi ** 3) * (r ** 6)) / 6

# Para ejecucion simple, se ejecuta el método para la cantidad de iteraciones elegida
if tipo_de_ejecucion == "simple":
   
  # Ejecutar método de monte carlo dependiendo del modo elegido
  X, V, tiempo_de_ejecucion = montecarlo_iterativo(n, modo)

  # Mostrar resultados
  print("Tiempo:", round(tiempo_de_ejecucion, 3), "segundos")
  print("Resultados:")
  print("Volumen exacto:", volumen_exacto)
  print("Estimador:", X)
  print("Desviacion estandar:", V)

# Para ejecucion de busqueda, se ejecuta el métod para distintos valores de n, generando una tabla comparativa
else: 

  # Inicializar parametros
  n = 10 # Cantidad de iteraciones, aumenta en potencia de 10
  tiempo_alcanzado = False # Bandera para parar ejecucion una vez se alcancen 60 segundos
  resultados = [] # Tabla de resultados
  contador = 1
  grafica_eje_x = []
  grafica_eje_y_X = []
  grafica_eje_y_V = []

  # Iterar hasta alcanzar el tiempo limite de 60 segundos
  while not tiempo_alcanzado and n <= 10000000:

    # Ejecutar método de monte carlo para cantidad de iteraciones correspondiente
    X, V, tiempo_de_ejecucion = montecarlo_iterativo(n, modo)

    # Generar matriz de resultados
    resultados_intermedios = [n, X, V, tiempo_de_ejecucion]
    resultados.append(resultados_intermedios)

    # Comprobar si el tiempo alcanzado
    tiempo_alcanzado = tiempo_de_ejecucion > 60

    # Agregar cantidad de iteraciones a lista para grafica
    grafica_eje_x.append(f"10^{contador}")
    grafica_eje_y_X.append(X)
    grafica_eje_y_V.append(V)

    # Aumentar cantidad de iteraciones como potencia de 10
    n *= 10
    contador += 1

  # Mostrar resultados en una tabla
  df = pd.DataFrame(resultados, columns=["Iteraciones", "X", "V", "Tiempo"], index=range(1, len(resultados) + 1))
  print()
  print(df)

  # Generar gráfica de X para estudiar convergencia
  plt.figure(figsize=(8, 5))
  plt.plot(grafica_eje_x, grafica_eje_y_X, label='X', color='red', linewidth=2)
  if modo != "con_restricciones":
    plt.axhline(y=volumen_exacto, color='green', linestyle='--', label='Volumen exacto')
  plt.title("Comportamiento de estimador X")
  plt.xlabel("Cantidad de iteraciones")
  plt.ylabel("Valor de X")
  plt.legend()
  plt.show()

  # Generar gráfica de V para estudiar convergencia
  plt.figure(figsize=(8, 5))
  plt.plot(grafica_eje_x, grafica_eje_y_V, label='V', color='blue', linewidth=2)
  plt.title("Comportamiento de estimador V")
  plt.xlabel("Cantidad de iteraciones")
  plt.ylabel("Valor de V")
  plt.legend()
  plt.show()