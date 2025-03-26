# DEPENDENCIAS
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# FUNCIONES PRINCIPALES

# Version iterativa del metodo, generando valores aleatorios y acumulando en cada iteracion
def montecarlo_iterativo(n: int):

  # Inicializar contador
  tic = time.time()

  # Inicializar estimadores
  X = 0
  V = 0

  # Iterar segun el tamaño de muestra elegido
  for _ in range(0, n):
      
      # Sortear valores de t1 a t10 respetando sus distribuciones uniformes
      t1 = np.random.uniform(40, 56)
      t2 = np.random.uniform(24, 32)
      t3 = np.random.uniform(20, 40)
      t4 = np.random.uniform(16, 48)
      t5 = np.random.uniform(10, 30)
      t6 = np.random.uniform(15, 30)
      t7 = np.random.uniform(20, 25)
      t8 = np.random.uniform(30, 50)
      t9 = np.random.uniform(40, 60)
      t10 = np.random.uniform(8, 16)
      
      # Acumular valores de tiempo segun restricciones hasta llegar al tiempo de completitud
      # T2 y T3 dependen de T1
      T2 = t1 + t2
      T3 = t1 + t3
      # Buscar maximo entre T2 y T3 para acumular tareas que dependan de ambas
      T23 = -np.sort(-np.array([T2, T3]))[0]
      # T4 y T5 dependen de T2 y T3
      T4 = T23 + t4
      T5 = T23 + t5
      # T6 y T7 dependen de T3
      T6 = T3 + t6 # Depende de T3
      T7 = T3 + t7 # Depende de T3
      # Buscar maximo entre T4, T5, T6 y T7 para acumular tareas que dependan de todas
      T4567 = -np.sort(-np.array([T4, T5, T6, T7]))[0]
      # T8 depende de T4, T5, T6 y T7
      T8 = T4567 + t8
      # T9 depende de T5
      T9 = T5 + t9
      # Buscar maximo entre T7, T8 y T9 para acumular tareas que dependan de todas
      T789 = -np.sort(-np.array([T7, T8, T9]))[0]
      # T10 depende de T7, T8 y T9
      T10 = T789 + t10 

      # Acumular valor final en estimadores siguiendo sus respectivas formulas
      X += T10
      V += T10 ** 2

  # Calcular estimadores luego de acumulados los valores de la muestra
  X = X / n
  V = (V / (n * (n-1))) - ((X ** 2) / (n-1))

  # Apagar contador y calcular tiempo de ejecución
  toc = time.time()
  tiempo_de_ejecucion = toc - tic

  # Retornar estimadores para flujo principal
  return X, V, tiempo_de_ejecucion
    
# Version vectorial del metodo, utilizando operaciones vectoriales y evitando las iteraciones
def montecarlo_vectorizado(n: int):

  # Inicializar contador
  tic = time.time()
  
  # Sortear valores de t1 a t10 respetando sus distribuciones uniformes
  t1 = np.random.uniform(40, 56, n)
  t2 = np.random.uniform(24, 32, n)
  t3 = np.random.uniform(20, 40, n)
  t4 = np.random.uniform(16, 48, n)
  t5 = np.random.uniform(10, 30, n)
  t6 = np.random.uniform(15, 30, n)
  t7 = np.random.uniform(20, 25, n)
  t8 = np.random.uniform(30, 50, n)
  t9 = np.random.uniform(40, 60, n)
  t10 = np.random.uniform(8, 16, n)
  
  # Acumular valores de tiempo segun restricciones hasta llegar al tiempo de completitud
  # T2 y T3 dependen de T1
  T2 = t1 + t2
  T3 = t1 + t3
  # Buscar maximo entre T2 y T3 para acumular tareas que dependan de ambas
  T23 = np.maximum(T2, T3)
  # T4 y T5 dependen de T2 y T3
  T4 = T23 + t4
  T5 = T23 + t5
  # T6 y T7 dependen de T3
  T6 = T3 + t6 # Depende de T3
  T7 = T3 + t7 # Depende de T3
  # Buscar maximo entre T4, T5, T6 y T7 para acumular tareas que dependan de todas
  T4567 = np.max(np.stack((T4, T5, T6, T7), axis=0), axis=0)
  # T8 depende de T4, T5, T6 y T7
  T8 = T4567 + t8
  # T9 depende de T5
  T9 = T5 + t9
  # Buscar maximo entre T7, T8 y T9 para acumular tareas que dependan de todas
  T789 = np.max(np.stack((T7, T8, T9), axis=0), axis=0)
  # T10 depende de T7, T8 y T9
  T10 = T789 + t10 

  # Calcular estimadores
  X = np.mean(T10)
  V = (np.sum(np.square(T10)) / (n * (n-1))) - ((X ** 2) / (n-1))

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
modo = "iterativo" if len(sys.argv) <= 2 else sys.argv[2] # Modo iterativo o vectorizado
tipo_de_ejecucion = "simple" if len(sys.argv) <= 3 else sys.argv[3] # Tipo de ejecucion simple o busqueda por tiempos

# Para ejecucion simple, se ejecuta el método en el modo elegido y para la cantidad de iteraciones elegida
if tipo_de_ejecucion == "simple":

  # Ejecutar método de monte carlo dependiendo del modo elegido
  if modo == "vectorizado":
    X, V, tiempo_de_ejecucion = montecarlo_vectorizado(n)
  else:
    X, V, tiempo_de_ejecucion = montecarlo_iterativo(n)

  # Mostrar resultados
  print("Modo:", modo)
  print("Tiempo:", round(tiempo_de_ejecucion, 3), "segundos")
  print("Resultados:")
  print("X =", X)
  print("V =", V)

# Para ejecucion de busqueda, se ejecutan ambos modos para distintos valores de n, generando una tabla comparativa
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

    # Ejecutar método de monte carlo dependiendo del modo elegido
    if modo == "vectorizado":
      X, V, tiempo_de_ejecucion = montecarlo_vectorizado(n)
    else:
      X, V, tiempo_de_ejecucion = montecarlo_iterativo(n)

    # Generar matriz de resultados
    resultados_intermedios = [n, X, V, tiempo_de_ejecucion]
    resultados.append(resultados_intermedios)

    # Comprobar si el tiempo alcanzado
    tiempo_alcanzado = tiempo_de_ejecucion > 60

    # Mostrar resultados intermedios para tener seguimiento
    df = pd.DataFrame([resultados_intermedios], columns=["Iteraciones", "X", "V", "Tiempo"])
    print(df)

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

  # Generar gráfica de Xpara estudiar convergencia
  plt.figure(figsize=(8, 5))
  plt.plot(grafica_eje_x, grafica_eje_y_X, label='X', color='red', linewidth=2)
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
