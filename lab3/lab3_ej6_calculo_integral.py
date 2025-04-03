# DEPENDENCIAS
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# FUNCIONES AUXILIARES

def funcion_ej61(x):

  # Coordenadas del punto
  x1, x2 = x

  # Altura y radio del cerro
  H = 8.0
  r = 0.4

  # Calculo de la función
  distancia_centro = np.sqrt((x1 - 0.5)**2 + (x2 - 0.5)**2)
  return H - (H/r) * distancia_centro if distancia_centro <= r else 0.0

def funcion_ej62(x):

  # Coordenadas del punto
  x1, x2, x3, x4, x5 = x

  # Calculo de la funcion
  return x1 * (x2 ** 2) * (x3 ** 3) * (x4 ** 4) * (x5 ** 5)

# FUNCIONES PRINCIPALES

# Version iterativa del metodo, generando valores aleatorios y acumulando en cada iteracion
def montecarlo(n: int, funcion, semilla):

  # Inicializar contadores
  tic = time.time()

  # Iniciar semilla
  np.random.seed(semilla)

  # Inicializar estimadores
  estimador = 0.0
  acumulador = 0.0
  acumulador_previo = 0.0
  varianza_funcion = 0.0
  desviacion_estimador = 0.0

  # Iterar desde 1 a n+1 para evitar errores en calculo de varianza
  for i in range(1, n+1):
    
    # Sortear puntos de 5 coordenadas
    x1 = np.random.uniform(0, 1)
    x2 = np.random.uniform(0, 1)
    x3 = np.random.uniform(0, 1)
    x4 = np.random.uniform(0, 1)
    x5 = np.random.uniform(0, 1)

    # Calcular valor de la función 
    valor_funcion = funcion((x1, x2, x3, x4, x5))

    # Acumular valor de la función para posterior estimador
    acumulador += valor_funcion

    # Acumular calculo de la varianza de la función
    if i > 1:
      varianza_funcion += (1 - 1/i) * (valor_funcion - (acumulador_previo / (i - 1))) ** 2

    # Guardar acumulador actual para siguiente iteración
    acumulador_previo = acumulador

  # Calcular estimadores luego de acumulados los valores de la muestra
  estimador = acumulador / n
  varianza_funcion = varianza_funcion / (n-1)
  desviacion_estimador = np.sqrt(varianza_funcion / n)

  # Apagar contador y calcular tiempo de ejecución
  toc = time.time()
  tiempo_de_ejecucion = toc - tic

  # Retornar estimadores para flujo principal
  return estimador, varianza_funcion, desviacion_estimador, tiempo_de_ejecucion

# Calculo de intervalo de confianza siguiendo el método de aproximación normal
def intervalo_normal(estimador: float, desviacion_estimador: float, d: float):

  # Calculos previos para simplificar formula
  z = norm.ppf(1 - d/2)

  # Calculo de cota inferior y superior
  w1 = estimador - z * desviacion_estimador
  w2 = estimador + z * desviacion_estimador
  return w1, w2

# Calculo de tamaño de muestra según criterio de aproximación normal
def calculo_nn_normal(e: float, d: float, s: float):
  return math.ceil((norm.ppf(1 - (d / 2)) * s / e) ** 2)

# MÉTODO PRINCIPAL

# Setear precision de pandas
pd.set_option("display.float_format", "{:.10f}".format)

# Integral exacta calculada por métodos analíticos
integral_exacta = 1 / 720

# Leer parametros
n = int(sys.argv[1]) # Cantidad de iteraciones (n)
e = 0.0001 if len(sys.argv) <= 2 else float(sys.argv[2]) # Error (e)
d = 0.05 if len(sys.argv) <= 3 else float(sys.argv[3]) # Nivel de confianza (1-d)
tipo_de_ejecucion = "simple" if len(sys.argv) <= 4 else sys.argv[4] # Tipo de ejecucion simple, busqueda o cobertura

# Para ejecucion simple (parte a y b), se ejecuta el método para la cantidad de iteraciones elegida
if tipo_de_ejecucion == "simple":
   
  # Ejecutar método de monte carlo
  estimador, varianza_funcion, desviacion_estimador, tiempo_de_ejecucion = montecarlo(n, funcion_ej62, 42)

  # Calcular intervalo de confianza según método de aproximación normal
  w1, w2 = intervalo_normal(estimador, desviacion_estimador, d)
  
  # Calcular cantidad de replicaciones
  nN = calculo_nn_normal(e, d, np.sqrt(varianza_funcion))

  # Mostrar resultados
  print()
  print("Tiempo:", round(tiempo_de_ejecucion, 3), "segundos")
  print("Resultados:")
  print("Estimador:", estimador)
  print("Desviacion del estimador:", desviacion_estimador)
  print("Varianza de la función:", varianza_funcion)
  print("Intervalo de confianza:", w1, w2)
  print(f"Cantidad de iteraciones para e={e} y 1-d={1-d}:", nN)

# Para ejecucion de cobertura (parte c), se ejecuta el método para 500 distintas semillas y se calcula la cobertura empírica
elif tipo_de_ejecucion == "cobertura": 

  # Inicializar parámetros
  L = 500
  niveles_de_confianza = [0.10, 0.05, 0.01]
  coberturas = { nc: 0 for nc in niveles_de_confianza }

  # Inicializar auxiliares
  resultados = [] # Tabla de resultados
  contador = 0
  grafica_eje_x = range(0, L)
  grafica_eje_y_estimador = []
  grafica_eje_y_estimador_in = { nc: [] for nc in niveles_de_confianza }
  grafica_eje_y_confianza_w1 = { nc: [] for nc in niveles_de_confianza }
  grafica_eje_y_confianza_w2 = { nc: [] for nc in niveles_de_confianza }
  
  # Iterar para rango de semillas
  for i in range(0, L):
      
      # Ejecutar método de monte carlo
      estimador, varianza_funcion, desviacion_estimador, tiempo_de_ejecucion = montecarlo(n, funcion_ej62, i)
      
      # Iterar para valores de 1-d, calculando intervalo de confianza según método de aproximación normal
      for d_nc in niveles_de_confianza:
        
        # Calcular intervalo de confianza según método de aproximación normal
        w1, w2 = intervalo_normal(estimador, desviacion_estimador, d_nc)

        # Acumular si valor exacto se encuentra en intervalo de confianza
        if w1 <= integral_exacta <= w2:
          coberturas[d_nc] += 1

          # Acumular valores para gráfica
          grafica_eje_y_estimador_in[d_nc].append(estimador)

        # Si valor exacto no se encuentra en intervalo de confianza, guardar para gráfica
        else:

          # Acumular valores para gráfica
          grafica_eje_y_estimador_in[d_nc].append(0)

        # Acumular valores para gráfica
        grafica_eje_y_confianza_w1[d_nc].append(w1)
        grafica_eje_y_confianza_w2[d_nc].append(w2)

      # Acumular valores para gráfica
      grafica_eje_y_estimador.append(estimador)
  
  # Normalizar coberturas según cantidad de experimentos L
  coberturas_finales = { nc: coberturas[nc] / L for nc in niveles_de_confianza }
  resultados = [[1 - nc, coberturas_finales[nc]] for nc in niveles_de_confianza]

  # Mostrar resultados en una tabla
  pd.set_option("display.float_format", "{:.3f}".format)
  df = pd.DataFrame(resultados, columns=["Nivel de confianza (1-d)", "Cobertura empírica (%)"], index=range(1, len(resultados) + 1))
  print()
  print(df)

  # Generar gráficas para cada nivel de confianza
  for d_nc in niveles_de_confianza:

    # Generar gráfica de X para estudiar convergencia
    plt.figure(figsize=(8, 5))
    plt.axhline(y=integral_exacta, color='gold', linestyle='dashed', label='Integral exacta')
    plt.plot(grafica_eje_x, grafica_eje_y_confianza_w1[d_nc], label='w1', color='limegreen', linewidth=0.5)
    plt.plot(grafica_eje_x, grafica_eje_y_confianza_w2[d_nc], label='w2', color='limegreen', linewidth=0.5)
    plt.scatter(grafica_eje_x, grafica_eje_y_estimador, label='X', color='orangered', linewidth=1)
    plt.title(f"Comportamiento de estimador e intervalos para 1-d = {d_nc}")
    plt.xlabel("Número de experimento")
    plt.ylabel("Valor de X")
    plt.legend()
    plt.savefig(f'estimador_X_{1-d_nc}.png')

    # Generar gráfica de X en intervalo
    plt.figure(figsize=(8, 5))
    plt.axhline(y=integral_exacta, color='gold', linestyle='dashed', label='Integral exacta')
    plt.scatter(grafica_eje_x, grafica_eje_y_estimador_in[d_nc], label='X', color='navy', linewidth=0.5)
    plt.title(f"Comportamiento de estimador para 1-d = {1-d_nc} en relación al intervalo")
    plt.xlabel("Número de experimento")
    plt.ylabel("Valor de X")
    plt.legend()
    plt.savefig(f'estimador_X_in_{1-d_nc}.png')


# Para ejecucion de busqueda, se ejecuta el método para distintos valores de n, generando una tabla comparativa
else:

  # Inicializar auxiliares
  resultados = [] # Tabla de resultados
  contador = 0
  grafica_eje_x = []
  grafica_eje_y_estimador = []
  grafica_eje_y_varianza_f = []
  grafica_eje_y_varianza_e = []
  grafica_eje_y_confianza_w1 = []
  grafica_eje_y_confianza_w2 = []

  # Iterar para cada valor del rango 10^4 a 10^6, agregando entre medio el caso del n óptimo calculado en parte b
  for n in [10000, 36413, 100000, 1000000]:

    # Ejecutar método de monte carlo para cantidad de iteraciones correspondiente
    estimador, varianza_funcion, desviacion_estimador, tiempo_de_ejecucion = montecarlo(n, funcion_ej62, 42)
    print(f"Ejecutado para {n} valores")

    # Calcular intervalo de confianza según método de aproximación normal
    w1, w2 = intervalo_normal(estimador, desviacion_estimador, d)
    
    # Calcular cantidad de replicaciones
    nN = calculo_nn_normal(e, d, np.sqrt(varianza_funcion))

    # Generar matriz de resultados
    resultados_intermedios = [n, estimador, varianza_funcion, desviacion_estimador, w1, w2, nN, tiempo_de_ejecucion]
    resultados.append(resultados_intermedios)

    # Agregar cantidad de iteraciones a lista para grafica
    grafica_eje_x.append(["10^4", "36413", "10^5", "10^6"][contador])
    grafica_eje_y_estimador.append(estimador)
    grafica_eje_y_varianza_f.append(varianza_funcion)
    grafica_eje_y_varianza_e.append(desviacion_estimador)
    grafica_eje_y_confianza_w1.append(w1)
    grafica_eje_y_confianza_w2.append(w2)

    # Aumentar contador para grafica
    contador += 1

  # Mostrar resultados en una tabla
  df = pd.DataFrame(resultados, columns=[
     "Iteraciones", "Estimador (X)", "Varianza de la función (Vf)", "Desviacion del estimador (Vx)",
     "IdeC Normal (w1)", "IdeC Normal (w2)", "Cantidad de replicaciones óptima (nN)", 
     "Tiempo (s)"], index=range(1, len(resultados) + 1))
  print()
  print(df)

  # Generar gráfica de X para estudiar convergencia
  plt.figure(figsize=(8, 5))
  plt.axhline(y=integral_exacta, color='gold', linestyle='dashed', label='Integral exacta')
  plt.plot(grafica_eje_x, grafica_eje_y_estimador, label='X', color='orangered', linewidth=2)
  plt.plot(grafica_eje_x, grafica_eje_y_confianza_w1, label='w1', linestyle='dashed', color='limegreen', linewidth=2)
  plt.plot(grafica_eje_x, grafica_eje_y_confianza_w2, label='w2', linestyle='dashed', color='limegreen', linewidth=2)
  plt.title("Comportamiento de estimador puntual")
  plt.xlabel("Cantidad de iteraciones")
  plt.ylabel("Valor de X")
  plt.legend()
  plt.savefig('estimador_X.png')

  # Generar gráfica de V para estudiar convergencia
  plt.figure(figsize=(8, 5))
  plt.plot(grafica_eje_x, grafica_eje_y_varianza_f, label='Vf', color='royalblue', linewidth=2)
  plt.plot(grafica_eje_x, grafica_eje_y_varianza_e, label='Vx', color='gold', linewidth=2)
  plt.title("Comportamiento de estimadores de varianza")
  plt.xlabel("Cantidad de iteraciones")
  plt.ylabel("Valor de V y Vx")
  plt.legend()
  plt.savefig('estimador_V.png')
