# DEPENDENCIAS
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# FUNCIONES AUXILIARES

# Generar punto utilizando derivación del libro
def generar_punto():
    
    # Paso 1: Generar r con F(r) = r^2 -> r = sqrt(U(0,1))
    u = np.random.rand(1)[0]
    r = np.sqrt(u)

    # Paso 2: Generar Z1, Z2 = N(0,1)
    z1 = np.random.randn(1)[0]
    z2 = np.random.randn(1)[0]
    raiz = np.sqrt(z1**2 + z2**2)

    # Paso 3: Aplicar derivación y generar puntos en (0, 0) y radio 1
    x1 = r * z1 / raiz
    x2 = r * z2 / raiz

    # Paso 4: Transformar punto para círculo de centro (0.5, 0.5) y radio 0.4
    x1 = 0.5 + 0.4 * x1
    x2 = 0.5 + 0.4 * x2
  
    return x1, x2

# Calcular valor de la función altura
def funcion_ej61(x):

  # Coordenadas del punto
  x1, x2 = x

  # Altura y radio del cerro
  H = 8.0
  r = 0.4

  # Calculo de la función
  return H - (H/r) * np.sqrt((x1 - 0.5)**2 + (x2 - 0.5)**2)

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

  # Calcular área de base
  area = np.pi * 0.4 ** 2

  # Iterar desde 1 a n+1 para evitar errores en calculo de varianza
  for i in range(1, n+1):
    
    # Sortear puntos de 2 coordenadas
    x = generar_punto()

    # Calcular valor de la función 
    valor_funcion = funcion(x)

    # Acumular valor de la función para posterior estimador
    acumulador += valor_funcion

    # Acumular calculo de la varianza de la función
    if i > 1:
      varianza_funcion += (1 - 1/i) * (valor_funcion - (acumulador_previo / (i - 1))) ** 2

    # Guardar acumulador actual para siguiente iteración
    acumulador_previo = acumulador

  # Calcular estimadores luego de acumulados los valores de la muestra
  estimador = acumulador * area / n
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

# Leer parametros
n = int(sys.argv[1]) # Cantidad de iteraciones (n)
e = 0.0001 if len(sys.argv) <= 2 else float(sys.argv[2]) # Error (e)
d = 0.05 if len(sys.argv) <= 3 else float(sys.argv[3]) # Nivel de confianza (1-d)
tipo_de_ejecucion = "simple" if len(sys.argv) <= 4 else sys.argv[4] # Tipo de ejecucion simple, busqueda o cobertura

# Elegir funcion
funcion = funcion_ej61

# Para ejecucion simple (parte a y b), se ejecuta el método para la cantidad de iteraciones elegida
if tipo_de_ejecucion == "simple":

  # Ejecutar método de monte carlo
  estimador, varianza_funcion, desviacion_estimador, tiempo_de_ejecucion = montecarlo(n, funcion, 42)

  # Calcular intervalo de confianza según método de aproximación normal
  w1, w2 = intervalo_normal(estimador, desviacion_estimador, d)

  # Mostrar resultados
  print()
  print("Tiempo:", round(tiempo_de_ejecucion, 3), "segundos")
  print("Resultados:")
  print("Estimador:", estimador)
  print("Desviacion del estimador:", desviacion_estimador)
  print("Varianza de la función:", varianza_funcion)
  print("Intervalo de confianza:", w1, w2)

# Para ejecucion de busqueda, se ejecuta el método para distintos valores de n, generando una tabla comparativa
else:

  # Inicializar auxiliares
  lista_n = [10000, 100000, 1000000, 1000000]
  resultados = [] # Tabla de resultados
  contador = 0
  grafica_eje_x = []
  grafica_eje_y_estimador = []
  grafica_eje_y_varianza_f = []
  grafica_eje_y_varianza_e = []
  grafica_eje_y_confianza_w1 = []
  grafica_eje_y_confianza_w2 = []

  # Iterar para cada valor del rango 10^4 a 10^6
  for n in lista_n:

    # Ejecutar método de monte carlo para cantidad de iteraciones correspondiente
    estimador, varianza_funcion, desviacion_estimador, tiempo_de_ejecucion = montecarlo(n, funcion, 42)
    print(f"Ejecutado para {n} valores")

    # Calcular intervalo de confianza según método de aproximación normal
    w1, w2 = intervalo_normal(estimador, desviacion_estimador, d)

    # Generar matriz de resultados
    resultados_intermedios = [n, estimador, varianza_funcion, desviacion_estimador, w1, w2, tiempo_de_ejecucion]
    resultados.append(resultados_intermedios)

    # Agregar cantidad de iteraciones a lista para grafica
    grafica_eje_x.append(["10^4", "10^5", "10^6", "10^7"][contador])
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
     "IdeC Normal (w1)", "IdeC Normal (w2)", 
     "Tiempo (s)"], index=range(1, len(resultados) + 1))
  print()
  print(df)

  # Generar gráfica de X para estudiar convergencia
  plt.figure(figsize=(8, 5))
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
