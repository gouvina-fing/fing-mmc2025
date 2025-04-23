# DEPENDENCIAS
import sys
import time
import math
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# FUNCIONES AUXILIARES

def obtener_random(batch: int):

  # 
  url = f"https://api.quantumnumbers.anu.edu.au?length={batch}&type=uint16"
  key = "P1s1OpSBXo7BthkNmZ61lGq2XdwrknZ66hibhXA3"

  #
  response = requests.get(url, headers={"x-api-key": key})

  # 
  data = response.json()

  print(data)
  
  #
  data = np.array(data["data"]) / 65535

  return data

#
def pertenece(p: float):
    
    #
    centro = np.array([0.45, 0.5, 0.6, 0.6, 0.5, 0.45])
    radio = 0.35
    
    # Verifico si el punto está dentro de la hiperesfera
    if np.sum((p - centro) ** 2) > radio ** 2:
        return False
    
    # 
    x1, x2, x3, x4, x5, x6 = p
    if not (3*x1 + 7*x4 <= 5):
        return False
    if not (x3 + x4 <= 1):
        return False
    if not (x1 - x2 - x5 + x6 >= 0):
        return False
    
    #
    return True

# FUNCIONES PRINCIPALES

# Version iterativa y vectorial del metodo, generando valores aleatorios y acumulando en cada iteracion
def montecarlo_batch(n: int, batch: int):

  # Inicializar contadores
  tic = time.time()
  iteraciones, sobrantes = divmod(n, batch)

  # Inicializar estimadores
  estimador = 0
  acumulador = 0
  desviacion = 0

  # Iterar segun coeficiente tamaño_de_muestra / tamaño_de_batch + 1, para el caso donde sobren iteraciones
  for i, chunk in enumerate(pd.read_csv("lab5/tabla.csv", chunksize=batch, header=None)):
    
    # Para última iteración, saltarla si no hay sobrantes
    if sobrantes == 0 and i == iteraciones:
      continue

    # Transformar filas en puntos
    puntos = chunk.to_numpy()

    # Comprobar para cada punto si pertenece a la region y acumular suma total de puntos que pertenecen
    acumulador += np.sum([pertenece(p) for p in puntos])

    # Mostrar mensaje intermedio para ejecuciones largas
    if i < iteraciones:
      print(f"Ejecutado para {(i+1) * batch} valores")
    else:
      print(f"Ejecutado para {(i * batch) + sobrantes} valores")

  # Calcular estimadores luego de acumulados los valores de la muestra
  estimador = acumulador / n
  desviacion = math.sqrt((estimador * (1-estimador)) / (n-1))

  # Apagar contador y calcular tiempo de ejecución
  toc = time.time()
  tiempo_de_ejecucion = toc - tic

  # Retornar estimadores para flujo principal
  return estimador, desviacion, acumulador, tiempo_de_ejecucion

# Calculo de intervalo de confianza siguiendo el método de desigualdad de Chebyshev
def intervalo_chebyshev(S: float, n: int, b: float):

  # Calculos previos para simplificar formula
  b2 = b ** 2
  raiz = (b * math.sqrt((b2 / 4) + ((S * (n-S)) / n)))

  # Calculo de cota inferior y superior
  w1 = (S + (b2 / 2) - raiz) / (n + b2)
  w2 = (S + (b2 / 2) + raiz) / (n + b2)
  return w1, w2

# Calculo de intervalo de confianza siguiendo el método de Agresti-Coull
def intervalo_agresti_coull(S: float, n: int, d: float):

  # Calculos previos para simplificar formula
  k = norm.ppf(1 - (d / 2))
  S_tilde = S + (k ** 2) / 2
  n_tilde = n + (k ** 2)
  p_tilde = S_tilde / n_tilde
  q_tilde = 1 - p_tilde

  # Calculo de cota inferior y superior
  w1 = p_tilde - (k * math.sqrt(p_tilde * q_tilde) / math.sqrt(n_tilde))
  w2 = p_tilde + (k * math.sqrt(p_tilde * q_tilde) / math.sqrt(n_tilde))
  return w1, w2

# MÉTODO PRINCIPAL

# Iniciar semilla
np.random.seed(42)

# Setear precision de pandas
pd.set_option("display.float_format", "{:.10f}".format)

# Leer parametros
n = int(sys.argv[1]) # Cantidad de iteraciones (n)
batch = 1000 if len(sys.argv) <= 2 else int(sys.argv[2]) # Nivel de confianza (1-d)
d = 0.05 if len(sys.argv) <= 3 else float(sys.argv[3]) # Nivel de confianza (1-d)
intervalo = "chebyshev" if len(sys.argv) <= 4 else sys.argv[4] # Tipo de intervalo de confianza (chebyshev o agresti-coull)
tipo_de_ejecucion = "simple" if len(sys.argv) <= 5 else sys.argv[5] # Tipo de ejecucion simple o busqueda por tiempos

# Para ejecucion simple, se ejecuta el método para la cantidad de iteraciones elegida
if tipo_de_ejecucion == "simple":
   
  # Ejecutar método de monte carlo
  estimador, desviacion, acumulador, tiempo_de_ejecucion = montecarlo_batch(n, batch)

  # Calcular intervalo de confianza segun método elegido
  if intervalo == "chebyshev":
    w1, w2 = intervalo_chebyshev(acumulador, n, 1 / math.sqrt(d))
  else:
    w1, w2 = intervalo_agresti_coull(acumulador, n, d)

  # Mostrar resultados
  print()
  print("Tiempo:", round(tiempo_de_ejecucion, 3), "segundos")
  print("Resultados:")
  print("Acumulador:", acumulador)
  print("Estimador:", estimador)
  print("Desviacion estandar:", desviacion)
  print("Intervalo de confianza:", w1, w2)

# Para ejecucion de busqueda, se ejecuta el métod para distintos valores de n, generando una tabla comparativa
else: 

  # Inicializar parametros
  batch = 100000
  d = 0.05

  # Inicializar auxiliares
  resultados = [] # Tabla de resultados
  contador = 1
  grafica_eje_x = []
  grafica_eje_y_estimador = []
  grafica_eje_y_desviacion = []
  grafica_eje_y_confianza_w1 = []
  grafica_eje_y_confianza_w2 = []
  grafica_eje_y_confianza_w3 = []
  grafica_eje_y_confianza_w4 = []

  # Iterar para cada valor del rango 10^4 a 10^8, agregando como ultimo caso el de n optimo 
  for n in [10000, 100000, 1000000, 10000000, 100000000, 184443973]:

    # Ejecutar método de monte carlo para cantidad de iteraciones correspondiente
    estimador, desviacion, acumulador, tiempo_de_ejecucion = montecarlo_batch(n, batch)
    print()

    # Calcular intervalo de confianza para ambos metodos
    w1, w2 = intervalo_chebyshev(acumulador, n, 1 / math.sqrt(d))
    w3, w4 = intervalo_agresti_coull(acumulador, n, d)

    # Generar matriz de resultados
    resultados_intermedios = [n, estimador, desviacion, w1, w2, w3, w4, tiempo_de_ejecucion]
    resultados.append(resultados_intermedios)

    # Agregar cantidad de iteraciones a lista para grafica
    grafica_eje_x.append(f"10^{contador + 3}")
    grafica_eje_y_estimador.append(estimador)
    grafica_eje_y_desviacion.append(desviacion)
    grafica_eje_y_confianza_w1.append(w1)
    grafica_eje_y_confianza_w2.append(w2)
    grafica_eje_y_confianza_w3.append(w3)
    grafica_eje_y_confianza_w4.append(w4)

    # Aumentar contador para grafica
    contador += 1

  # Mostrar resultados en una tabla
  df = pd.DataFrame(resultados, columns=[
     "Iteraciones", "Estimador (X)", "Desviacion (V)",
     "IdeC Chebyshev (w1)", "IdeC Chebyshev (w2)", "IdeC Agresti-Coull (w1)","IdeC Agresti-Coull (w2)", 
     "Tiempo (s)"], index=range(1, len(resultados) + 1))
  print()
  print(df)

  # Generar gráfica de X para estudiar convergencia
  plt.figure(figsize=(8, 5))
  plt.plot(grafica_eje_x, grafica_eje_y_estimador, label='X', color='orangered', linewidth=2)
  plt.plot(grafica_eje_x, grafica_eje_y_confianza_w1, label='w1 (ch)', linestyle='dashed', color='limegreen', linewidth=2)
  plt.plot(grafica_eje_x, grafica_eje_y_confianza_w2, label='w2 (ch)', linestyle='dashed', color='limegreen', linewidth=2)
  plt.plot(grafica_eje_x, grafica_eje_y_confianza_w3, label='w1 (ac)', linestyle='dashed', color='gold', linewidth=2)
  plt.plot(grafica_eje_x, grafica_eje_y_confianza_w4, label='w2 (ac)', linestyle='dashed', color='gold', linewidth=2)
  plt.title("Comportamiento de estimador X")
  plt.xlabel("Cantidad de iteraciones")
  plt.ylabel("Valor de X")
  plt.legend()
  plt.savefig('estimador_X.png')

  # Generar gráfica de V para estudiar convergencia
  plt.figure(figsize=(8, 5))
  plt.plot(grafica_eje_x, grafica_eje_y_desviacion, label='V', color='royalblue', linewidth=2)
  plt.title("Comportamiento de estimador V")
  plt.xlabel("Cantidad de iteraciones")
  plt.ylabel("Valor de V")
  plt.legend()
  plt.savefig('estimador_V.png')
