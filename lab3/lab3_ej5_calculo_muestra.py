# DEPENDENCIAS
import sys
import math
import pandas as pd
from scipy.stats import norm

# FUNCIONES PRINCIPALES

def calculo_nc(e: float, d: float):
  return math.ceil(1 / (4 * d * (e ** 2)))

def calculo_nn(e: float, d: float):
  return math.ceil(((norm.ppf(1 - (d / 2))) / (2 * e)) ** 2)

def calculo_nh(e: float, d: float):
  return math.ceil((2 * math.log(2 / d)) / (4 * (e ** 2)))


# MÉTODO PRINCIPAL

# Leer parametros
tipo_de_ejecucion = "simple" if len(sys.argv) <= 1 else sys.argv[1] # Tipo de ejecucion simple o busqueda en lista
e = 0.01 if len(sys.argv) <= 2 else float(sys.argv[2]) # Epsilon, error
d = 0.05 if len(sys.argv) <= 3 else float(sys.argv[3]) # Delta, nivel de confianza
algoritmo = "nc" if len(sys.argv) <= 4 else sys.argv[4] # Algoritmo

# Para ejecucion simple, se ejecuta el método para el algoritmo y los valores de e y d elegidos
if tipo_de_ejecucion == "simple":

  # Inicializar valor
  resultado = 0

  # Ejecutar algoritmo elegido
  if algoritmo == "nc":
    resultado = calculo_nc(e, d)
  elif algoritmo == "nn":
    resultado = calculo_nn(e, d)
  elif algoritmo == "nh":
    resultado = calculo_nh(e, d)

  # Acumular resultados en lista
  resultados = [[e, 1-d, algoritmo, resultado]]

  # Mostrar resultados en una tabla
  df = pd.DataFrame(resultados, columns=["Error (e)", "Nivel de confianza (1-d)", "Algoritmo", "Tamaño de muestra"])
  print()
  print(df)

# Para ejecucion de busqueda, se ejecuta el método para distintos valores de d, generando una tabla comparativa
else: 
  
  # Inicializar parametros
  ds = [0.001, 0.01, 0.05]
  resultados = []

  # Iterar segun valores de delta
  for d in ds:

    # Calcular tamaño de muestra para cada metodo
    nc = calculo_nc(e, d)
    nn = calculo_nn(e, d)
    nh = calculo_nh(e, d)

    # Acumular resultados en matriz
    resultados_intermedios = [d, nc, nn, nh]
    resultados.append(resultados_intermedios)

    # Mostrar resultados en una tabla
    df = pd.DataFrame(resultados, columns=["Nivel de confianza", "nC", "nN", "nH"], index=range(1, len(resultados) + 1))
    print()
    print(df)

