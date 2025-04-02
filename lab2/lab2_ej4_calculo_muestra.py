# DEPENDENCIAS
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

# Inicializar parametros
e = 0.01
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

