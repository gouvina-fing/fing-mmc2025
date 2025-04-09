# DEPENDENCIAS
import sys
import time
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm

# FUNCIONES AUXILIARES

# Crear matriz de ejemplo
def leer_matriz():
  return np.array([
    [1,1,1,0,0,1,0,1],
    [1,0,1,1,0,1,1,0],
    [1,0,0,1,1,0,1,1],
    [1,1,0,1,0,1,1,1],
    [1,0,1,0,0,1,0,1],
    [0,1,0,1,0,0,1,1],
    [0,0,1,0,1,0,0,1],
    [0,1,1,0,1,0,1,1],
    [0,1,0,0,1,1,0,1],
    [0,0,1,1,0,1,1,1],
    [0,1,0,0,1,1,1,0],
    [0,0,1,1,1,1,1,0],
    [1,1,0,0,0,1,1,1],
    [1,1,0,1,0,1,1,0],
    [0,1,1,0,1,1,1,1]
  ])

# FUNCIONES PRINCIPALES

# Version iterativa  del metodo, generando valores aleatorios y acumulando en cada iteracion
def montecarlo_iterativo(n: int, E: int, Z: int, M: int):

  # Leer matriz
  matriz = leer_matriz()

  # Calcular cardinal del objeto a estudiar, es decir
  # Todas las combinaciones posibles de tomar M zonas del total de Z
  X = math.comb(Z, M)

  # Inicializar contadores
  tic = time.time()

  # Inicializar estimadores
  estimador = 0
  acumulador = 0
  desviacion = 0

  # Iterar segun coeficiente tamaño_de_muestra / tamaño_de_batch + 1, para el caso donde sobren iteraciones
  for i in range(0, n):

    # Inicializar conjuntos Z, M y E
    Z_conjunto = list(range(Z)) # Se acumulan las Z zonas totales
    M_conjunto = [] # Se acumulan las M especies sorteadas
    E_conjunto = set() # Se acumulan las especies de las M zonas

    # Sortear M zonas sin repeticiones para optimizar
    for _ in range(0, M):

      # Se sortea en el largo Z - N° de iteracion (len(Z_conjunto))
      index = np.random.randint(0, len(Z_conjunto))

      # Se obtiene el valor de la zona z
      z = Z_conjunto[index]

      # Se agrega la zona z al conjunto M para hacer el chequeo luego
      M_conjunto.append(z)

      # Eliminar zona z de la lista de zonas factibles Z para no repetir
      del Z_conjunto[index]

    # Iterar segun las M zonas sorteadas anteriormente
    for z in M_conjunto:

      # Iterar segun las E especies
      for e in range(0,E):

        # Si la especie e se encuentra en la zona z, se agrega la especie a la lista
        if matriz[z][e] == 1:
          E_conjunto.add(e) # Se agrega a un conjunto entonces no se repite

    # Acumular en el caso de que se hayan encontrado las E especies en las M zonas
    if len(E_conjunto) == E:
      acumulador += 1

  # Calcular estimadores luego de acumulados los valores de la muestra
  # Se agrega el valor de |X| en el calculo
  estimador = X * acumulador / n
  desviacion = np.sqrt((estimador * (X - estimador)) / (n-1))

  # Apagar contador y calcular tiempo de ejecución
  toc = time.time()
  tiempo_de_ejecucion = toc - tic

  # Retornar estimadores para flujo principal
  return estimador, desviacion, acumulador, X, tiempo_de_ejecucion

# Calculo de intervalo de confianza siguiendo el método de Agresti-Coull
def intervalo_agresti_coull(S: float, X: int, n: int, d: float):

  # Calculos previos para simplificar formula
  k = norm.ppf(1 - (d / 2))
  S_tilde = (X * S) + (k ** 2) / 2 # El acumulador S se sustitutye por XS
  n_tilde = n + (k ** 2)
  p_tilde = S_tilde / n_tilde

  # Calculo de cota inferior y superior
  # El termino 1-p_tilde se sustitutye por X-p_tilde
  w1 = p_tilde - (k * np.sqrt(p_tilde * (X - p_tilde) / n_tilde)) 
  w2 = p_tilde + (k * np.sqrt(p_tilde * (X - p_tilde) / n_tilde))
  return w1, w2

# MÉTODO PRINCIPAL

# Iniciar semilla
np.random.seed(42)

# Setear precision de pandas
pd.set_option("display.float_format", "{:.10f}".format)

# Leer parametros
tipo_de_ejecucion = "simple" if len(sys.argv) <= 1 else sys.argv[1] # Tipo de ejecucion simple o busqueda por tiempos
n = 1000 if len(sys.argv) <= 2 else int(sys.argv[2]) # Cantidad de iteraciones (n)
E = 8 if len(sys.argv) <= 3 else int(sys.argv[3]) # Cantidad de especies (|E|)
Z = 15 if len(sys.argv) <= 4 else int(sys.argv[4]) # Cantidad de zonas (|Z|)
M = 5 if len(sys.argv) <= 5 else int(sys.argv[5]) # Cantidad de zonas a elegir (M)
d = 0.05 if len(sys.argv) <= 6 else float(sys.argv[6]) # Nivel de confianza (1-d)

# Para ejecucion simple, se ejecuta el método para la cantidad de iteraciones elegida
if tipo_de_ejecucion == "simple":
   
  # Ejecutar método de monte carlo
  estimador, desviacion, acumulador, cardinal, tiempo_de_ejecucion = montecarlo_iterativo(n, E, Z, M)

  # Calcular intervalo de confianza
  w1, w2 = intervalo_agresti_coull(acumulador, cardinal, n, d)

  # Mostrar resultados
  print()
  print("Tiempo:", round(tiempo_de_ejecucion, 3), "segundos")
  print("Resultados:")
  print("Acumulador:", acumulador)
  print("Estimador:", estimador)
  print("Desviacion:", desviacion)
  print("Intervalo de confianza:", w1, w2)

# Para ejecucion de busqueda, se ejecuta el métod para distintos valores de n, generando una tabla comparativa
else:

  # Iterar para los valores de M pedidos
  for m in [M, M+1]:

    # Inicializar auxiliares
    resultados = [] # Tabla de resultados
    contador = 1
    grafica_eje_x = []
    grafica_eje_y_estimador = []
    grafica_eje_y_desviacion = []
    grafica_eje_y_confianza_w1 = []
    grafica_eje_y_confianza_w2 = []

    # Iterar para cada valor del rango 10^3 a 10^6 
    for n in [1000, 10000, 100000, 1000000]:

      # Ejecutar método de monte carlo para cantidad de iteraciones correspondiente
      estimador, desviacion, acumulador, cardinal, tiempo_de_ejecucion = montecarlo_iterativo(n, E, Z, m)

      # Calcular intervalo de confianza
      w1, w2 = intervalo_agresti_coull(acumulador, cardinal, n, d)

      # Generar matriz de resultados
      resultados_intermedios = [n, estimador, desviacion, w1, w2, tiempo_de_ejecucion]
      resultados.append(resultados_intermedios)

      # Agregar cantidad de iteraciones a lista para grafica
      grafica_eje_x.append(f"10^{contador + 2}")
      grafica_eje_y_estimador.append(estimador)
      grafica_eje_y_desviacion.append(desviacion)
      grafica_eje_y_confianza_w1.append(w1)
      grafica_eje_y_confianza_w2.append(w2)

      # Aumentar contador para grafica
      contador += 1

    # Mostrar resultados en una tabla
    df = pd.DataFrame(resultados, columns=[
      "Iteraciones", "Estimador (X)", "Desviacion (V)", "IdeC Agresti-Coull (w1)","IdeC Agresti-Coull (w2)", 
      "Tiempo (s)"], index=range(1, len(resultados) + 1))
    print()
    print(df)

    # Generar gráfica de X para estudiar convergencia
    plt.figure(figsize=(8, 5))
    plt.plot(grafica_eje_x, grafica_eje_y_estimador, label='X', color='orangered', linewidth=2)
    plt.plot(grafica_eje_x, grafica_eje_y_confianza_w1, label='w1 (ac)', linestyle='dashed', color='gold', linewidth=2)
    plt.plot(grafica_eje_x, grafica_eje_y_confianza_w2, label='w2 (ac)', linestyle='dashed', color='gold', linewidth=2)
    plt.title(f"Comportamiento de estimador X para M = {m}")
    plt.xlabel("Cantidad de iteraciones")
    plt.ylabel("Valor de X")
    plt.legend()
    plt.savefig(f'estimador_X{m}.png')

    # Generar gráfica de V para estudiar convergencia
    plt.figure(figsize=(8, 5))
    plt.plot(grafica_eje_x, grafica_eje_y_desviacion, label='V', color='royalblue', linewidth=2)
    plt.title(f"Comportamiento de estimador V para M = {m}")
    plt.xlabel("Cantidad de iteraciones")
    plt.ylabel("Valor de V")
    plt.legend()
    plt.savefig(f'estimador_V{m}.png')
