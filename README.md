# Métodos de Monte Carlo (2025)
Tareas del curso de Métodos de Monte Carlo 2025, Facultad de Ingeniería (UdelaR).

## Requisitos

1. Instalar *Python 3.12*

2. Crear ambiente virtual <br>
`python -m venv env`

3. Instalar requerimientos <br>
`pip install -r requirements.txt`

4. Ejecutar script elegido <br>
`python lab.py <parámetros>`

## Ejecución

### Laboratorio 1
Esta entrega cuenta con un único script `lab1.py`, el cual ejecuta un método de monte carlo para la estimación de una variable aleatoria ofrecida en la consigna.

Para ejecutarlo, es necesario correr el comando <br>
`python lab1/lab1.py <tamaño_de_muestra> <modo> <busqueda>` <br>
- `tamaño_de_muestra` es un entero, el único parámetro obligatorio.
- `modo` es un string que puede ser `"iterativo"` o `"vectorizado"`, en relación al modo de ejecución del método de monte carlo. Por defecto el valor es `"iterativo"`.
- `busqueda` es un string que puede ser `"simple"` o `"busqueda"`, establece si ejecutar el método para el tamaño de muestra dado o si realizar una exploración con distintos tamaños de muestra y generar tablas y gráficas. Por defecto el valor es `"simple"`.


### Laboratorio 2
Esta entrega cuenta con dos scripts `lab2_calculo_volumen.py`, el cual ejecuta un método de monte carlo para la estimación del volumen de una región ofrecida en la consigna, y `lab2_calculo_muestra.py`, el cual se utiliza para generar los resultados para distintas derivaciones de tamaño de muestra de peor caso siguiendo la consigna.

Para ejecutar `lab2_calculo_volumen.py`, es necesario correr el comando <br>
`python lab2/lab2_calculo_volumen.py <tamaño_de_muestra> <modo> <busqueda>` <br>
- `tamaño_de_muestra` es un entero, el único parámetro obligatorio.
- `modo` es un string que puede ser `"con_restricciones"` o `"sin_restricciones"`, establece si inlcuir las restricciones de la región o simplemente utilizar los valores de la hiperesfera. Por defecto el valor es `"con_restricciones"`.
- `busqueda` es un string que puede ser `"simple"` o `"busqueda"`, establece si ejecutar el método para el tamaño de muestra dado o si realizar una exploración con distintos tamaños de muestra y generar tablas y gráficas. Por defecto el valor es `"simple"`.

Para ejecutar `lab2_calculo_muestra.py`, es necesario correr el comando <br>
`python lab2/lab2_calculo_muestra.py`