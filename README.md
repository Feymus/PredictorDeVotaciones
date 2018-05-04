Predictor De Votaciones
=====================

La clasificación es aquel proceso que nos permite asignar objetos a diferentes categorías predefinidas, un modelo de clasificación intenta extraer alguna conclusión de los valores observados. Dadas una o más entradas al modelo de clasificación, intentaremos predecir el valor de uno o más resultados. Los resultados son etiquetas que se pueden aplicar a un conjunto de datos. Por ejemplo, al filtrar correos electrónicos "spam" o "no spam"

En este proyecto se realizaron múltiples modelos de clasificación que nos permiten predecir los votos de las elecciones del año 2018 en Costa Rica (primera ronda, segunda ronda y segunda ronda basandonos en los votos de la primera), basándose para esto en un conjunto de muestras generadas aleatoriamente con los indicadores proporcionados por la PEN (Programa Estado de la Nación) y los votos de la primera y segunda ronda.

## Aspectos Técnicos

En esta sección se pretende abarcar las herramientas utilizadas y demás aspectos técnicos del proyecto.

### Simulador de votantes

Con este modulo generador de votantes (desarrollado previamente a este proyecto) se generará muestras de votantes en las votaciones de Costa Rica. El generador puede crear muestras tanto para todo el país como para solo cierta provincia.

**Sobre los datos de las muestras**

Los datos utilizados para generar estas muestra han sido recolectados de las Actas de Sesión de la primera y segunda ronda de elecciones del año 2018 y de los Indicadores Cantonales y Censos del año 2011.

El único dato que se sacó de un archivo externo fue el de rango de edades, para el cual se han utilizado los datos del documento de *Estimaciones y Proyecciones de Población por sexo y edad (1950-2100)* de la INEC. Más específicamente de la sección *Proyecciones de población del periodo 2000-2050* (cuadros 2.3 y 2.4)

**Funcionamiento del simulador**

Como fue mencionado el simulador utiliza los datos de Indicadores Cantonales y Censos, para cada votante generado se aletoriza cada una de las propiedades del votante segun estos censos, y para la elección de canton se decide al azar segun los votos de primera ronda. al igual que las propiedaes el voto de la persona generada es establecido de manera aleatoria, esto segun las Actas de Sesión de primera o segunda ronda (segun sea necesario).

### Librerías utilizadas

Para la clasificación SVM y manejo de los datos se usó principalmente la librería scikit de python junto con numpy. En la clasificación por Modelos Lineales se utilizó la librería de Tensorflow para la creación de los tensores que nos ayudan a la clasificación de los datos por medio de una regresión logística y la librería OneHotEncoder, el cual es un proceso mediante el cual las variables categóricas se convierten en una forma que podría proporcionarse a los algoritmos de modelos lineales para hacer un mejor trabajo en la predicción. La clasificación por Redes Neuronales además de haber utilizado las librerías mencionadas anteriormente con el mismo objetivo, a este se le incorpora la librería de Keras, el cuál nos permite crear las capas de las neuronal 

## Reportes
### Modelos lineales
### Redes neuronales
### Árboles de decisión
### KNN
### SVM

**Parametros del modelo**

El modelo trabaja con tres diferentes parametros, el kernel, C y gamma:

Kernel: Un kernel es una función de similitud. Se proporciona a un algoritmo de aprendizaje automático el cual toma dos entradas y retorna que tan similares son.

C: Intercambia errores de clasificación de ejemplos de entrenamiento contra la simplicidad de la superficie de decisión. Una C baja hace que la superficie de decisión sea suave, mientras que una C alta tiene como objetivo clasificar correctamente todos los ejemplos de entrenamiento. En otras palabras C define cuánto se quiere evitar clasificar erróneamente cada ejemplo.

Gamma: Define cuánta influencia tiene un único ejemplo de entrenamiento. Cuanto más grande es gamma, más cerca deben verse otros ejemplos para ser afectados.

**Análisis de resultados**

Para el análisis del modelo se pretende utilizar muestras de tamaños 100, 1000 y 10000, para todas se guardará un dos por ciento de las muestras para realizar la prueba final. Además, en SVM se probarán los kernel "rbf" y "sigmoid", para los valores de C se probarán valores 1 y 10, para gamma se probarán valores exponenciales de 1 a 0.000000001 y el valor auto (que se calcula segun la cantidad de propiedades).

Cada prueba muestra el error de entrenamiento (ER) promedio del modelo luego de 30 corridas.

Pruebas (rbf):

1) Kernel: rbf, C: 1, Gamma: 1

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     |  0.772  |   0.77    |           |
| Segunda ronda     |  0.444  |   0.41    |           |
| Basado en primera |  0.445  |   0.39    |           |

2) Kernel: rbf, C: 1, Gamma: 0.000000001

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.792   |  0.766    |           |
| Segunda ronda     | 0.442   |  0.4      |           |
| Basado en primera | 0.442   |  0.4      |           |

2) Kernel: rbf, C: 1, Gamma: auto

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.762   |  0.74     |           |
| Segunda ronda     | 0.432   |  0.3975   |           |
| Basado en primera | 0.43    |  0.3975   |           |

1) Kernel: rbf, C: 10, Gamma: 1

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.792   | 0.743     |           |
| Segunda ronda     | 0.385   | 0.413     |           |
| Basado en primera | 0.385   | 0.408     |           |

2) Kernel: rbf, C: 10, Gamma: 0.000000001

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.81    | 0.76      |           |
| Segunda ronda     | 0.415   | 0.41      |           |
| Basado en primera | 0.415   | 0.41      |           |

2) Kernel: rbf, C: 10, Gamma: auto

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.78    | 0.76      |           |
| Segunda ronda     | 0.447   | 0.41      |           |
| Basado en primera | 0.457   | 0.407     |           |

Pruebas (sigmoid):

1) Kernel: sigmoid, C: 1, Gamma: 1

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.735   | 0.7455    |           |
| Segunda ronda     | 0.417   | 0.41      |           |
| Basado en primera | 0.417   | 0.41      |           |

2) Kernel: sigmoid, C: 1, Gamma: 0.000000001

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.783   | 0.806     |           |
| Segunda ronda     | 0.433   | 0.489     |           |
| Basado en primera | 0.433   | 0.489     |           |

3) Kernel: sigmoid, C: 1, Gamma: auto

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.785   | 0.75      |           |
| Segunda ronda     | 0.397   | 0.387     |           |
| Basado en primera | 0.397   | 0.387     |           |

4) Kernel: sigmoid, C: 10, Gamma: 1

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.775   | 0.757     |           |
| Segunda ronda     | 0.398   | 0.409     |           |
| Basado en primera | 0.398   | 0.409     |           |

5) Kernel: sigmoid, C: 10, Gamma: 0.000000001

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.82    | 0.8       |           |
| Segunda ronda     | 0.51    | 0.47      |           |
| Basado en primera | 0.51    | 0.47      |           |

6) Kernel: sigmoid, C: 10, Gamma: auto

|                   |   100   |   1000    |   10000   |
|-------------------|---------|-----------|-----------|
| Primera ronda     | 0.772   | 0.75      |           |
| Segunda ronda     | 0.428   | 0.408     |           |
| Basado en primera | 0.428   | 0.408     |           |

## Manual de usuario