from sklearn.model_selection import train_test_split
import time
import copy
from Normalizer import Normalizer
from SVMClassifier import SVMClassifier

from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)
from math import log2, sqrt
import random


class Kd_Tree():

    '''
    Metodo por default que inicializa la clase.
    Entrada: neightboards es la cantidad de vecinos a tomar en cuenta.
    Salida: NA.
    '''

    def __init__(self, neightboards):
        self.tree = None
        self.r = None
        self.neightboards = neightboards
        self.results = None
    '''
    Entrena el modelo.
    Entrada: Conjunto de muestras.
    Salida: NA
    '''

    def train(self, samples):
        self.tree = None
        self.r = None
        self.results = copy.copy(samples.get('trainingClasses'))
        samples2 = copy.copy(samples.get('trainingFeatures'))

        self.r = samples2
        self.tree = kd_trees(list(self.r), 0)
    '''
    Clasifica una muestra segun el modelo.
    Entrada: Muestra a evaluar.
    Salida: Resultado de la clasificacion.
    '''

    def classify(self, test):

        test = copy.copy(test[0])
        mini_test = kdtree_closest_point(self.tree, test, 0, []).tolist()

        neightboards = []

        points = top_points(test, self.neightboards)

        for j in points:
            samples_list = self.r.tolist()
            pos = samples_list.index(j)

            o = self.results.tolist()[pos]
            neightboards += [o]

        return [best_vote_percent(neightboards)]


class BinaryTree():

   '''
   Metodo por default que inicializa la clase.
   Entrada: NA.
   Salida: NA.
   '''

   def __init__(self):
        self.left = None
        self.right = None
        self.dimension = None


    '''
    Obtiene el hijo izquierdo del arbol.
    Entrada: NA
    Salida: Arbol del hijo izquierdo.
    '''
    
    def getLeftChild(self):
        return self.left
    '''
    Metodo por default que inicializa la clase.
    Entrada: neightboards es la cantidad de vecinos a tomar en cuenta.
    Salida: NA.
    '''

    def getRightChild(self):
        return self.right
    '''
    Metodo que obtiene la dimension.
    Entrada: NA
    Salida: Número resultado de la dimension.
    '''

    def getDimension(self):
        return self.dimension
    '''
    Valor del Nodo.
    Entrada: Valor
    Salida: NA
    '''

    def setNodeValue(self, value):
        self.dimension = value
    '''
    Valor del Nodo.
    Entrada: Valor
    Salida: NA
    '''

    def getNodeValue(self):
        return self.dimension
    '''
    Insertar nodo derecho del arbol.
    Entrada: Arbol que representa el nuevo nodo.
    Salida: NA
    '''

    def insertRight(self, newNode):
        if self.right == None:
            self.right = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.right = self.right
            self.right = tree
    '''
    Insertar nodo izquierdo del arbol.
    Entrada: Arbol que representa el nuevo nodo.
    Salida: NA
    '''

    def insertLeft(self, newNode):
        if self.left == None:
            self.left = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.left = self.left
            self.left = tree




'''
Imprime el árbol.
Entrada: Arbol.
Salida: NA
'''


def printTree(tree):
    if tree != None:
        printTree(tree.getLeftChild())
        print(tree.getNodeValue())
        printTree(tree.getRightChild())

'''
Crea el modelo kd_tree.
Entrada: Conjunto de puntos, profundidad.
Salida: Arbol
'''


def kd_trees(list_points, depth):
    if not list_points:
        return None
    else:
        k = len(list_points[0])
        axis = depth % k
        list_points.sort(key=lambda x: x[axis])
        median = len(list_points) // 2

        node = BinaryTree()

        node.setNodeValue(list_points[median])
        node.left = kd_trees(list_points[0:median], depth + 1)
        node.right = kd_trees(list_points[median + 1:], depth + 1)
        return node
'''
Calcula distancia entre puntos.
Entrada: Punto 1 y Punto 2.
Salida: Calculo de la distancia.
'''


def distance(point1, point2):

    values = []
    for i in range(len(point1)):
        d_temp = point1[i] - point2[i]
        values += [d_temp * d_temp]

    return sqrt(sum(values))

'''
Obtiene el punto mas cercano.
Entrada: Conjunto de Putos y Punto a evaluar.
Salida: Punto mas cercano.
'''


def closest_point(all_points, new_point):
    best_point = None
    best_distance = None

    for current_point in all_points:
        current_distance = distance(new_point, current_point)

        if best_distance is None or current_distance < best_distance:
            best_distance = current_distance
            best_point = current_point

    return best_point

'''
Obtiene el punto mas cercano del arbol.
Entrada: Arbol, Punto a evaluar, profundidad y mejor punto.
Salida: Punto mas cercano.
'''


def kdtree_naive_closest_point(root, point, depth=0, best=None):
    if root is None:
        return best
    k = len(point)
    axis = depth % k

    next_best = None
    next_branch = None

    if best is None or distance(point, best) > distance(point, root.dimension):
        next_best = root.dimension
    else:
        next_best = best

    if point[axis] < root.dimension[axis]:
        next_branch = root.getLeftChild()
    else:
        next_branch = root.getRightChild()

    return kdtree_naive_closest_point(next_branch, point, depth + 1, next_best)

'''
Compara entre dos puntos cual es el mas cercano a un punto especifico.
Entrada: Punto base de comparacion, punto candidato 1 y punto candidato 2.
Salida: Punto mas cercano.
'''


def closer_distance(pivot, p1, p2):
    if p1 is None:
        return p2

    if p2 is None:
        return p1

    d1 = distance(pivot, p1)
    d2 = distance(pivot, p2)

    if d1 < d2:
        return p1
    else:
        return p2

kn_final = []
'''
Obtiene el punto mas cercano.
Entrada: Arbol, punto a evaluas, profundidad, conjunto de puntos.
Salida: Punto mas cercano.
'''


def kdtree_closest_point(root, point, depth=0, kn=[]):
    global kn_final
    kn_final = []
    if root is None:
        return None
    k = len(point)
    axis = depth % k

    next_branch = None
    opposite_branch = None

    if point[axis] < root.dimension[axis]:
        next_branch = root.getLeftChild()
        opposite_branch = root.getRightChild()
    else:
        next_branch = root.getRightChild()
        opposite_branch = root.getLeftChild()

    best = closer_distance(point,
                           kdtree_closest_point(next_branch,
                                                point,
                                                depth + 1, kn),
                           root.dimension)

    if distance(point, best) > abs(point[axis] - root.dimension[axis]):

        best = closer_distance(point,
                               kdtree_closest_point(opposite_branch,
                                                    point,
                                                    depth + 1, kn),
                               best)

    kn_final.append([best])

    return best

'''
Obtiene determinado numero de puntos mas cercanos
Entrada: Punto y numero de puntos cercanos.
Salida: Puntos mas cercano.
'''


def top_points(point, k):
    kn = copy.copy(kn_final)
    points = []
    distance_temp_list = []
    for i in kn:

        if (len(distance_temp_list) < k):
            distance_temp = distance(point, i[0])
            points += [i[0].tolist()]
            distance_temp_list += [distance_temp]
        else:
            if(max(distance_temp_list) > distance(point, i[0])):
                indx = distance_temp_list.index(max(distance_temp_list))
                distance_temp_list.pop(indx)
                points.pop(indx)
                distance_temp += distance(point, i[0])
                points += [i[0].tolist()]
                distance_temp_list += [distance_temp]

    return points

'''
Obtiene el valor que mas aparece de un conjunto de puntos.
Entrada: Conjunto de puntos
Salida: El valor que mas aparecio en los puntos.
'''


def best_vote_percent(neightboards):
    temp_list = list(set(copy.copy(neightboards)))
    votes = 0
    best = ""
    for i in temp_list:

        vote_temp = neightboards.count(i)
        if (vote_temp >= votes):
            votes = vote_temp
            best = i
    return best
