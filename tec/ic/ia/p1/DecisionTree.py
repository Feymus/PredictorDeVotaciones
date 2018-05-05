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
import numpy as np
import pandas as pd
import scipy.stats as stats
import sys
import operator


class DecisionTree():

    '''
    Método que trae por default python que inicializa la clase.
    '''

    def __init__(self, threshold):
        self.tree = None  # Guarda el Árbol Generado
        self.attr = None  # Contiene la lista de atributos evaluados
        self.threshold = threshold

    '''
    Entrena el modelo con los conjuntos de datos dados.
    Entrada: una lista con las muestras de entrenamiento.
    Salida: NA
    '''

    def train(self, samples):
        print("######################################################")
        print(samples)

        array_samples = samples["trainingFeatures"].tolist()
        prunning_tests = samples["testingFeatures"].tolist()

        data = []  # Contiene los posibles resultados
        for i in array_samples:
            data += [i[len(i) - 1]]
            data = list(set(data))

        attr = []  # Contiene los atributos evaluados
        for i in range(len(array_samples[0]) - 1):
            attr += ["attr" + str(i)]

        self.attr = attr

        self.tree = desition_tree(array_samples, attr, data)
        for i in prunning_tests:
            result = self.tree.test(i, attr, i[len(i) - 1])
        self.tree.pruning_chi(self.threshold, attr, data)

    '''
    Predice el valor según una muestra.
    Entrada: una lista con los atributos y el valor esperado.
    Salida: NA
    '''

    def classify(self, test):
        # test_sample=test
        test_sample = test[0].tolist()
        print(test_sample)
        # print(self.tree.test(test_sample, self.attr,
        # test_sample[len(test_sample) - 1]))
        return self.tree.test(
            test_sample, self.attr, test_sample[len(test_sample) - 1])


class Tree(object):

    '''
    Método que trae por default python que inicializa la clase.
    Entrada: el nombre de la raíz.
    Salida: NA

    '''

    def __init__(self, name='root', children=None):

        self.name = name
        self.children = []
        self.gain = 0
        self.votes = None
        self.votes_test_result = None
        if children is not None:
            for child in children:
                self.add_child(child)
    '''
    Método de python que muestra como se va a representar la clase a la hora
    de imprimir.
    Entrada: NA
    Salida: String con Nombre
    '''

    def __repr__(self):
        return str(self.name)
    '''
    Método de python que muestra como se va a representar la clase a la hora
    de imprimir.
    Entrada: NA.
    Salida: String con Nombre y Atributos.
    '''

    def __str__(self):
        return str(self.__class__) + ": \n" + str(self.__dict__)

    '''
    Método que agrega un hijo al árbol.
    Entrada: El nodo hijo.
    Salida: NA

    '''

    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)
    '''
    Método que se da la ganancia.
    Entrada: Número de ganacia.
    Salida: NA

    '''

    def set_gain(self, gain):
        self.gain = gain
    '''
    Método que obtiene el atributo name.
    Entrada: NA
    Salida: String Nombre
    '''

    def get_name(self):
        return self.name

    '''
    Método que obtiene los hijos.
    Entrada: NA
    Salida: Conjunto de Hijos
    '''

    def get_children(self):
        return self.children
    '''
    Método que da el total de hijo que tiene el árbol.
    Entrada: NA
    Salida: Entero con total de números de hijos.
    '''

    def num_children(self):
        return len(self.children)
    '''
    Método que obtiene un hijo que tiene el árbol.
    Entrada: NA
    Salida: Nodos Hijos.
    '''

    def get_child(self, name):
        for i in self.children:
            if(name == i.get_name()):
                return i
        return None
    '''
    Método que elimina un hijo.
    Entrada: Hijo a eliminar.
    Salida: NA.
    '''

    def delete_child(self, child):
        self.children.remove(child)
    '''
    Método que da las hojas.
    Entrada: NA
    Salida: Entero con total de números de hijos.
    '''

    def get_leafs(self):
        votes = []
        if(self.votes is not None):
            for i in self.votes.keys():

                votes += [[i]]

        return Tree(votes, [])
    '''
    Imprime el Árbol.
    Entrada: NA
    Salida: Entero con total de números de hijos.
    '''

    def printTree(self):

        if(self.children != []):
            print(
                "------------------------------------------------------------")

            print(self)
            for i in self.children:
                i.printTree()
        else:
            print(self)
            print(
                "------------------------------------------------------------")

    '''
    Prueba el modelo.
    Entrada: Lista
    Salida: Entero con total de números de hijos.
    '''

    def test(self, test_list, attributes, expected):
        dic = {}
        if(self.name in attributes):
            position_attr = attributes.index(self.name)
            attr = test_list[position_attr]
            child = self.get_child(attr)
            valor = ['NULO']
            if(child is not None):
                for i in child.children:
                    valor = i.test(test_list, attributes, expected)
                    if(i.children == []):
                        self.add_votes_test_result(i.name)
                        i.add_votes_test_result(i.name)
                        child.add_votes_test_result(i.name)
                        return i.name
                child.add_votes_test_result(valor)
                self.add_votes_test_result(valor)
                return valor

            else:
                self.add_votes_test_result(['NULO'])
                return ['NULO']
        else:
            name = self.name
            self.add_votes_test_result(name)
            return self.name
    '''
    Predice el valor según una muestra.
    Entrada: una lista con los atributos y el valor esperado.
    Salida: NA
    '''

    def add_votes_test_result(self, name):
        if(isinstance(name, list)):
            name = name[0]
            if(isinstance(name, list)):
                name = name[0]
        if(self.votes_test_result is not None):
            count_values = self.votes_test_result.get(name)
            if(count_values is not None):
                self.votes_test_result[name] = count_values + 1
            else:
                self.votes_test_result[name] = 1
        else:
            self.votes_test_result = {}
            self.votes_test_result[name] = 1

    '''
    Calcula chi cuadrado.
    Entrada: un diccionario con los datos del nodo.
    Salida: Valor de Chi cuadrado.
    '''

    def chi_square(self, data):
        desv = 0
        for i in self.children:
            for j in data:
                n_k = 0
                if(i.votes_test_result is not None):
                    if(i.votes_test_result.get(j) is not None and
                            self.votes_test_result.get(j) is not None):
                        p = self.votes_test_result.get(j)
                        n = count_votes_dic(self.votes_test_result) - p
                        pk = i.votes_test_result.get(j)
                        nk = count_votes_dic(i.votes_test_result) - pk
                        pl = (pk + nk) / (p + n)
                        n_k = p * pl
                        desv += (n_k - pk)**2 / n_k

        return desv
    '''
    Método que poda el arbol con chi.
    Entrada: threshold es el umbral de poda, conjunto de atributos, data
    conjunto de posibles valores de salida.
    Salida: el chi de la poda.
    '''

    def pruning_chi(self, threshold, attr, data):
        if (self.children == []):
            return self.chi_square(data)
        else:
            for i in self.children:
                n = i.pruning_chi(threshold, attr, data)
                p = self.chi_square(data)
                if(n < threshold and p < threshold):
                    dic = self.votes
                    max_value = max(dic.values())
                    for key in dic.keys():
                        if(dic[key] == max_value):
                            max_key = key
                            break
                    self.children = [Tree([max_key], [])]
                    break
            return p


'''
Método que cuenta los votos de un diccionario.
Entrada: una lista con los atributos y el valor esperado.
Salida: NA
'''


def count_votes_dic(votes):
    val = 0
    for i in votes.keys():
        val += votes.get(i)
    return val


'''
Crea el modelo árbol de decisión.
Entrada: conjunto de atributos,
Salida: NA
'''


def decision_tree_learning(examples, attributes, parent_examples):
    # print(examples)
    # print(attributes)
    # print(parent_examples)

    if not examples:

        return plurality_values(parent_examples)
    elif same_classification(examples):

        return classifications(examples)
    elif not attributes:

        return plurality_values(examples)
    else:

        importance_value, max_gain = importance(examples)
        exs_attribute = get_column_result(examples, importance_value)
        A = get_attribute(exs_attribute)
        tree = Tree(attributes[importance_value], [])
        tree.set_gain(max_gain)
        tree.votes = count_votes_list(examples)
        for v in A:

            temp_attr = copy.copy(attributes)
            temp_attr.pop(importance_value)
            exs = get_examples_with_attribute(examples, v, importance_value)

            if(len(temp_attr) + 1 == len(exs[0])):

                subtree = decision_tree_learning(exs, temp_attr, examples)

                if(isinstance(subtree, Tree)):

                    temp_tree = Tree(v, [subtree])
                else:

                    temp_tree = Tree(v, [Tree(subtree, [])])
                temp_tree.votes = count_votes_list(exs)
                tree.add_child(temp_tree)
        return tree
    return val


'''
Cuenta.
Entrada: conjunto de atributos,
Salida: NA
'''


def count_votes_list(examples):
    temp_list = []
    for i in examples:
        temp_list += [i[len(i) - 1]]
    dic = {}
    for i in list(set(temp_list)):

        dic.setdefault(i, temp_list.count(i))

    return dic


'''
Revisa si una lista de listas tienen el mismo tamaño.
Entrada: conjunto de muestras.
Salida: Booleano que dice si si o no tienen el mismo tamaño.
'''


def same_size(examples):
    size = len(examples[0])
    for i in examples:
        if(len(i) != size):
            return False
    return True


'''
Revisa si una lista de listas tienen el mismo tamaño.
Entrada: conjunto de muestras.
Salida: Booleano que dice si si o no tienen el mismo tamaño.
'''


def get_examples_with_attribute(examples, value, column):

    exs = []
    examples_temp = copy.copy(examples)

    for i in examples_temp:

        if (i[column] == value):
            j = copy.copy(i)
            j.pop(column)

            exs += [j]

    return exs


'''
Obtiene el conjunto de atributos sin repetir.
Entrada: conjunto de resultados.
Salida: Valores posibles de salidas de resultados.
'''


def get_attribute(results):
    attributes = []
    for i in results:

        attributes.append(i[0])
    return list(set(attributes))


'''
Obtiene el atributo con mayor ganancia.
Entrada: conjunto de muestras.
Salida: tupla con la columna de mayor ganacia y la cantidad de ganacia.
'''


def importance(examples):

    if (examples != []):
        max_gain = 0
        column = 0

        for i in range(len(examples[0]) - 2):
            attribute = get_column_result(examples, i)
            gain_temp = gain(attribute)
            if(max_gain < gain_temp):
                max_gain = gain_temp
                column = i

        return [column, max_gain]


'''
Obtiene la columna de resultado.
Entrada: conjunto de muestras.
Salida: tupla con la columna de mayor ganacia y la cantidad de ganacia.
'''


def get_column_result(examples, column):
    list_values = []
    for i in examples:
        value = [i[column]] + [i[len(i) - 1]]
        list_values += [value]

    return list_values


'''
Calculo de la entropia.
Entrada: conjunto de porcentajes.
Salida: valor de la entropia.
'''


def entropy(q):
    sum_q = 0
    for i in q:
        sum_q += i * log2(i)
    return sum_q * -1


'''
Calcula la sumatoria de la ganacia de los hijos.
Entrada: atributo al cual se le va a sacar el remainder.
Salida: sumatoria de la entropia.
'''


def remainder(attribute):
    dic = {'total': len(attribute)}
    for i in attribute:
        if (dic.get(i[0]) is None):

            dic_temp = {i[len(i) - 1]: 1}

            dic.setdefault(i[0], dic_temp)
        else:

            temp_dic = dic.get(i[0])

            temp_value = temp_dic.get(i[len(i) - 1])
            if (temp_value is None):
                dic_temp = {i[len(i) - 1]: 1}

            else:
                dic_temp = {i[len(i) - 1]: temp_value + 1}
            temp_dic.update(dic_temp)

    sum_entropy = 0

    for key in list(dic.keys())[1:]:

        value_temp = sum(list(dic.get(key).values())) / dic.get('total')

        entropy_child = value_temp * \
            entropy(list(get_prob(dic.get(key)).values()))
        sum_entropy += entropy_child

    return sum_entropy


'''
Contar la cantidad de votos por atributo.
Entrada: Atributo con el resultado.
Salida: Diccionario con la cantidad de votos.
'''


def get_count(attribute):
    dic = {'total': len(attribute)}
    for i in attribute:

        if (dic.get(i[len(i) - 1]) is None):

            dic.setdefault(i[len(i) - 1], 1)
        else:

            temp_value = dic.get(i[len(i) - 1])
            dic_temp = {i[len(i) - 1]: temp_value + 1}
            dic.update(dic_temp)
    return dic


'''
Obtiene la probabilidad de un atributo por resultado.
Entrada: Atributo con el resultado. .
Salida: probabilidad de resultado en un diccionario .
'''


def get_prob(attribute):
    dic_count = get_count(attribute)
    total_values = dic_count.get('total')
    dic_temp = {}

    for key in list(dic_count.keys())[1:]:

        value = dic_count.get(key)
        prob = value / total_values
        dic_temp.setdefault(key, prob)
    return dic_temp


'''
Obtiene la ganacia de un atributo especifico.
Entrada: Atributo con el resultado. .
Salida: .
'''


def gain(attribute):

    return entropy(list(get_prob(attribute).values())) - remainder(attribute)


'''
Revisa si un conjunto de muestras tiene la misma clasificacion.
Entrada: Conjunto de muestras .
Salida: Un booleano que dice si si o no tienen el mismo resultado.
'''


def same_classification(examples):
    if(examples == []):
        return False
    else:
        votes = classifications(examples)
        if(votes != []):
            if(len(votes) == votes.count(votes[0])):
                return True
            else:
                return False


'''
Obtiene la clasificacion de un conjuntode muestras.
Entrada: Conjunto de muestras.
Salida: Lista con los resultado de las muestras.
'''


def classifications(examples):
    list_values = []

    for i in examples:
        size = len(i)
        list_values += [i[size - 1]]

    # print(delete_duplicates(list_values))
    return delete_duplicates(list_values)


'''
Elimina duplicados de una lista.
Entrada: Conjunto de valores en lista.
Salida: Lista sin repetidos.
'''


def delete_duplicates(values):
    set_values = set(values)

    result = list(set_values)
    return result


'''
Obtiene la clasificacion con mayor presencia.
Entrada: Conjunto de muestras.
Salida: String con la mayor clasificación.
'''


def plurality_values(examples):
    votes = classifications(examples)

    classif = delete_duplicates(votes)
    ties = []
    value_max = 0
    classif_max = ""
    for i in classif:
        temp = votes.count(i)
        if(temp > value_max):
            value_max = temp
            classif_max = i
            if(ties != []):
                ties = []
        if(temp == value_max):
            ties.append(classif_max)
    if(ties == []):

        return classif_max
    else:
        return random.choice(ties)


'''
Crea un arbol de decision
Entrada: Conjunto de muestras, Conjunto de Atributos, Conjunto de salidas de
resultados.
Salida: Arbol de decision.
'''


def desition_tree(samples, attr, data):

    tree_test = decision_tree_learning(samples, attr, [])
    return (tree_test)
