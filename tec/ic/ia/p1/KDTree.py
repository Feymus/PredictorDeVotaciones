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

    def __init__(self, neightboards):
        self.tree = None
        self.samples = None
        self.neightboards = neightboards

    def train(self, samples):
        self.samples = samples
        self.tree = kd_trees(list(samples), 0)

    def classify(self, test):
        mini_test = kdtree_closest_point(self.tree, test, 0, []).tolist()
        neightboards = []
        for j in top_points(test, self.neightboards):
            pos = self.samples.tolist().index(j)
            o = r1_results.tolist()[pos]
            neightboards += [o]

        return best_vote_percent(neightboards)


class BinaryTree():

    def __init__(self):
        self.left = None
        self.right = None
        self.dimension = None

    def getLeftChild(self):
        return self.left

    def getRightChild(self):
        return self.right

    def getDimension(self):
        return self.dimension

    def setNodeValue(self, value):
        self.dimension = value

    def getNodeValue(self):
        return self.dimension

    def insertRight(self, newNode):
        if self.right == None:
            self.right = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.right = self.right
            self.right = tree

    def insertLeft(self, newNode):
        if self.left == None:
            self.left = BinaryTree(newNode)
        else:
            tree = BinaryTree(newNode)
            tree.left = self.left
            self.left = tree


def printTree(tree):
    if tree != None:
        printTree(tree.getLeftChild())
        print(tree.getNodeValue())
        printTree(tree.getRightChild())


def knn(point, node):
    return


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


def distance(point1, point2):

    values = []
    for i in range(len(point1)):
        d_temp = point1[i] - point2[i]
        values += [d_temp * d_temp]

    return sqrt(sum(values))


def closest_point(all_points, new_point):
    best_point = None
    best_distance = None

    for current_point in all_points:
        current_distance = distance(new_point, current_point)

        if best_distance is None or current_distance < best_distance:
            best_distance = current_distance
            best_point = current_point

    return best_point


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


def kdtree_closest_point(root, point, depth=0, kn=[]):

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


values3 = [
    ["Full", "Thai", "no"], ["Full", "French", "no"], ["Some", "French", "yes"], [
        "Full", "Thai", "yes"], ["Full", "Italian", "no"], ["Some", "Burger", "yes"],
    ["None", "Burger", "no"], ["Some", "Italian", "yes"], ["Some", "Thai", "yes"], ["Full", "Burger", "no"], ["None", "Thai", "no"], ["Full", "Burger", "yes"]]

lenData = 600
samples = generar_muestra_pais(lenData)

normalizer = Normalizer()
samples_normalizar = normalizer.prepare_data(samples, 0.2)

r1 = samples_normalizar.get('trainingFeatures')
r1_tests = samples_normalizar.get('testingFeatures')
r1_results = samples_normalizar.get('trainingClassesFirst')
r1_tests_results = samples_normalizar.get('testingClassesFirst')
tree = Kd_Tree(15)
tree.train(r1_tests)
#print(tree.classify(r1_tests[0]))
'''

r1_tree=kd_trees(list(r1),0)
fail=0
win=0
for i in r1_tests.tolist():
    print(i)
    mini_test=kdtree_closest_point(r1_tree,i,0,[]).tolist()


    pos_test=r1_tests.tolist().index(i)
    e=r1_tests_results.tolist()[pos_test]
    neightboards=[]
    print(r1)
    for j in top_points(i,15):

        pos=r1.tolist().index(j)
        o=r1_results.tolist()[pos]

        neightboards+=[o]


    kn_final=[]
    if (best_vote_percent(neightboards)==e):
        win+=1
    else:
        fail+=1






print(win)
print(fail)
print((len(r1_tests)-fail)/len(r1_tests)*100)

'''
'''
r2=samples_normalizar.get('trainingFeaturesFirstInclude')
r2_tests=samples_normalizar.get('testingFeaturesFirstInclude')
r2_results=samples_normalizar.get('trainingClassesSecond')
r2_tests_results=samples_normalizar.get('testingClassesSecond')


r2_tree=kd_trees(list(r2),0)
fail=0
win=0

for i in r2_tests.tolist():

    mini_test=kdtree_closest_point(r2_tree,i,0,[]).tolist()


    pos_test=r2_tests.tolist().index(i)
    e=r2_tests_results.tolist()[pos_test]
    neightboards=[]
    for j in top_points(i,15):
        pos=r2.tolist().index(j)
        o=r2_results.tolist()[pos]

        neightboards+=[o]


    kn_final=[]
    if (best_vote_percent(neightboards)==e):
        win+=1
    else:
        fail+=1

print(win)
print(fail)
print((len(r2_tests)-fail)/len(r2_tests)*100)
'''
