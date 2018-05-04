# from main import *
import pytest
import unittest
import sys
sys.path.append("..")
from KDTree import *
from main import *


'''
Esta clase es la encargada de realizar lo que son las pruebas unitarias. Cada
función dentro de ella empieza con test_ y le sigue el nombres de
la función que se va a evaluar.
'''

# distancias
# closest_point
# closer_distance
# best_vote_percent

class MyTest(unittest.TestCase):

    tree = kd_trees([[1,2,3], [4,5,6]], 0)
    printTree(tree)


if __name__ == "__main__":
    unittest.main()
