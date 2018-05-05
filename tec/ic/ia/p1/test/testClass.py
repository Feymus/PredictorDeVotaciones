# from main import *
import pytest
import unittest
import sys
sys.path.append("..")
from KDTree import *
from DecisionTree import *
from Normalizer import Normalizer
# from main import *


'''
Esta clase es la encargada de realizar lo que son las pruebas unitarias. Cada
función dentro de ella empieza con test_ y le sigue el nombres de
la función que se va a evaluar.
'''


class MyTest(unittest.TestCase):

    '''
    Prueba Unitario de la funcion getDImension
    Entrada: NA
    Salida: Assert
    '''

    def test_tree_getDimension(self):
        tree = kd_trees([[0, 1, 2], [3, 4, 5], [6, 7, 8]], 0)
        self.assertEqual(len(tree.getDimension()), 3)
    '''
    Prueba Unitaria de distancia
    Entrada: NA
    Salida: Assert
    '''

    def test_distance(self):
        normalizer = Normalizer()
        self.assertEqual(distance([0, 1, 2], [3, 4, 5]), 5.196152422706632)
    '''
    Prueba Unitario de la funcion closest_point
    Entrada: NA
    Salida: Assert
    '''

    def test_closest_point(self):
        self.assertEqual(
            closest_point([[0, 1, 2], [3, 4, 5], [6, 7, 8]], [3, 5, 5]), [3, 4, 5])

    '''
    Prueba Unitario de la funcion closer_distance
    Entrada: NA
    Salida: Assert
    '''

    def test_closer_distance(self):
        self.assertEqual(
            closer_distance([3, 5, 5], [0, 1, 2], [3, 4, 5]), [3, 4, 5])

    '''
    Prueba Unitario de la funcion kdtree_naive_closest_point
    Entrada: NA
    Salida: Assert
    '''

    def test_kdtree_naive_closest_point(self):
        tree = kd_trees([[0, 1, 2], [3, 4, 5], [6, 7, 8]], 0)
        self.assertEqual(kdtree_naive_closest_point(
            tree, [3, 5, 5], depth=0, best=None), [3, 4, 5])
    '''
    Prueba Unitario de la funcion best_vote_percent
    Entrada: NA
    Salida: Assert
    '''

    def test_best_vote_percent(self):
        self.assertEqual(
            best_vote_percent(["PAC", "RESTAURACION", "PAC"]), "PAC")
    '''
    Prueba Unitario de la funcion count_votes_dic
    Entrada: NA
    Salida: Assert
    '''

    def test_count_votes_dic(self):
        dic = {"PAC": 3, "RESTAURACION": 5, "LIBERACION": 7}
        self.assertEqual((count_votes_dic(dic)), 15)
    '''
    Prueba Unitario de la funcion count_votes_list
    Entrada: NA
    Salida: Assert
    '''

    def test_count_votes_list(self):
        test_list = [["holi", "PAC"], ["holi", "RESTAURACION"],
                     ["holi", "RESTAURACION"], ["holi", "RESTAURACION"]]
        self.assertEqual(
            count_votes_list(test_list), {"PAC": 1, "RESTAURACION": 3})
    '''
    Prueba Unitario de la funcion same_size
    Entrada: NA
    Salida: Assert
    '''

    def test_same_size(self):
        self.assertEqual(same_size([[3, 1, 2], [4, 5, 6]]), True)
    '''
    Prueba Unitario de la funcion get_examples_with_attribute
    Entrada: NA
    Salida: Assert
    '''

    def test_get_examples_with_attribute(self):
        examples = [["holi", "PAC"], ["hi", "RESTAURACION"],
                    ["holi", "RESTAURACION"], ["holi", "RESTAURACION"]]
        self.assertEqual(get_examples_with_attribute(
            examples, "hi", 0), [["RESTAURACION"]])
    '''
    Prueba Unitario de la funcion get_attribute
    Entrada: NA
    Salida: Assert
    '''

    def test_get_attribute(self):
        self.assertEqual(get_attribute([["PAC"], ["PAC"]]), ['PAC'])
    '''
    Prueba Unitario de la funcion test_importance
    Entrada: NA
    Salida: Assert
    '''

    def test_importance(self):
        values3 = [
            ["Full", "Thai", "no"], ["Full", "French", "no"], ["Some", "French", "yes"], [
                "Full", "Thai", "yes"], ["Full", "Italian", "no"], ["Some", "Burger", "yes"],
            ["None", "Burger", "no"], ["Some", "Italian", "yes"], ["Some", "Thai", "yes"], ["Full", "Burger", "no"], ["None", "Thai", "no"], ["Full", "Burger", "yes"]]

        self.assertEqual(importance(values3), [0, 0.5])
    '''
    Prueba Unitario de la funcion get_column_result
    Entrada: NA
    Salida: Assert
    '''

    def test_get_column_result(self):
        values3 = [
            ["Full", "Thai", "no"], ["Full", "French", "no"], ["Some", "French", "yes"], [
                "Full", "Thai", "yes"], ["Full", "Italian", "no"], ["Some", "Burger", "yes"],
            ["None", "Burger", "no"], ["Some", "Italian", "yes"], ["Some", "Thai", "yes"], ["Full", "Burger", "no"], ["None", "Thai", "no"], ["Full", "Burger", "yes"]]
        result = [
            ['Full', 'no'], ['Full', 'no'], ['Some', 'yes'], [
                'Full', 'yes'], ['Full', 'no'], ['Some', 'yes'],
            ['None', 'no'], ['Some', 'yes'], ['Some', 'yes'], ['Full', 'no'], ['None', 'no'], ['Full', 'yes']]

        self.assertEqual(get_column_result(values3, 0), result)
    '''
    Prueba Unitario de la funcion entropy
    Entrada: NA
    Salida: Assert
    '''

    def test_entropy(self):
        self.assertEqual(entropy([0.5, 0.5]), 1)
        self.assertEqual(entropy([0.99, 0.01]), 0.08079313589591118)
    '''
    Prueba Unitario de la funcion remainder
    Entrada: NA
    Salida: Assert
    '''

    def test_remainder(self):
        values3 = [
            ["Full", "Thai", "no"], ["Full", "French", "no"], ["Some", "French", "yes"], [
                "Full", "Thai", "yes"], ["Full", "Italian", "no"], ["Some", "Burger", "yes"],
            ["None", "Burger", "no"], ["Some", "Italian", "yes"], ["Some", "Thai", "yes"], ["Full", "Burger", "no"], ["None", "Thai", "no"], ["Full", "Burger", "yes"]]
        attributes = get_column_result(values3, 0)
        self.assertEqual(remainder(attributes), 0.5)
    '''
    Prueba Unitario de la funcion get_count
    Entrada: NA
    Salida: Assert
    '''

    def test_get_count(self):
        values3 = [
            ["Full", "Thai", "no"], ["Full", "French", "no"], ["Some", "French", "yes"], [
                "Full", "Thai", "yes"], ["Full", "Italian", "no"], ["Some", "Burger", "yes"],
            ["None", "Burger", "no"], ["Some", "Italian", "yes"], ["Some", "Thai", "yes"], ["Full", "Burger", "no"], ["None", "Thai", "no"], ["Full", "Burger", "yes"]]
        attributes = get_column_result(values3, 0)

        self.assertEqual(
            get_count(attributes), {'total': 12, 'no': 6, 'yes': 6})
    '''
    Prueba Unitario de la funcion get_prob
    Entrada: NA
    Salida: Assert
    '''

    def test_get_prob(self):
        values3 = [
            ["Full", "Thai", "no"], ["Full", "French", "no"], ["Some", "French", "yes"], [
                "Full", "Thai", "yes"], ["Full", "Italian", "no"], ["Some", "Burger", "yes"],
            ["None", "Burger", "no"], ["Some", "Italian", "yes"], ["Some", "Thai", "yes"], ["Full", "Burger", "no"], ["None", "Thai", "no"], ["Full", "Burger", "yes"]]
        attributes = get_column_result(values3, 0)

        self.assertEqual(get_prob(attributes), {'no': 0.5, 'yes': 0.5})
    '''
    Prueba Unitario de la funcion gain
    Entrada: NA
    Salida: Assert
    '''

    def test_gain(self):
        values3 = [
            ["Full", "Thai", "no"], ["Full", "French", "no"], ["Some", "French", "yes"], [
                "Full", "Thai", "yes"], ["Full", "Italian", "no"], ["Some", "Burger", "yes"],
            ["None", "Burger", "no"], ["Some", "Italian", "yes"], ["Some", "Thai", "yes"], ["Full", "Burger", "no"], ["None", "Thai", "no"], ["Full", "Burger", "yes"]]
        attributes = get_column_result(values3, 0)
        self.assertEqual(gain(attributes), 0.5)
    '''
    Prueba Unitario de la funcion classifications
    Entrada: NA
    Salida: Assert
    '''

    def test_classifications(self):
        values3 = [
            ["Full", "Thai", "no"], ["Full", "French", "no"], ["Some", "French", "yes"], [
                "Full", "Thai", "yes"], ["Full", "Italian", "no"], ["Some", "Burger", "yes"],
            ["None", "Burger", "no"], ["Some", "Italian", "yes"], ["Some", "Thai", "yes"], ["Full", "Burger", "no"], ["None", "Thai", "no"], ["Full", "Burger", "yes"]]
        examples = get_column_result(values3, 0)
        self.assertEqual(classifications(examples), ['no', 'yes'])
    '''
    Prueba Unitario de la funcion delete_duplicates
    Entrada: NA
    Salida: Assert
    '''

    def test_delete_duplicates(self):
        self.assertEqual(delete_duplicates(["PAC", "PAC"]), ["PAC"])
    '''
    Prueba Unitario de la funcion plurality_values
    Entrada: NA
    Salida: Assert
    '''

    def test_plurality_values(self):
        values3 = [
            ["Full", "Thai", "no"], ["Full", "French", "no"], ["Some", "French", "no"], [
                "Full", "Thai", "yes"], ["Full", "Italian", "no"], ["Some", "Burger", "yes"],
            ["None", "Burger", "no"], ["Some", "Italian", "yes"], ["Some", "Thai", "yes"], ["Full", "Burger", "no"], ["None", "Thai", "no"], ["Full", "Burger", "yes"]]
        self.assertEqual(plurality_values(values3), "no")

if __name__ == "__main__":
    unittest.main()
