import pytest
import unittest
import sys
sys.path.append("..")
from g06 import *

'''
Esta clase es la encargada de realizar lo que son las pruebas unitarias. Cada 
función dentro de ella empieza con test_ y le sigue el nombres de 
la función que se va a evaluar. 
'''


class MyTest(unittest.TestCase):
     cargar_csv()
     def test_random_pick(self):
       	self.assertEqual(random_pick( ["URBANO", "RURAL"],[1, 0]),"URBANO")
     def test_search_province_by_canton(self):
       	self.assertEqual(search_province_by_canton("ACOSTA"),"SAN JOSE")
     def test_is_dependent(self):
       	self.assertEqual(is_dependent("18-24"),"PRODUCTIVO")
     def test_quantity_by_province(self):
       	self.assertEqual(votes_quantity_by_province("SAN JOSE"),762420)
     def test_generated_vote(self):
       	assert(generated_vote("SAN JOSE", "DESAMPARADOS") in [
        "ACCESIBILIDAD SIN EXCLUSION", "ACCION CIUDADANA",
        "ALIANZA DEMOCRATA CRISTIANA", "DE LOS TRABAJADORES", "FRENTE AMPLIO",
        "INTEGRACION NACIONAL", "LIBERACION NACIONAL", "MOVIMIENTO LIBERTARIO",
        "NUEVA GENERACION", "RENOVACION COSTARRICENSE",
        "REPUBLICANO SOCIAL CRISTIANO", "RESTAURACION NACIONAL",
        "UNIDAD SOCIAL CRISTIANA", "NULO", "BLANCO"
    ])
     def test_is_literate_pct(self):
       	self.assertEqual(is_literate_pct("18-24", "SAN JOSE", "TIBAS"),0.9954247951)
     def test_regular_edu_pct(self):
       	self.assertEqual(regular_edu_pct("18-24", "SAN JOSE", "TIBAS"),0.5274672437)
     def test_get_work_pct(self):
       	self.assertEqual(get_work_pct("MUJER", "SAN JOSE", "SAN JOSE"),0.44168248990000003)
     def test_get_insured_pct(self):
       	self.assertEqual(get_insured_pct("NO", "CARTAGO", "CARTAGO"),0.8607689083)

if __name__ == "__main__":
    unittest.main()
