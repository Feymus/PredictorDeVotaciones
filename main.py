import sys, getopt

from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)


def main(argv):
   poblacion = 0
   porcentaje = 0
   prueba = 0
   try:
      opts, args = getopt.getopt(argv,"hi:o:",["poblacion=","porcentaje-pruebas="])
   except getopt.GetoptError:
      print ('main.py --poblacion <cantidad> --porcentaje-prueba <porcentaje, prueba>')
      sys.exit(2)
   for opt, arg in opts:
      if opt == '-h':
         print ('main.py --poblacion <cantidad> --porcentaje-prueba <porcentaje, prueba>')
         sys.exit()
      elif opt in ("--poblacion"):
         poblacion = arg
        
      elif opt in ("--porcentaje-pruebas"):
      		porcentaje =arg
   print ('Poblacion', poblacion)
   print ('Porcentaje', porcentaje)
   print(generar_muestra_pais(5))

if __name__ == '__main__':
    main(sys.argv[1:])
