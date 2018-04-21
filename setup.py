from setuptools import setup, find_packages

setup(name='tec.ic.ia.pc1.g06',
      version='0.3.3',
      description='Modulo para la generación de muestras',
      url='https://github.com/Xbolanos/PC1-IA/',
      author='Grupo6-IA-ITCR',
      py_modules=['tec.ic.ia.pc1.g06'],
      packages=['tec', 'tec.ic', 'tec.ic.ia', 'tec.ic.ia.pc1'],
      package_data={
        'tec.ic.ia.pc1': ['minutes.csv', 'properties.csv']
      })
