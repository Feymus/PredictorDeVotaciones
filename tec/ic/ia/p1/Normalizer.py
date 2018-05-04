from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
import copy
import numpy as np
#import tensorflow as tf


class Normalizer(object):

    '''Convertidor y normalizador de datos para los clasificadores.'''

    def __init__(self, norm="l2"):
        super(Normalizer, self).__init__()
        self.featureNames = [
            "Provincia", "Canton", "Total de la población", "Superficie",
            "Densidad de la población", "Urbano/Rural", "Género", "Edad",
            "Dependencia", "Alfabeta", "Escolaridad promedio",
            "Escolaridad regular", "Trabaja", "Asegurado",
            "Cant. casas individuales", "Ocupantes promedio", "Condicion",
            "Hacinada", "Nacido en...", "Discapacitado", "Jefatura femenina",
            "Jefatura compartida", "Voto ronda 1", "Voto ronda 2"
        ]
        self.converter = DictVectorizer(sparse=False)
        self.norm = norm
        self.normalData = {}
        self.convertedData = {}
        self.tensorColumns = []

    def create_columns_tensor(self, samples):
        tf.feature_column.categorical_column_with_hash_bucket(
            'province', hash_bucket_size=100)
        tf.feature_column.numeric_column('numeric')

        featureNum = 0
        for feature in samples[0]:

            try:
                feature = float(feature)
            except ValueError:
                # La propiedad es un string
                pass

            featureNum = 0
            for feature in featureList:
                try:
                    feature = float(feature)
                except ValueError:
                    # La propiedad es un string
                    pass
                dictFeatures[self.featureNames[featureNum]] = feature
                featureNum += 1

            features.append(dictFeatures)

        return features


    def prepare_data_tensor(self, samples, pct_test):
        data = self.separate_data(samples, pct_test)

        for key in data:
            if "Classes" not in key:
                data[key] = self.create_columns_tensor(data[key])

    '''
    Retorna los datos de las muestras pasadas por parametro en un diccionario
    de la forma:
        {
            "trainingFeatures": <Datos de entrenamiento>,
            "testingFeatures": <Datos de testing>,
            "trainingFeaturesFirstInclude": <Datos de entrenamiento
                                                con primer voto>,
            "testingFeaturesFirstInclude": <Datos de testin
                                                con primer voto>,
            "trainingClassesFirst": <Resultados de primera ronda de los datos
                                        de entrenamiento>,
            "trainingClassesSecond": <Resultados de segunda ronda de los datos
                                        de entrenamiento>,
            "testingClassesFirst": <Resultados de primera ronda de los datos
                                        de pruebas>,
            "testingClassesSecond": <Resultados de segunda ronda de los datos
                                        de pruebas>
        }
    Entrada: Datos generados por el generador de muestras, porcentaje que se
    usara para pruebas.
    Salida: Los datos en forma de diccionario
    '''
    def prepare_data(self, samples, pct_test):
        data = self.separate_data(samples, pct_test)
        # Los datos se transforman a solo numeros
        self.convert_data(data)
        return data

    '''
    Convierte todos los datos a valores entre 0 y 1
    Entrada: datos ya transformados a numeros
    Salida: los nuevos datos se guardan en el mismo diccionario
    '''
    def normalize_data(self, data):
        data["trainingFeatures"] = normalize(
            data["trainingFeatures"],
            norm=self.norm,
            copy=False
        )
        data["testingFeatures"] = normalize(
            data["testingFeatures"],
            norm=self.norm,
            copy=False
        )
        data["trainingFeaturesFirstInclude"] = normalize(
            data["trainingFeaturesFirstInclude"],
            norm=self.norm,
            copy=False
        )
        data["testingFeaturesFirstInclude"] = normalize(
            data["testingFeaturesFirstInclude"],
            norm=self.norm,
            copy=False
        )

    '''
    Convierte los datos en un diccionario segun los indicadores (nombre de las
    propiedades)
    Entrada: datos a transformar de la forma:
        [["Genero", "Canton",...],...]
    Salida: los datos en forma de una lista de diccionarios
    '''
    def convert_to_dict_list(self, samples):
        features = []

        for featureList in samples:

            dictFeatures = {}
            featureNum = 0
            for feature in featureList:
                try:
                    feature = float(feature)
                except ValueError:
                    # La propiedad es un string
                    pass
                dictFeatures[self.featureNames[featureNum]] = feature
                featureNum += 1

            features.append(dictFeatures)

        return features

    def convert_to_list(self, dict):
        list = []

        for key in dict:
            list.append(dict[key])

        return list

    '''
    Convierte los datos en numericos
    Entrada: lista de datos en forma de diccionario
    Salida: guarda los datos en el mismo diccionario de entrada
    '''
    def convert_data(self, data):

        for key in data:
            if "Classes" not in key:
                data[key] = self.convert_to_dict_list(data[key])

        self.converter.fit(
            np.append(
                data["trainingFeatures"],
                data["testingFeatures"],
                axis=0
            )
        )

        data["trainingFeatures"] = self.converter.transform(
            data["trainingFeatures"]
        )
        data["testingFeatures"] = self.converter.transform(
            data["testingFeatures"]
        )

        self.converter.fit(
            np.append(
                data["trainingFeaturesFirstInclude"],
                data["testingFeaturesFirstInclude"],
                axis=0
            )
        )

        data["trainingFeaturesFirstInclude"] = self.converter.transform(
            data["trainingFeaturesFirstInclude"]
        )
        data["testingFeaturesFirstInclude"] = self.converter.transform(
            data["testingFeaturesFirstInclude"]
        )

        self.convertedData = copy.deepcopy(data)

    '''
    Separa los datos en datos de entrenamiento y de pruebas
    Entrada: Los datos generados por el generador, el procentaje a usar para
    pruebas.
    Salida: Un diccionario con los datos separados
    '''
    def separate_data(self, samples, pct_test):

        samplesArray = np.array(samples)
        X = samplesArray[:, :22]
        y = samplesArray[:, 22:]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=pct_test,
            random_state=42
        )

        y_train_first_round = y_train[:, 0]
        y_train_second_round = y_train[:, 1]

        y_test_first_round = y_test[:, 0]
        y_test_second_round = y_test[:, 1]

        X_train_2 = np.append(X_train, y_train[:, :1], axis=1)
        X_test_2 = np.append(X_test, y_test[:, :1], axis=1)

        self.normalData = {
            "trainingFeatures": X_train,
            "testingFeatures": X_test,
            "trainingFeaturesFirstInclude": X_train_2,
            "testingFeaturesFirstInclude": X_test_2,
            "trainingClassesFirst": y_train_first_round,
            "trainingClassesSecond": y_train_second_round,
            "testingClassesFirst": y_test_first_round,
            "testingClassesSecond": y_test_second_round
        }

        return {
            "trainingFeatures": X_train,
            "testingFeatures": X_test,
            "trainingFeaturesFirstInclude": X_train_2,
            "testingFeaturesFirstInclude": X_test_2,
            "trainingClassesFirst": y_train_first_round,
            "trainingClassesSecond": y_train_second_round,
            "testingClassesFirst": y_test_first_round,
            "testingClassesSecond": y_test_second_round
        }

    def separate_data_2(self, samples, pct_test):

        samplesArray = np.array(samples)
        X = samplesArray
        y = samplesArray[:, 22:]

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=pct_test,
            random_state=42
        )

        X_train_2 = np.delete(X_train, [22], axis=1)
        X_test_2 = np.delete(X_test, [22], axis=1)

        X_train_3 = X_train
        X_test_3 = X_test

        X_train = np.delete(X_train, [23], axis=1)
        X_test = np.delete(X_test, [23], axis=1)

        y_train_first_round = y_train[:, 0]
        y_train_second_round = y_train[:, 1]

        y_test_first_round = y_test[:, 0]
        y_test_second_round = y_test[:, 1]

        self.normalData = {
            "trainingFeatures": X_train,
            "testingFeatures": X_test,
            "trainingFeaturesSecond": X_train_2,
            "testingFeaturesSecond": X_test_2,
            "trainingFeaturesFirstInclude": X_train_3,
            "testingFeaturesFirstInclude": X_test_3,
            "trainingClassesFirst": y_train_first_round,
            "trainingClassesSecond": y_train_second_round,
            "testingClassesFirst": y_test_first_round,
            "testingClassesSecond": y_test_second_round
        }

        return {
            "trainingFeaturesFirst": X_train,
            "testingFeaturesFirst": X_test,
            "trainingFeaturesSecond": X_train_2,
            "testingFeaturesSecond": X_test_2,
            "trainingFeaturesFirstInclude": X_train_3,
            "testingFeaturesFirstInclude": X_test_3,
            "trainingClassesFirst": y_train_first_round,
            "trainingClassesSecond": y_train_second_round,
            "testingClassesFirst": y_test_first_round,
            "testingClassesSecond": y_test_second_round
        }

    def get_normal_data(self):
        return self.normalData

    def get_converted_data(self):
        return self.convertedData

    '''
    Compara el tamaño de dos listas y retorna el número del tamaño de la lista
    más grande
    Entrada: Recibe dos listas
    Salida: Tamaño de la lista más grande
    '''
    def bigger_size(self, list1, list2):
        if len(list1)>=len(list2):
            return len(list1)
        else:
            return len(list2)


    '''
    Crea una lista de ceros del tamaño deseado
    Entrada: Tamaño de la lista a crear
    Salida: Lista con ceros del tamaño ingresado
    '''
    def extra_list(self, num):
        temp = []
        for i in range (num):
            temp+=[0]
        return temp
