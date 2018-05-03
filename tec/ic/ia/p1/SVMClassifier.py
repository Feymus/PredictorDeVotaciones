from sklearn import svm
from sklearn.model_selection import GridSearchCV
import numpy as np
import time


class SVMClassifier(object):

    '''Un clasificador de modelos en SVM.'''

    def __init__(self, pKernel, pC, pGamma):
        super(SVMClassifier, self).__init__()
        self.classifier = svm.SVC(kernel=pKernel, C=pC, gamma=pGamma)
        self.trainingTime = 0

    def test_parameters(self, X, y):

        parameters = {
            'kernel': ('sigmoid', 'rbf', 'linear'),
            'C': [1, 5, 10],
            'gamma': [
                1, 0.0001, 0.0000001, 0.000000001
            ]
        }

        svc = svm.SVC()
        self.classifier = GridSearchCV(svc, parameters)

        self.classifier.fit(X, y)
        print(self.classifier.cv_results_)

    '''
    Entrena el modelo con los conjuntos de datos dados
    Entrada: un diccionario con los datos de entrenamiento y sus
    correspondientes salidas.
    Salida: NA
    '''
    def train(self, data):
        # time1 = time.time()
        self.classifier.fit(
            data["trainingFeatures"],
            data["trainingClasses"]
        )
        # time2 = time.time()
        # seconds = ((time2-time1)*1000.0)*1000.0
        # self.setTrainingTime(seconds)

    '''
    Retorna la clasificación a la que pertenece la entrada
    Entrada: propiedades del ejemplo a clasificar de la forma:
            [[1,2,3,4,5,...]] : donde cada numero es una propiedad
    Salida: string de la clasificacion del ejemplo.
    '''
    def classify(self, features):
        prediction = self.classifier.predict(features)
        return prediction

    '''
    Setea el tiempo que duro el modelo en entrenarse
    Entrada: float con el valor en segundos de cuanto duro
    Salida: NA
    '''
    def setTrainingTime(self, time):
        self.trainingTime = time
