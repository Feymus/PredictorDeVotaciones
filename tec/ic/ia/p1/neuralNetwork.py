from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
import numpy 
from Normalizer import Normalizer
from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)
from logisticRegression import logistic_regression_classifier

'''Lista con los Partidos Políticos'''
political_party = ['ACCESIBILIDAD SIN EXCLUSION','ACCION CIUDADANA',
                   'ALIANZA DEMOCRATA CRISTIANA','DE LOS TRABAJADORES',
                   'FRENTE AMPLIO','INTEGRACION NACIONAL','LIBERACION NACIONAL',
                   'MOVIMIENTO LIBERTARIO','NUEVA GENERACION','RENOVACION COSTARRICENSE',
                   'REPUBLICANO SOCIAL CRISTIANO','RESTAURACION NACIONAL',
                   'UNIDAD SOCIAL CRISTIANA','NULO','BLANCO']

class neural_network_classifier(object):

    '''Clasificador de modelos en red neuronal'''

    def __init__(self, layers, unit_per_layer, activation_func):
        self.layers = layers
        self.unit_per_layer = unit_per_layer
        self.activation_func = activation_func
        

    def train(self, data):

        x_train = data["trainingFeatures"]
        y_train = data["trainingClasses"]
        x_test = data["testingFeatures"]
        y_test = data["testingClasses"]
        
        oneHot = OneHotEncoder()
        var = logistic_regression_classifier()
        y_train = var.replace_political_party(y_train).reshape(-1,1)
        oneHot.fit(y_train)
        y_train = oneHot.transform(y_train).toarray()

        y_test = var.replace_political_party(y_test).reshape(-1,1)
        y_test = oneHot.transform(y_test).toarray()

        # create model
        self.model = Sequential()
        # input layer
        self.model.add(Dense(12, input_dim=x_train.shape[1], activation='relu'))
        # hidden layers
        for i in range(int(self.layers)):
            self.model.add(Dense(self.unit_per_layer[i], activation=self.activation_func))
        # output layer
        self.model.add(Dense(y_train.shape[1], activation='softmax'))

        # Compile model
        self.model.compile(loss='mean_squared_error',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Fit the model
        results = self.model.fit(x_train, y_train,
                            epochs=300,
                            batch_size=5,
                            verbose=0,
                            validation_data=(x_test, y_test))

    def classify(self, data):
        x_test = data["testingFeatures"]
        
        predictions = self.model.predict_classes(x_test)
        print(self.model.predict(x_test))
        print(predictions)
        return political_party[predictions[0]]


samples = generar_muestra_pais(100)
quantity_for_testing = int(100*0.2)
normalizer = Normalizer()
data = normalizer.prepare_data(samples, quantity_for_testing)
classes = numpy.append(
        data["trainingClassesFirst"],
        data["testingClassesFirst"],
        axis=0
    )
sample = { "trainingFeatures": data["trainingFeatures"], "trainingClasses": data["trainingClassesFirst"],"testingFeatures": data["testingFeatures"], "testingClasses": data["testingClassesFirst"]}
sample2 = { "testingFeatures": data["testingFeatures"], "testingClasses": data["testingClassesFirst"]}

pruebas = neural_network_classifier(10,[1,2,3,4,5,2,3,10,23,10],'sigmoid')
pruebas.train(sample)
print(pruebas.classify(sample2))

