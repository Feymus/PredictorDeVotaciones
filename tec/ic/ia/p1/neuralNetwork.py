from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
import numpy 
from Normalizer import Normalizer
from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)
import logisticRegression

class neural_network_classifier(object):

    '''Clasificador de modelos en red neuronal'''

    def __init__(self, layers, unit_per_layer, activation_func):
        self.layers = layers
        self.unit_per_layer = unit_per_layer
        self.activation_func = activation_func
        

    def train(self, data):

        train_x = data["trainingFeatures"]
        train_y = data["trainingClasses"]
        
        oneHot = OneHotEncoder()
        y_train = logisticRegression.replace_political_party(y_train).reshape(-1,1)
        oneHot.fit(y_train)
        y_train = oneHot.transform(y_train).toarray()

        # create model
        model = Sequential()
        # input layer
        model.add(Dense(12, input_dim=x_train.shape[1], activation='relu'))
        # hidden layers
        for i in range(int(self.init_per_layer)):
            model.add(Dense(self.layers[i], activation=self.activation_func))
        # output layer
        model.add(Dense(y_train.shape[1], activation='sigmoid'))

        # Compile model
        model.compile(loss='categorical_crossentropy',
                      optimizer='adam',
                      metrics=['accuracy'])

        # Fit the model
        results = model.fit(x_train, y_train,
                            epochs=150,
                            batch_size=10,
                            verbose=0)

    def classify(self, data):
        x_test = data["testingFeatures"]
        
        predictions = model.predict_classes(x_test)
        return logisticRegression.political_party[predictions[0]])
        
