from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Normalizer import Normalizer
import copy
from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)


'''Lista con los Partidos Políticos'''
political_party = ['ACCESIBILIDAD SIN EXCLUSION','ACCION CIUDADANA',
                   'ALIANZA DEMOCRATA CRISTIANA','DE LOS TRABAJADORES',
                   'FRENTE AMPLIO','INTEGRACION NACIONAL','LIBERACION NACIONAL',
                   'MOVIMIENTO LIBERTARIO','NUEVA GENERACION','RENOVACION COSTARRICENSE',
                   'REPUBLICANO SOCIAL CRISTIANO','RESTAURACION NACIONAL',
                   'UNIDAD SOCIAL CRISTIANA','NULO','BLANCO']

class logistic_regression_classifier(object):

    '''Clasificador de Modelos de Regresion Logistica'''
  

    def __init__(self, l_regulizer, classes):
        self.l_regulizer = l_regulizer
        self.X = None
        self.y = None
        self.W = None
        self.b = None
        self.y_ = None
        self.oneHot = OneHotEncoder()
        allClasses = copy.copy(classes)
        allClasses = self.replace_political_party(allClasses).reshape(-1, 1)
        self.oneHot.fit(allClasses)

    def __init__(self):
        pass

    """
    Cambia los partidos politicos por un numero, el cual va a ser el indice que tienen en la
    lista global political_party
    Entradas: Lista con los nombres del partido
    Salida: Lista con los partidos cambiados a numeros
    """
    def replace_political_party(self, party):
        for i in range(len(party)):
            party[i] = political_party.index(party[i])
        return party

    def toparty(self, lista):
        temp = []
        for i in range(len(lista)):
            temp+=[political_party[lista[i].index(1)]]
        return temp

    def train(self, data, scale = 0.0001, epochs = 800):
        learning_rate = 0.05
        print("Learning rate: ",learning_rate)
        print("Scale: ", scale,"\n")
        print("Epochs: ", epochs)
        display_step = 1

        train_x = data["trainingFeatures"]
        train_y = data["trainingClasses"]
        test_x = data["testingFeatures"]
        test_y = data["testingClasses"]

        train_y = self.replace_political_party(train_y).reshape(-1,1)
        train_y = self.oneHot.transform(train_y).toarray()

        test_y = self.replace_political_party(test_y).reshape(-1,1)
        test_y = self.oneHot.transform(test_y).toarray()

        shape_x = train_x.shape[1]
        shape_y = train_y.shape[1]
        
        with tf.name_scope("Declaring_placeholder"):
            self.X = tf.placeholder(tf.float32, shape = [None, shape_x])
            self.y = tf.placeholder(tf.float32, shape = [None, shape_y])

        #Weights
        with tf.name_scope("Declaring_variables"):
            self.W = tf.Variable(tf.zeros([shape_x, shape_y]))
            self.b = tf.Variable(tf.zeros([shape_y]))

        with tf.name_scope("Prediction_functions"):
            self.y_ = tf.nn.softmax(tf.add(tf.matmul(self.X, self.W), self.b))

        with tf.name_scope("calculating_cost"):
            # calculando costo
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=self.y,
                                                                             logits=self.y_))

        with tf.name_scope("regulizer"):
            if (self.l_regulizer == 1):
                weights = tf.trainable_variables() #all variables of the graph
                l1_regularizer = tf.contrib.layers.l1_regularizer(scale = scale, scope=None)
                regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer,
                                                                                weights)
                regularized_loss = cost + regularization_penalty
                print("Using L1 Regulizer")
                
            if (self.l_regulizer == 2):
                weights = tf.trainable_variables() #all variables of the graph
                l2_regularizer = tf.contrib.layers.l2_regularizer(scale = scale, scope=None)
                regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer,
                                                                                weights)
                regularized_loss = cost + regularization_penalty
                print("Using L2 Regulizer")
                
        with tf.name_scope("declaring_gradient_descent"):
            # optimizer
            # usamos gradient descent para optimizar
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(regularized_loss)

        with tf.name_scope("starting_tensorflow_session"):
            with tf.Session() as sess:
                # inicializa las variables
                sess.run(tf.global_variables_initializer())
                for epoch in range(epochs):
                    cost_in_each_epoch = 0
                    # empieza el entrenamiento
                    _, c = sess.run([optimizer, cost], feed_dict={self.X: train_x,
                                                                  self.y: train_y})
                    cost_in_each_epoch += c
    ##                # you can uncomment next two lines of code for printing cost when training
    ##                if (epoch+1) % display_step == 0:
    ##                    print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch))


    def classify(self,data):
        test_x = data["testingFeatures"]
        test_y = data["testingClasses"]
        
        test_y = test_y.reshape(-1,1)
        test_y = self.oneHot.transform(test_y).toarray()

        with tf.name_scope("starting_tensorflow_session"):
            with tf.Session() as sess:
                # inicializa las variables
                sess.run(tf.global_variables_initializer())
                for epoch in range(epochs):
                    cost_in_each_epoch = 0
                    # empieza el entrenamiento
                    _, c = sess.run([optimizer, cost], feed_dict={self.X: train_x,
                                                                  self.y: train_y})
                    cost_in_each_epoch += c
    ##                # you can uncomment next two lines of code for printing cost when training
    ##                if (epoch+1) % display_step == 0:
    ##                    print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch))

                print("Accuracy Training:", accuracy.eval({X: train_x, y: train_y}))
##        sess = tf.Session()
##        with sess.as_default():
##            return self.toparty(self.y.eval({self.X: test_x, self.y: test_y}).tolist())
##            


##samples = generar_muestra_pais(100)
##quantity_for_testing = int(100*0.2)
##normalizer = Normalizer()
##data = normalizer.prepare_data(samples, quantity_for_testing)
##classes = np.append(
##        data["trainingClassesFirst"],
##        data["testingClassesFirst"],
##        axis=0
##    )
##sample = { "trainingFeatures": data["trainingFeatures"], "trainingClasses": data["trainingClassesFirst"],"testingFeatures": data["testingFeatures"], "testingClasses": data["testingClassesFirst"]}
##sample2 = { "testingFeatures": data["testingFeatures"], "testingClasses": data["testingClassesFirst"]}
##print(sample2["testingClasses"])
##prueba = logistic_regression_classifier(1,classes)
##prueba.train(sample)
##print(prueba.classify(sample2))

