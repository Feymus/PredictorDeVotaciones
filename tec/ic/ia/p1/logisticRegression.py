from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Normalizer import Normalizer
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
    

    def __init__(self, samples, test_percent, l_regulizer):
        self.samples = samples
        self.test_percent = test_percent
        self.l_regulizer = l_regulizer



    """
    Cambia los partidos politicos por un numero, el cual va a ser el indice que tienen en la
    lista global political_party
    Entradas: Lista con los nombres del partido
    Salida: Lista con los partidos cambiados a numeros
    """
    def replace_political_party(self, party):
        for i in range(len(party)):
            party[i] = political_party.index(party[i].upper())
        return party

    """
    Regresion logistica para la primera ronda
    """
    def logistic_regression_r1(self):
        print("\n------FIRST ROUND------\n")
        
        quantity_for_testing = int(len(self.samples)*0.2)
        normalizer = Normalizer()
        data = normalizer.prepare_data(self.samples, quantity_for_testing)

        train_x = data["trainingFeatures"]
        train_y = data["trainingClassesFirst"]
        test_x = data["testingFeatures"]
        test_y = data["testingClassesFirst"]
        self.logic_regression(train_x , train_y, test_x, test_y)

    """
    Regresion logistica para la segunda ronda
    """
    def logistic_regression_r2(self):
        print("\n------SECOND ROUND------\n")

        quantity_for_testing = int(len(self.samples)*0.2)
        normalizer = Normalizer()
        data = normalizer.prepare_data(self.samples, quantity_for_testing)

        train_x = data["trainingFeatures"]
        train_y = data["trainingClassesSecond"]
        test_x = data["testingFeatures"]
        test_y = data["testingClassesSecond"]
        logic_regression(train_x , train_y, test_x, test_y)

    """
    Regresion logistica para la primera y segunda ronda
    """
    def logistic_regression_r2_r1(self):
        print("\n------THIRD ROUND------\n")

        quantity_for_testing = int(len(self.samples)*0.2)
        normalizer = Normalizer()
        data = normalizer.prepare_data(self.samples, quantity_for_testing)

        train_x = data["trainingFeaturesFirstInclude"]
        train_y = data["trainingClassesSecond"]
        test_x = data["testingFeaturesFirstInclude"]
        test_y = data["testingClassesSecond"]
        logic_regression(train_x , train_y, test_x, test_y)

    """
    Funcion de Regresion logistica
    """
    def logic_regression(self, train_x , train_y, test_x, test_y):
     
        learning_rate = 0.5
        print("Learning rate: ",learning_rate)
        scale = 0.0001
        print("Scale: ", scale,"\n")
        epochs = 800
        print("Epochs: ", epochs)
        display_step = 1

        oneHot = OneHotEncoder()
        train_y = self.replace_political_party(train_y).reshape(-1,1)
        oneHot.fit(train_y)
        train_y = oneHot.transform(train_y).toarray()

        test_y = self.replace_political_party(test_y).reshape(-1,1)
        oneHot.fit(test_y)
        test_y = oneHot.transform(test_y).toarray()

        #train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_percent, random_state=0)

        #Descomentar para ver el shape de cada test y train
##        print("Shape of X_train: ", train_x.shape)
##        print("Shape of y_train: ", train_y.shape)
##        print("Shape of X_test: ", test_x.shape)
##        print("Shape of y_test", test_y.shape)
##        print(" ")

        shape_x = train_x.shape[1]
        shape_y = train_y.shape[1]
        
        with tf.name_scope("Declaring_placeholder"):
            X = tf.placeholder(tf.float32, shape = [None, shape_x])
            y = tf.placeholder(tf.float32, shape = [None, None])

        #Weights
        with tf.name_scope("Declaring_variables"):
            W = tf.Variable(tf.zeros([shape_x, shape_y]))
            b = tf.Variable(tf.zeros([shape_y]))

        with tf.name_scope("Declaring_functions"):
            y_ = tf.nn.softmax(tf.add(tf.matmul(X, W), b))

        with tf.name_scope("calculating_cost"):
            # calculando costo
            cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=y_))

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
                    _, c = sess.run([optimizer, cost], feed_dict={X: train_x, y: train_y})
                    cost_in_each_epoch += c
                      # you can uncomment next two lines of code for printing cost when training
##                    if (epoch+1) % display_step == 0:
##                        print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch))
                
                print("Optimization Finished!")

                # Test model
                correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
                # calcula el accuracy
                accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                print("Accuracy:", accuracy.eval({X: train_x, y: train_y})*100)

                
##samples = generar_muestra_pais(2000)
##prueba = logistic_regression_classifier(samples, 20, 1)
##prueba.logistic_regression_r1()
