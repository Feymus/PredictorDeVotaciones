from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from Normalizer import Normalizer
from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)


political_party = ['ACCESIBILIDAD SIN EXCLUSION','ACCION CIUDADANA',
                   'ALIANZA DEMOCRATA CRISTIANA','DE LOS TRABAJADORES',
                   'FRENTE AMPLIO','INTEGRACION NACIONAL','LIBERACION NACIONAL',
                   'MOVIMIENTO LIBERTARIO','NUEVA GENERACION','RENOVACION COSTARRICENSE',
                   'REPUBLICANO SOCIAL CRISTIANO','RESTAURACION NACIONAL',
                   'UNIDAD SOCIAL CRISTIANA','NULO','BLANCO']

"""
Cambia los partidos politicos por un numero, el cual va a ser el indice que tienen en la
lista global political_party
Entradas: Lista con los nombres del partido
Salida: Lista con los partidos cambiados a numeros
"""
def replace_political_party(party):
    for i in range(len(party)):
        party[i] = political_party.index(party[i].upper())
    return party


#x = data["trainingFeatures"]
#test_x = data["testingFeatures"]

#y = data["trainingClassesSecond"]
#test_y = replace_political_party(data["testingClassesSecond"]).reshape(-1,1)
#test_y = np.array(test_y)

def logistic_regression_r1(cant, scale, epochs, test_percent, l_regulizer):
    print("\n------FIRST ROUND------\n")
    
    samples = generar_muestra_pais(cant)
    quantity_for_testing = int(cant*0.2)
    normalizer = Normalizer()
    data = normalizer.prepare_data(samples, quantity_for_testing)

    train_x = data["trainingFeatures"]
    train_y = data["trainingClassesFirst"]
    test_x = data["testingFeatures"]
    test_y = data["testingClassesFirst"]
    logic_regression(train_x , train_y, test_x, test_y, scale, epochs, test_percent, l_regulizer)


def logistic_regression_r2(cant, scale, epochs, test_percent, l_regulizer):
    print("\n------SECOND ROUND------\n")

    samples = generar_muestra_pais(cant)
    quantity_for_testing = int(cant*0.2)
    normalizer = Normalizer()
    data = normalizer.prepare_data(samples, quantity_for_testing)

    train_x = data["trainingFeatures"]
    train_y = data["trainingClassesSecond"]
    test_x = data["testingFeatures"]
    test_y = data["testingClassesSecond"]
    logic_regression(train_x , train_y, test_x, test_y, scale, epochs, test_percent, l_regulizer)


def logistic_regression_r2_r1(cant, scale, epochs, test_percent, l_regulizer):
    print("\n------THIRD ROUND------\n")

    samples = generar_muestra_pais(cant)
    quantity_for_testing = int(cant*0.2)
    normalizer = Normalizer()
    data = normalizer.prepare_data(samples, quantity_for_testing)

    train_x = data["trainingFeaturesFirstInclude"]
    train_y = data["trainingClassesSecond"]
    test_x = data["testingFeaturesFirstInclude"]
    test_y = data["testingClassesSecond"]
    logic_regression(train_x , train_y, test_x, test_y, scale, epochs, test_percent, l_regulizer)


def logic_regression(train_x , train_y, test_x, test_y, scale, epochs, test_percent, l_regulizer):
 
    learning_rate = 0.01
    print("Learning rate: ",learning_rate)
    print("Scale: ", scale,"\n")
    print("Epochs: ", epochs)
    display_step = 1

    oneHot = OneHotEncoder()
    train_y = replace_political_party(train_y).reshape(-1,1)
    oneHot.fit(train_y)
    train_y = oneHot.transform(train_y).toarray()

    test_y = replace_political_party(test_y).reshape(-1,1)
    oneHot.fit(test_y)
    test_y = oneHot.transform(test_y).toarray()

    #train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = test_percent, random_state=0)

    # let's print shape of each train and testing
    print("Shape of X_train: ", train_x.shape)
    print("Shape of y_train: ", train_y.shape)
    print("Shape of X_test: ", test_x.shape)
    print("Shape of y_test", test_y.shape)
    print(" ")

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
        y_ = tf.sigmoid(tf.add(tf.matmul(X, W), b))

    with tf.name_scope("calculating_cost"):
        # calculating cost
        cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=y_))

    with tf.name_scope("regulizer"):
        if (l_regulizer == 1):
            weights = tf.trainable_variables() #all variables of the graph
            l1_regularizer = tf.contrib.layers.l1_regularizer(scale = scale, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer,
                                                                            weights)
            regularized_loss = cost + regularization_penalty
            print("Using L1 Regulizer")
            
        if (l_regulizer == 2):
            weights = tf.trainable_variables() #all variables of the graph
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale = scale, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer,
                                                                            weights)
            regularized_loss = cost + regularization_penalty
            print("Using L2 Regulizer")
            
    with tf.name_scope("declaring_gradient_descent"):
        # optimizer
        # we use gradient descent for our optimizer 
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(regularized_loss)

    with tf.name_scope("starting_tensorflow_session"):
        with tf.Session() as sess:
            # initialize all variables
            sess.run(tf.global_variables_initializer())
            for epoch in range(epochs):
                cost_in_each_epoch = 0
                # let's start training
                _, c = sess.run([optimizer, cost], feed_dict={X: train_x, y: train_y})
                cost_in_each_epoch += c
##                # you can uncomment next two lines of code for printing cost when training
##                if (epoch+1) % display_step == 0:
##                    print("Epoch: {}".format(epoch + 1), "cost={}".format(cost_in_each_epoch))
            
            print("Optimization Finished!")

            # Test model
            correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(y, 1))
            # Calculate accuracy for 3000 examples
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            print("Accuracy:", accuracy.eval({X: test_x, y: test_y}))
        


#data = svm.get_data(generar_muestra_pais(1000), 0.2)
#converter = svm.convert_data(data) 
#logic_regression(x, y, 0.00000001, 100, 20, 1)
#logic_regression(0.00000001, 100, 2000, 20, 2)

logistic_regression_r1(3000, 0.01, 100, 20, 1)
logistic_regression_r2(3000, 0.01, 100, 20, 2)
logistic_regression_r2_r1(3000, 0.01, 100, 20, 2)


# The data is turned to arrys of numbers only


##ENTRENAMIENDO PRIMERA RONDA
##print(data["trainingFeatures"])
##print(data["trainingClassesFirst"])

##PRUEBA PRIMERA RONDA
##print(data["testingFeatures"])
##print(data["testingClassesFirst"])

##SEGUNDA RONDA
##print(data["trainingFeatures"])
##print(data["trainingClassesSecond"])

##print(data["testingFeatures"])
##print(data["testingClassesSecond"])

##LA RARA
#print(data["trainingFeaturesFirstInclude"])
#print(data["trainingClassesSecond"])

##print(data["testingFeaturesFirstInclude"])
##print(data["trainingClassesSecond"])







