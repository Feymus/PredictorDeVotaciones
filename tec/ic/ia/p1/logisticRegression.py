from sklearn.feature_extraction import DictVectorizer
import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)
import svm

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
    

def LogicRegression(scale, epochs, cant, l_regulizer):
    
    samples = generar_muestra_pais(cant)

    data = svm.get_data(samples, 0.2)
    converter = svm.convert_data(data)

    x = data["trainingFeatures"]
    #test_x = data["testingFeatures"]

    y = data["trainingClassesSecond"]
    #test_y = replace_political_party(data["testingClassesSecond"]).reshape(-1,1)
    #test_y = np.array(test_y)
    
    learning_rate = 0.0001
    print("Learning rate: ",learning_rate)
    print("Scale: ", scale,"\n")
    print("Epochs: ", epochs)
    data = svm.get_data(samples, 0.2)
    converter = svm.convert_data(data)
    display_step = 1

    oneHot = OneHotEncoder()
    y = replace_political_party(y).reshape(-1,1)
    oneHot.fit(y)
    y = oneHot.transform(y).toarray()

    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1,
                                                        random_state=0)
    # let's print shape of each train and testing
    print("Shape of X_train: ", train_x.shape)
    print("Shape of y_train: ", train_y.shape)
    print("Shape of X_test: ", test_x.shape)
    print("Shape of y_test", test_y.shape)
    print(" ")
    
    shape_x = train_x.shape[1]
    shape_y = train_y.shape[1]

    with tf.name_scope("Declaring_placeholder"):
        X = tf.placeholder(tf.float32, [None, shape_x])
        y = tf.placeholder(tf.float32, [None, shape_y])

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
            regularization_penalty = tf.contrib.layers.apply_regularization(l1_regularizer, weights)
            regularized_loss = cost + regularization_penalty

        if (l_regulizer == 2):
            weights = tf.trainable_variables() #all variables of the graph
            l2_regularizer = tf.contrib.layers.l2_regularizer(scale = scale, scope=None)
            regularization_penalty = tf.contrib.layers.apply_regularization(l2_regularizer, weights)
            regularized_loss = cost + regularization_penalty

        else:
            print("Regulizer doesn't exist...")

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
        

LogicRegression(0.00000001, 100, 2000, 1)
LogicRegression(0.00000001, 100, 2000, 2)
# The data is turned to arrys of numbers only
#data = svm.get_data(samples, 0.2)
#converter = svm.convert_data(data)

##print(data["trainingFeatures"])
####print(data["testingFeatures"])
####print(data["trainingFeaturesFirstInclude"])
####print(data["testingFeaturesFirstInclude"])
##oneHot = OneHotEncoder()
##x = replace_political_party(data["trainingClassesFirst"]).reshape(-1,1)
##print(x)
##oneHot.fit(x)
##x = oneHot.transform(x).toarray()
##print(x)
##print(x.shape)
##print(data["trainingClassesSecond"])
##print(data["testingClassesFirst"])




