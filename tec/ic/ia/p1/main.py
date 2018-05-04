from sklearn.model_selection import train_test_split
import csv
import numpy as np
import sys
import getopt

import time

from Normalizer import Normalizer
from SVMClassifier import SVMClassifier
from DecisionTree import DecisionTree
from KDTree import Kd_Tree
#from logisticRegression import logistic_regression_classifier
#from neuralNetwork import neural_network_classifier
from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)


savedSamples = []


def clear_csv():
    with open("./data.csv", "w", newline='') as file:

        writer = csv.writer(file, delimiter=",")

        writer.writerow("")


def make_csv(k, data, lenData, pctTest, predictions):

    featureNames = [
        "Provincia", "Canton", "Total de la población", "Superficie",
        "Densidad de la población", "Urbano/Rural", "Género", "Edad",
        "Dependencia", "Alfabeta", "Escolaridad promedio",
        "Escolaridad regular", "Trabaja", "Asegurado",
        "Cant. casas individuales", "Ocupantes promedio", "Condicion",
        "Hacinada", "Nacido en...", "Discapacitado", "Jefatura femenina",
        "Jefatura compartida", "Voto ronda 1", "Voto ronda 2",
        "es_entrenamiento", "prediccion_r1", "prediccion_r2",
        "prediccion_r2_con_r1"
    ]
    quantity_for_testing = int(lenData*0.3)
    quantityTraining = len(data["trainingFeatures"])

    with open("./data.csv", "a", newline='') as file:

        writer = csv.writer(file, delimiter=",")

        writer.writerow(
            featureNames
        )

        group = 0
        groupLen = int((lenData-(lenData*pctTest)-(lenData*0.3)) // k)

        for i in range(0, k):
            index = 0
            for j in range(group, group+groupLen):
                writer.writerow(
                    data["trainingFeatures"].tolist()[j] +
                    [data["trainingClassesFirst"].tolist()[j]] +
                    [data["trainingClassesSecond"].tolist()[j]] +
                    ["Verdadero"] + [predictions[0][2][i][1][index]] +
                    [predictions[1][2][i][1][index]] +
                    [predictions[2][2][i][1][index]]
                )
                index += 1

        for i in range(0, quantity_for_testing):
            writer.writerow(
                data["trainingFeatures"].tolist()[i] +
                [data["trainingClassesFirst"].tolist()[i]] +
                [data["trainingClassesSecond"].tolist()[i]] + ["Verdadero"] +
                [predictions[0][1][i]] + [predictions[1][1][i]] +
                [predictions[2][1][i]]
            )

        for i in range(0, len(data["testingFeatures"])):
            writer.writerow(
                data["testingFeatures"].tolist()[i] +
                [data["testingClassesFirst"].tolist()[i]] +
                [data["testingClassesSecond"].tolist()[i]] +
                ["Falso"] + [predictions[0][4][i]] + [predictions[1][4][i]] +
                [predictions[2][4][i]]
            )

        file.close()


def get_accuracy(classifier, toTrain, toTest):

    predictions = []
    classifier.train(toTrain)

    for sample in toTest["testingFeatures"]:
        prediction = classifier.classify([sample])
        predictions.append(prediction[0])


    testingClasses = toTest["testingClasses"]
    right = 0

    for i in range(0, len(predictions)):
        if (predictions[i] == testingClasses[i]):
            right += 1

    accuracy = right / len(predictions)

    return (accuracy, predictions)


accList = []

def k_fold_cross_validation2(k, classifier, data, lenData):
    groupLen = len(data["trainingFeatures"]) // k
    group = 0
    toTrain = {}
    toTest = {}
    results = []

    while group < len(data["trainingFeatures"]):

        testingFeatures = data["trainingFeatures"][group:group+groupLen]
      

        toTest["testingFeatures"] = testingFeatures
       

        trainingFeatures = np.append(
            data["trainingFeatures"][:group],
            data["trainingFeatures"][group+groupLen:],
            axis=0
        )


        toTrain["trainingFeatures"] = trainingFeatures
     

        results.append(get_accuracy(classifier, toTrain, toTest))

        group += groupLen

    return results
def k_fold_cross_validation(k, classifier, data, lenData):
    groupLen = len(data["trainingFeatures"]) // k
    group = 0
    toTrain = {}
    toTest = {}
    results = []

    while group < len(data["trainingFeatures"]):

        testingFeatures = data["trainingFeatures"][group:group+groupLen]
        testingClasses = data["trainingClasses"][group:group+groupLen]

        toTest["testingFeatures"] = testingFeatures
        toTest["testingClasses"] = testingClasses

        trainingFeatures = np.append(
            data["trainingFeatures"][:group],
            data["trainingFeatures"][group+groupLen:],
            axis=0
        )

        trainingClasses = np.append(
            data["trainingClasses"][:group],
            data["trainingClasses"][group+groupLen:],
            axis=0
        )

        toTrain["trainingFeatures"] = trainingFeatures
        toTrain["trainingClasses"] = trainingClasses

        results.append(get_accuracy(classifier, toTrain, toTest))

        group += groupLen

    return results
def cross_validation2(
        k,
        classifier,
        data,
        lenData,
        training_name,
        testing_name):

    quantity_for_testing = int(lenData*0.3)
    results = []

    toTrain = {
        "trainingFeatures": data[training_name],
      
    }

    X_train = toTrain["trainingFeatures"][quantity_for_testing:]
   

    X_test = toTrain["trainingFeatures"][:quantity_for_testing]
   

    toTrain = {
        "trainingFeatures": X_train,
    
    }

    results = k_fold_cross_validation2(k, classifier, toTrain, lenData)

    toTest = {
        "testingFeatures": X_test,
        
    }

    accuracyCV, predictionsCV = get_accuracy(classifier, toTrain, toTest)

    toTrain = {
        "trainingFeatures": data[training_name],
     
    }

    toFinalTest = {
        "testingFeatures": data[testing_name],
      
    }

    accuracyReal, predictions = get_accuracy(classifier, toTrain, toFinalTest)
    accList.append((accuracyCV, accuracyReal))

    return (accuracyCV, predictionsCV, results, accuracyReal, predictions)


def cross_validation(
        k,
        classifier,
        data,
        lenData,
        training_name,
        testing_name,
        round):

    quantity_for_testing = int(lenData*0.3)
    results = []

    toTrain = {
        "trainingFeatures": data[training_name],
        "trainingClasses": data["trainingClasses" + round]
    }

    X_train = toTrain["trainingFeatures"][quantity_for_testing:]
    y_train = toTrain["trainingClasses"][quantity_for_testing:]

    X_test = toTrain["trainingFeatures"][:quantity_for_testing]
    y_test = toTrain["trainingClasses"][:quantity_for_testing]

    toTrain = {
        "trainingFeatures": X_train,
        "trainingClasses": y_train
    }

    results = k_fold_cross_validation(k, classifier, toTrain, lenData)

    toTest = {
        "testingFeatures": X_test,
        "testingClasses": y_test
    }

    accuracyCV, predictionsCV = get_accuracy(classifier, toTrain, toTest)

    toTrain = {
        "trainingFeatures": data[training_name],
        "trainingClasses": data["trainingClasses" + round]
    }

    toFinalTest = {
        "testingFeatures": data[testing_name],
        "testingClasses": data["testingClasses" + round]
    }

    accuracyReal, predictions = get_accuracy(classifier, toTrain, toFinalTest)
    accList.append((accuracyCV, accuracyReal))

    return (accuracyCV, predictionsCV, results, accuracyReal, predictions)


def show_accuracy(model, predictions):
    print("----------------------------------------------")
    print("Tasa de error para: " + model)
    print()
    print("K-fold Cross validation>")
    print()
    print("Holdout Cross validation>")
    print("Primera ronda: " + str(1-predictions[0][0]))
    print("Segunda ronda: " + str(1-predictions[1][0]))
    print("Segunda ronda (con primera incluida): " + str(1-predictions[2][0]))
    print()
    print("Pruebas>")
    print("Primera ronda: " + str(1-predictions[0][3]))
    print("Segunda ronda: " + str(1-predictions[1][3]))
    print("Segunda ronda (con primera incluida): " + str(1-predictions[2][3]))
    print("----------------------------------------------")


def svm_classification(k, lenData, pctTest, C=1, gamma=1, kernel="rbf"):

    clear_csv()

    samples = generar_muestra_pais(lenData)
    quantity_for_testing = int(lenData * pctTest)

    normalizer = Normalizer()
    data = normalizer.prepare_data(samples, quantity_for_testing)

    svmClassifier = SVMClassifier(kernel, C, gamma)
    firstRound = cross_validation(
        k,
        svmClassifier,
        data,
        lenData,
        "trainingFeatures",
        "testingFeatures",
        "First"
    )

    secondRound = cross_validation(
        k,
        svmClassifier,
        data,
        lenData,
        "trainingFeatures",
        "testingFeatures",
        "Second"
    )

    secondWithFirst = cross_validation(
        k,
        svmClassifier,
        data,
        lenData,
        "trainingFeaturesFirstInclude",
        "testingFeaturesFirstInclude",
        "Second"
    )

    normalData = normalizer.get_normal_data()
    predictions = [firstRound, secondRound, secondWithFirst]

    # show_accuracy("SVM", predictions)
    make_csv(k, normalData, lenData, pctTest, predictions)


def kd_tree_classification(k, lenData, pctTest, neightboards):

    clear_csv()

    samples = generar_muestra_pais(lenData)
    quantity_for_testing = int(lenData * pctTest)

    normalizer = Normalizer()
    data = normalizer.prepare_data(samples, quantity_for_testing)

    svmClassifier = Kd_Tree(neightboards)
    firstRound = cross_validation(
        k,
        svmClassifier,
        data,
        lenData,
        "trainingFeatures",
        "testingFeatures",
        "First"
    )

    secondRound = cross_validation(
        k,
        svmClassifier,
        data,
        lenData,
        "trainingFeatures",
        "testingFeatures",
        "Second"
    )

    secondWithFirst = cross_validation(
        k,
        svmClassifier,
        data,
        lenData,
        "trainingFeaturesFirstInclude",
        "testingFeaturesFirstInclude",
        "Second"
    )

    normalData = normalizer.get_normal_data()
    predictions = [firstRound, secondRound, secondWithFirst]

    show_accuracy("SVM", predictions)
    make_csv(k, normalData, lenData, pctTest, predictions)

def desicion_tree(k, lenData, pctTest,threshold):
    
    clear_csv()

    samples = generar_muestra_pais(lenData)
    quantity_for_testing = int(lenData * pctTest)

    normalizer = Normalizer()
    data = normalizer.separate_data_2(samples, quantity_for_testing)

    decisionTree = DecisionTree(threshold)
    firstRound = cross_validation2(
        k,
        decisionTree,
        data,
        lenData,
        "trainingFeaturesFirst",
        "testingFeaturesFirst"
    )

    secondRound = cross_validation2(
        k,
        decisionTree,
        data,
        lenData,
        "trainingFeaturesSecond",
        "testingFeaturesSecond"
    )

    secondWithFirst = cross_validation2(
        k,
        decisionTree,
        data,
        lenData,
        "trainingFeaturesFirstInclude",
        "testingFeaturesFirstInclude"
    
    )

    normalData = normalizer.get_normal_data()
    predictions = [firstRound, secondRound, secondWithFirst]

    # show_accuracy("SVM", predictions)
    make_csv(k, normalData, lenData, pctTest, predictions)

def pruebas():
    # svm_classification(1000, 0.2, C=10, gamma=0.00833333333, kernel="rbf")
    lenData = 5000
    print(lenData)
    print("kernel: ", "rbf", " C: ", 1, " G: ", 1)
    pctTest = 0.2

    # samples = generar_muestra_provincia(lenData, "SAN JOSE")
    # quantity_for_testing = int(lenData*pctTest)

    # normalizer = Normalizer()
    # data = normalizer.prepare_data(samples, quantity_for_testing)

    # svm_classification(10, lenData, pctTest, C=1, gamma=1, kernel="rbf")

    time1 = time.time()

    for i in range(0, 30):
        samples = generar_muestra_pais(lenData)
        quantity_for_testing = int(lenData*pctTest)

        normalizer = Normalizer()
        data = normalizer.prepare_data(samples, quantity_for_testing)
        svm_classification(
            10, lenData, pctTest, C=1, gamma=1, kernel="rbf")

    time2 = time.time()

    print("ms: ", ((time2-time1)*1000.0))

    totalacc = 0.0
    for i in range(0, len(accList), 3):
        totalacc += accList[i][1]
    print("ER: ", 1-(totalacc/30.0))

    totalacc = 0.0
    for i in range(1, len(accList), 3):
        totalacc += accList[i][1]
    print("ER: ", 1-(totalacc/30.0))

    totalacc = 0.0
    for i in range(2, len(accList), 3):
        totalacc += accList[i][1]
    print("ER: ", 1-(totalacc/30.0))

    # svmClassifier = SVMClassifier("rbf", 1, 1)
    # svmClassifier.test_parameters(
    #    data["trainingFeatures"],
    #    data["trainingClassesSecond"]
    # )


def main(argv):
    # print(argv)
    if(len(argv)<5):
        print("\n     ***INSTRUCCIONES***\n")
        print("     main.py --poblacion<poblacion> --porcentaje-pruebas <porcentaje>  bandera\n")
        print("     BANDERAS:\n")
        print("*    --regresion-logistica [--l1 o --l2] ")
        print("*    --red-neuronal --numero-capas <numero> --unidades-por-capa <numero> --funcion-activacion ? ")
        print("*    --knn --k <numero de vecinos>")
        print("*    --arbol --umbral-poda <numero>")
        print("*    --svm --kernel <linear, poly, rbf, sigmoid> --C <numero> --Gamma <numero>\n")
    else:

        lenData = int(argv[1])
        pctTest = float(argv[3])/100.0

        if(argv[4]=="--regresion-logistica"):
            print("REGRESION LOGISTICA")
            if(len(argv)==6):
                print(argv[5])
            else:
                print("ERROR: Parametros Incompletos")
                print("Debe ingresar --l1 o --l2")

        elif(argv[4]=="--red-neuronal"):
            print("RED NEURONAL")
            if(len(argv)==11):
                print(argv[5])
            else:
                print("ERROR: Parametros Incompletos")
                print("Debe ingresar --numero-capas <numero> --unidades-por-capa <numero> --funcion-activacion ?")
        elif(argv[4]=="--knn"):
            print("KD-TREE")
            if(len(argv)==7):
                print(argv[6])
                neightboards=int(argv[6])
                kd_tree_classification(3, lenData, pctTest, neightboards)
            else:
                print("ERROR: Parametros Incompletos")
                print("Debe ingresar --k <numero de vecinos>")
        elif(argv[4]=="--arbol"):
            print("ARBOL DE DECISION")
            if(len(argv)==7):
                print(argv[6])
                threshold=float(argv[6])
                desicion_tree(3, lenData, pctTest,threshold)


            else:
                print("ERROR: Parametros Incompletos")
                print("Debe ingresar --umbral-poda <numero>")

        elif(argv[4]=="--svm"):
            print("SVM")
            print(len(argv))
            if(len(argv)==11):
                # print(argv[5:])
                k = argv[6]
                c = float(argv[8])
                g = float(argv[10])
                svm_classification(3, lenData, pctTest, C=c, gamma=g, kernel=k)

            else:
                print("ERROR: Parametros Incompletos")
                print("Debe ingresar --kernel <linear, poly, rbf, sigmoid> --C <numero> --Gamma <numero>")
        else:
             print("ERROR: Bandera inexistente")



    #print(generar_muestra_pais(5))


if __name__ == '__main__':
    main(sys.argv[1:])
