from sklearn.model_selection import train_test_split
import csv
import numpy as np

from Normalizer import Normalizer
from SVMClassifier import SVMClassifier

from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)


def make_csv(data, predictions):

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

    with open("./data.csv", "w", newline='') as file:

        writer = csv.writer(file, delimiter=",")

        writer.writerow(
            featureNames
        )

        for i in range(0, len(data["trainingFeatures"])):
            writer.writerow(
                data["trainingFeatures"].tolist()[i] +
                [data["trainingClassesFirst"].tolist()[i]] +
                [data["trainingClassesSecond"].tolist()[i]] + ["Verdadero"]
            )

        for i in range(0, len(data["testingFeatures"])):
            writer.writerow(
                data["testingFeatures"].tolist()[i] +
                [data["testingClassesFirst"].tolist()[i]] +
                [data["testingClassesSecond"].tolist()[i]] +
                ["Falso"] + [predictions[0][2][i]] + [predictions[1][2][i]] +
                [predictions[2][2][i]]
            )


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

    accuracy = right/len(predictions)

    return (accuracy, predictions)


def holdout_cross_validation(
        classifier,
        data,
        lenData,
        training_name,
        testing_name,
        round):

    quantity_for_testing = int(lenData*0.3)

    toTrain = {
        "trainingFeatures": data[training_name],
        "trainingClasses": data["trainingClasses"+round]
    }

    X_train, X_test, y_train, y_test = train_test_split(
        toTrain["trainingFeatures"],
        toTrain["trainingClasses"],
        test_size=quantity_for_testing,
        random_state=42
    )

    toTrain = {
        "trainingFeatures": X_train,
        "trainingClasses": y_train
    }

    toTest = {
        "testingFeatures": X_test,
        "testingClasses": y_test
    }

    accuracyCV, _p = get_accuracy(classifier, toTrain, toTest)

    toTrain = {
        "trainingFeatures": data[training_name],
        "trainingClasses": data["trainingClasses"+round]
    }

    toFinalTest = {
        "testingFeatures": data[testing_name],
        "testingClasses": data["testingClasses"+round]
    }

    accuracyReal, predictions = get_accuracy(classifier, toTrain, toFinalTest)

    return (accuracyCV, accuracyReal, predictions)


def show_accuracy(model, predictions):
    print("----------------------------------------------")
    print("Tasa de error para: " + model)
    print()
    print("Cross validation>")
    print("Primera ronda: " + str(1-predictions[0][0]))
    print("Segunda ronda: " + str(1-predictions[1][0]))
    print("Segunda ronda (con primera incluida): " + str(1-predictions[2][0]))
    print()
    print("Pruebas>")
    print("Primera ronda: " + str(1-predictions[0][1]))
    print("Segunda ronda: " + str(1-predictions[1][1]))
    print("Segunda ronda (con primera incluida): " + str(1-predictions[2][1]))
    print("----------------------------------------------")


def svm_classification(lenData, pctTest, C, gamma, kernel):
    samples = generar_muestra_pais(lenData)
    quantity_for_testing = int(lenData*pctTest)

    normalizer = Normalizer()
    data = normalizer.prepare_data(samples, quantity_for_testing)

    svmClassifier = SVMClassifier(kernel, C, gamma)
    firstRound = holdout_cross_validation(
        svmClassifier,
        data,
        lenData,
        "trainingFeatures",
        "testingFeatures",
        "First"
    )

    secondRound = holdout_cross_validation(
        svmClassifier,
        data,
        lenData,
        "trainingFeatures",
        "testingFeatures",
        "Second"
    )

    secondWithFirst = holdout_cross_validation(
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
    make_csv(normalData, predictions)


def main():
    svm_classification(50000, 0.2, 1, 1, "rbf")


if __name__ == '__main__':
    main()
