from sklearn.model_selection import train_test_split
import time

from Normalizer import Normalizer
from SVMClassifier import SVMClassifier

from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)


def pasar_a_csv(muestras):

    with open("./muestras.csv", "w", newline='') as file:

        writer = csv.writer(file, delimiter=",")

        for muestra in muestras:
            writer.writerow(muestra)


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
    normalizer = Normalizer()

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


def main():
    lenData = 10000
    samples = generar_muestra_pais(lenData)
    quantity_for_testing = int(lenData*0.2)

    normalizer = Normalizer()
    data = normalizer.prepare_data(samples, quantity_for_testing)

    svmClassifier = SVMClassifier("rbf", 0.000000001)

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

    print(secondRound)


if __name__ == '__main__':
    main()
