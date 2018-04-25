from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)

indicators = [
    "Provincia", "Canton", "Total de la población", "Superficie",
    "Densidad de la población", "Urbano/Rural", "Género", "Edad",
    "Dependencia", "Alfabeta", "Escolaridad promedio",
    "Escolaridad regular", "Trabaja", "Asegurado",
    "Cant. casas individuales", "Ocupantes promedio", "Condicion",
    "Hacinada", "Nacido en...", "Discapacitado", "Jefatura femenina",
    "Jefatura compartida", "Voto ronda 1", "Voto ronda 2"
]


def convert_to_dict_list(samples, featureNames):
    features = []

    for featureList in samples:

        dictFeatures = {}
        featureNum = 0
        for feature in featureList:
            try:
                feature = float(feature)
            except ValueError:
                # The feature is a string
                pass
            dictFeatures[featureNames[featureNum]] = feature
            featureNum += 1

        features.append(dictFeatures)

    return features


def convert_data(data):

    global indicators

    for key in data:
        if "Classes" not in key:
            data[key] = convert_to_dict_list(data[key], indicators)

    v = DictVectorizer(sparse=False)

    data["trainingFeatures"] = v.fit_transform(data["trainingFeatures"])
    data["testingFeatures"] = v.transform(data["testingFeatures"])

    return v


def get_data(samples, pct_test):

    samplesArray = np.array(samples)
    X = samplesArray[:, :22]
    y = samplesArray[:, 22:]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=pct_test,
        random_state=42
    )

    y_train_first_round = y_train[:, 0]
    y_train_second_round = y_train[:, 1]

    y_test_first_round = y_test[:, 0]
    y_test_second_round = y_test[:, 1]

    X_train_2 = np.append(X_train, y_train[:, :1], axis=1)
    X_test_2 = np.append(X_test, y_test[:, :1], axis=1)

    return ({
        "trainingFeatures": X_train,
        "testingFeatures": X_test,
        "trainingFeaturesFirstInclude": X_train_2,
        "testingFeaturesFirstInclude": X_test_2,
        "trainingClassesFirst": y_train_first_round,
        "trainingClassesSecond": y_train_second_round,
        "testingClassesFirst": y_test_first_round,
        "testingClassesSecond": y_test_second_round
    })


def test(classifier, testingFeatures, testingClasses):
    print(classifier.score(testingFeatures, testingClasses))
    # predictions = classifier.predict(testingFeatures)
    # right = 0

    # for i in range(0, len(predictions)):
    #    if (predictions[i] == testingClasses[i]):
    #        right += 1

    # print(right/len(predictions))


def train_svm(samples, pct_test):

    data = get_data(samples, pct_test)
    # The data is turned to arrys of numbers only
    converter = convert_data(data)

    clf = svm.SVC(kernel='linear', C=1)
    clf.fit(data["trainingFeatures"], data["trainingClassesFirst"])
    test(clf, data["testingFeatures"], data["testingClassesFirst"])


def main():
    samples = generar_muestra_pais(10000)
    train_svm(samples, 0.2)


if __name__ == '__main__':
    main()
