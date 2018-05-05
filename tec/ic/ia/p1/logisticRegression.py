from tec.ic.ia.pc1.g06 import (
    generar_muestra_pais,
    generar_muestra_provincia
)
from Normalizer import Normalizer
import pandas as pd
import copy
import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf


political_party = ['ACCESIBILIDAD SIN EXCLUSION', 'ACCION CIUDADANA',
                   'ALIANZA DEMOCRATA CRISTIANA', 'DE LOS TRABAJADORES',
                   'FRENTE AMPLIO', 'INTEGRACION NACIONAL',
                   'LIBERACION NACIONAL', 'MOVIMIENTO LIBERTARIO',
                   'NUEVA GENERACION', 'RENOVACION COSTARRICENSE',
                   'REPUBLICANO SOCIAL CRISTIANO', 'RESTAURACION NACIONAL',
                   'UNIDAD SOCIAL CRISTIANA', 'NULO', 'BLANCO']

political_party2 = ['ACCION CIUDADANA', 'RESTAURACION NACIONAL',
                    'NULO', 'BLANCO']

strings = [
    "Provincia", "Canton", "Urbano/Rural", "Genero", "Edad",
    "Dependencia", "Alfabeta", "Escolaridadregular", "Trabaja", "Asegurado",
    "Condicion", "Hacinada", "Nacido", "Discapacitado", "Jefaturafemenina",
    "Jefaturacompartida", "Votoronda1", "Votoronda2"
]


class LogisticRegression(object):

    def __init__(self, round, norm):
        super(LogisticRegression, self).__init__()
        self.round = round
        if (round == 1):
            self.n = 15
        else:
            self.n = 4
        if (norm == 1):
            self.l1 = 1
            self.l2 = 0
        else:
            self.l1 = 0
            self.l2 = 1

    def classify(self, sample):
        testing = copy.copy(sample[0])

        for key in testing:
            testing[key] = [testing[key]]

        prediction = self.classifier.predict(
            input_fn=lambda: self.eval_input_fn(
                testing,
                labels=None
            )
        )

        for pred_dict in zip(prediction):
            predictedIndx = pred_dict[0]['class_ids'][0]

        if (self.round == 1):
            return political_party[predictedIndx]
        else:
            return political_party2[predictedIndx]

    def train(self, data):

        train_x, train_y = self.load_data(data)

        my_feature_columns = []
        for key in train_x.keys():
            if key not in strings:
                my_feature_columns.append(
                    tf.feature_column.numeric_column(key=key)
                )
            else:
                my_feature_columns.append(
                    tf.feature_column.categorical_column_with_hash_bucket(
                        key=key, hash_bucket_size=100
                    )
                )

        self.classifier = tf.estimator.LinearClassifier(
            feature_columns=my_feature_columns,
            n_classes=self.n,
            optimizer=tf.train.FtrlOptimizer(
                learning_rate=0.1,
                l1_regularization_strength=self.l1,
                l2_regularization_strength=self.l2
            )
        )

        # Train the Model.
        self.classifier.train(
            input_fn=lambda: self.train_input_fn(
                train_x, train_y
            ),
            steps=1
        )

    def load_data(self, data):

        if (type(data["trainingFeatures"]) != list):
            train_x = pd.DataFrame(copy.copy(data["trainingFeatures"]).tolist())
        else:
            train_x = pd.DataFrame(copy.copy(data["trainingFeatures"]))
        y = copy.copy(data["trainingClasses"])

        index = []
        for i in y:
            if (self.round == 1):
                index.append(political_party.index(i))
            else:
                index.append(political_party2.index(i))
        train_y = pd.DataFrame(index)

        return (train_x, train_y)

    def train_input_fn(self, features, labels):
        batch_size = 10
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

        # Shuffle, repeat, and batch the examples.
        dataset = dataset.shuffle(1).repeat().batch(batch_size)

        # Return the dataset.
        return dataset

    def eval_input_fn(self, features, labels):
        batch_size = 100
        features = dict(features)
        if labels is None:
            # No labels, use only features.
            inputs = features
        else:
            inputs = (features, labels)
        # Convert the inputs to a Dataset.
        dataset = tf.data.Dataset.from_tensor_slices(inputs)

        # Batch the examples
        assert batch_size is not None, "batch_size must not be None"
        dataset = dataset.batch(batch_size)

        # Return the dataset.
        return dataset



def load_data():

    lenData = 100
    pctTest = 0.2
    samples = generar_muestra_pais(lenData)
    quantity_for_testing = int(lenData * pctTest)

    normalizer = Normalizer()
    data = normalizer.prepare_data_tensor(samples, quantity_for_testing)

    train_x = pd.DataFrame(data["trainingFeatures"])
    print(data["trainingFeatures"])
    #train_y = pd.DataFrame(data["trainingClassesFirst"])
    test_x = data["testingFeatures"]
    #test_y = pd.DataFrame(data["testingClassesFirst"])

    y = data["trainingClassesSecond"]
    index = []
    for i in y:
        index.append(political_party2.index(i))

    train_y = pd.DataFrame(index)

    y = data["testingClassesSecond"]
    index = []
    for i in y:
        index.append(political_party2.index(i))

    test_y = data["testingClassesSecond"]

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(10).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)


    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset
