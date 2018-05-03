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


cant = 2000
samples = generar_muestra_pais(cant)
quantity_for_testing = int(cant*0.2)
normalizer = Normalizer()
data = normalizer.prepare_data(samples, quantity_for_testing)

x_train = data["trainingFeatures"]
y_train = data["trainingClassesFirst"]
x_test = data["testingFeatures"]
y_test = data["testingClassesFirst"]

oneHot = OneHotEncoder()

y_train = logisticRegression.replace_political_party(y_train).reshape(-1,1)
oneHot.fit(y_train)
y_train = oneHot.transform(y_train).toarray()
y_test = logisticRegression.replace_political_party(y_test).reshape(-1,1)
oneHot.fit(y_test)
y_test = oneHot.transform(y_test).toarray()

temp = []
if len(y_test[0])<len(y_train[0]):
    for i in y_test:
        temp+=[numpy.append(i,normalizer.extra_list(len(y_train[0])-len(y_test[0]))).tolist()]
    oneHot.fit(temp)
    temp = oneHot.transform(temp).toarray()
    y_test = temp
if len(y_train[0])<len(y_test[0]):
    for i in y_train:
        temp += [numpy.append(i,normalizer.extra_list(len(y_train[0])-len(y_test[0]))).tolist()]
    oneHot.fit(temp)
    temp = oneHot.transform(temp).toarray()
    y_train = temp

# create model
model = Sequential()
model.add(Dense(12, input_dim=x_train.shape[1], activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(y_train.shape[1], activation='sigmoid'))
# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
results = model.fit(x_train, y_train, epochs=150, batch_size=10)#, validation_data = (x_test,y_test))
# evaluate the model
#print(numpy.mean(results.history["val_acc"]))
scores = model.evaluate(x_test, y_test)
predictions = model.predict(x_test)
print(predictions)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
