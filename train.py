# python train_classifier.py -c sorriso
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from datetime import datetime
import os


try:
    dataset = pd.read_csv('mnist_train.csv')
except:
    exit()
    # TODO

Y = dataset['label'].values
onehot_encoder = OneHotEncoder(sparse=False)
Y = Y.reshape(len(Y), 1)
Y = onehot_encoder.fit_transform(Y)
X = dataset.drop('label', axis=1).values/255
X = X.reshape((X.shape[0], 28, 28, 1))
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=42)


### Treinamento ###
# DEFINE A ESTRUTURA DO MODELO
model = Sequential()
model.add(Conv2D(64, kernel_size=(3, 3),
                 activation='relu', input_shape=(28, 28, 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(10, activation='softmax'))

# COMPILANDO O MODELO
model.compile(loss='categorical_crossentropy',
              optimizer='Adam', metrics=['accuracy'])

# TREINANDO O MODELO
model.fit(X_train, Y_train, epochs=20, validation_data=(
    X_test, Y_test), batch_size=128, verbose=1)

# Exportando modelo
# serialize model to JSON
PATH = 'models/'
if not Path(PATH).exists():
    Path(PATH).mkdir()
name = str(datetime.now())

model_json = model.to_json()
with open(os.path.join(PATH, "model_{}_config.json".format(name)), "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights(
    os.path.join(PATH, "weights_{}.h5".format(name)))
print(" *** SAVED MODEL TO DISK ***")
