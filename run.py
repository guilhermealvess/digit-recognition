import argparse
import pandas as pd
import numpy as np
import pickle
from keras.models import model_from_json
from sklearn.preprocessing import OneHotEncoder


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--modelname', required=True,
                    help="Entre com os pesos da rede neural trinada")
args = vars(parser.parse_args())


try:
    test = pd.read_csv('mninst_test.csv')
except Exception as e:
    print(e)
    exit()

Y = test['label'].values
onehot_encoder = OneHotEncoder(sparse=False)
Y = Y.reshape(len(Y), 1)
Y = onehot_encoder.fit_transform(Y)
X = test.drop('label', axis=1).values/255
X = X.reshape((X.shape[0], 28, 28, 1))


model = model_from_json("model_{}_config.json".format(args['modelname']))
model.load_weights("weights_{}.h5".format(args['modelname']))


score = model.evaluate(X, Y, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
