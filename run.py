import argparse
import pandas as pd
import numpy as np
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', required=True,
                    help="Entre com os pesos da rede neural trinada")
args = vars(parser.parse_args())


try:
    weights = pickle.loads(open(args['weights'], 'wb').read())
except Exception as e:
    print(e)
    exit()

neuronInput = weights['neuronInput']
neuronHidden = weights['neuronHidden']
neuronOutput = weights['neuronOutput']
alpha = weights['alpha']
V = weights['weights_hidden']
W = weights['weights_output']
Bv = weights['bias_hidden']
Bw = weights['bias_output']
Zin = np.zeros((neuronHidden), dtype=np.float64)
Z = np.zeros((neuronHidden), dtype=np.float64)
Yin = np.zeros((neuronOutput), dtype=np.float64)
Y = np.zeros((neuronOutput), dtype=np.float64)

t0 = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
t1 = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
t2 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
t3 = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
t4 = [0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
t5 = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]
t6 = [0, 0, 0, 0, 0, 0, 1, 0, 0, 0]
t7 = [0, 0, 0, 0, 0, 0, 0, 1, 0, 0]
t8 = [0, 0, 0, 0, 0, 0, 0, 0, 1, 0]
t9 = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
t = [t0, t1, t2, t3, t4, t5, t6, t7, t8, t9]


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


dataset = pd.read_csv('mninst_test.csv')
labels = dataset['label']
dataset = dataset.drop('label', axis=1)
pixels = dataset.values

acc = 0

for row in range(len(pixels)):
    Xpad = []
    for pixel in pixels:
        Xpad.append(dataset[pixel][row])
    Xpad = np.array(Xpad)/255
    Xpad = dataset/255

    for i in range(neuronHidden):
        ac = 0
        for j in range(neuronInput):
            ac = ac + V[j][i] * Xpad[j]
        Zin[i] = ac + Bv[i]
        Z[i] = sigmoid(Zin[i])

    for i in range(neuronOutput):
        ac = 0
        for j in range(neuronHidden):
            ac = ac + Z[j] * W[j][i]
        Yin[i] = ac + Bw[i]
        Y[i] = sigmoid(Yin[i])

    Y = list(Y)
    number = labels[row]
    number_predict = Y.index(max(Y))-1
    if number == number_predict:
        acc += 1

acc = acc/neuronOutput
print('[INFO] ACURACIA', acc)
