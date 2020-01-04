import numpy as np
import pandas as pd
import json
import datetime
from hashlib import sha1
import pickle
from pathlib import Path

start = str(datetime.datetime.now())[0:19]
name_model = sha1((str(datetime.datetime.now()) +
                   str(np.random.random())).encode()).hexdigest()


def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))


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

neuronInput = 784
neuronHidden = 12
neuronOutput = len(t)
alpha = 0.05
epochs = 0
acc = 0.05

V = (np.random.rand(neuronInput, neuronHidden)) - 0.5
W = (np.random.rand(neuronHidden, neuronOutput)) - 0.5
Bv = (np.random.rand(neuronHidden)) - 0.5
Bw = (np.random.rand(neuronOutput)) - 0.5

Zin = np.zeros((neuronHidden), dtype=np.float64)
Z = np.zeros((neuronHidden), dtype=np.float64)
Yin = np.zeros((neuronOutput), dtype=np.float64)
Y = np.zeros((neuronOutput), dtype=np.float64)

deltaV = np.zeros((neuronInput, neuronHidden), dtype=np.float64)
deltaW = np.zeros((neuronHidden, neuronOutput), dtype=np.float64)
deltinhaW = np.zeros((neuronOutput), dtype=np.float64)
deltaBw = np.zeros((neuronOutput), dtype=np.float64)
deltinhaV = np.zeros((neuronHidden), dtype=np.float64)
deltaBv = np.zeros((neuronHidden), dtype=np.float64)

dataset = pd.read_csv('dataset/mnist_train.csv')
pixels = list(dataset.axes[1])
pixels.pop(0)

quadratic_error = 10
while acc < quadratic_error:
    quadratic_error = 0
    epochs += 1

    for row in range(len(dataset['label'])):
        number = int(dataset['label'][row])
        Xpad = []
        for pixel in pixels:
            Xpad.append(dataset[pixel][row])
        Xpad = np.array(Xpad)/255

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

        deltinhaW = (t[number] - Y) * (Y * (1 - Y))
        for i in range(neuronHidden):
            for j in range(neuronOutput):
                deltaW[i][j] = alpha * deltinhaW[j]*Z[i]

        deltaBw = alpha * deltinhaW

        for i in range(neuronHidden):
            for j in range(neuronOutput):
                deltinhaV[i] = deltinhaW[j]*W[i][j]*(Z[i]*(1-Z[i]))

        for i in range(neuronInput):
            for k in range(neuronHidden):
                deltaV[i][k] = alpha*deltinhaV[k]*Xpad[i]

        deltaBv = alpha*deltinhaV

        for i in range(neuronHidden):
            for k in range(neuronOutput):
                W[i][k] = W[i][k]+deltaW[i][k]

        Bw = Bw + deltaBw

        for i in range(neuronInput):
            for j in range(neuronHidden):
                V[i][j] = V[i][j] + deltaV[i][j]

        Bv = Bv + deltaBv

        #print('[INFO]: ')
        print("[INFO]: Img: {}/60000\nConclusão: {}%\n".format(row,
                                                               round((row/60000), 2)))
        print('[INFO]: ' + str(epochs))

    #EqTotal = EqTotal + 0.5 * ((Ypad - Y[0])**2)
    for i in range(neuronOutput):
        quadratic_error += 0.5((Y[i] - t[number][i])**2)

    #print("Img: {}/60000\nConclusão: {}%\n".format(row,round((row/60000),2)))

end = str(datetime.datetime.now())[0:19]

# Salvando o modelo no final do ciclo
model = {
    'architecture': {
        'neuron_input': neuronInput,
        'neuron_hidden': neuronHidden,
        'neuron_output': neuronOutput
    },
    'connection_weights': {
        'weights_hidden': V,
        'weights_output': W
    },
    'bias_weights': {
        'bias_hidden': Bv,
        'bias_output': Bw
    },
    'training_time': {
        'start': start,
        'end': end
    },
    'results': {
        'epochs': epochs,
        'quadratic_error': '',
        'alpha': alpha
    }
}

try:
    if not Path('model/').exists():
        Path('model/').mkdir()
    f = open('model/{}.pickle'.format(name_model), 'wb')
    f.write(pickle.dumps(model))
finally:
    f.close()
