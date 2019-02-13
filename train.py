import datetime
import numpy as np
import pandas as pd
import json
from hashlib import md5


start = str(datetime.datetime.now())[0:19]
name_model = md5((str(datetime.datetime.now()) + str(np.random.random())).encode()).hexdigest()
def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

t0 = [1,0,0,0,0,0,0,0,0,0]
t1 = [0,1,0,0,0,0,0,0,0,0]
t2 = [0,0,1,0,0,0,0,0,0,0]
t3 = [0,0,0,1,0,0,0,0,0,0]
t4 = [0,0,0,0,1,0,0,0,0,0]
t5 = [0,0,0,0,0,1,0,0,0,0]
t6 = [0,0,0,0,0,0,1,0,0,0]datetime
t7 = [0,0,0,0,0,0,0,1,0,0]
t8 = [0,0,0,0,0,0,0,0,1,0]
t9 = [0,0,0,0,0,0,0,0,0,1]
t = [t0,t1,t2,t3,t4,t5,t6,t7,t8,t9]

neuronInput = 784
neuronHidden = 12
neuronOutput = 10
alpha = 0.05         
epochs = 0

V = (np.random.rand(neuronInput,neuronHidden)) -0.5
W = (np.random.rand(neuronHidden,neuronOutput)) -0.5
Bv = (np.random.rand(neuronHidden)) -0.5
Bw = (np.random.rand(neuronOutput)) -0.5

Zin = np.zeros((neuronHidden),dtype=np.float64)
Z = np.zeros((neuronHidden),dtype=np.float64)
Yin = np.zeros((neuronOutput),dtype=np.float64)
Y = np.zeros((neuronOutput),dtype=np.float64)

deltaV = np.zeros((neuronInput, neuronHidden), dtype=np.float64)
deltaW = np.zeros((neuronHidden, neuronOutput), dtype=np.float64)
deltinhaW = np.zeros((neuronOutput), dtype=np.float64)
deltaBw = np.zeros((neuronOutput), dtype=np.float64)
deltinhaV = np.zeros((neuronHidden), dtype=np.float64)
deltaBv = np.zeros((neuronHidden), dtype=np.float64)

dataset = pd.read_csv('dataset/mnist_train.csv')
pixels = list(dataset.axes[1])
pixels.pop(0)

for row in range(len(dataset['label'])):
    number = dataset['label'][row]
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
            deltaW[i][j] = alfa * deltinhaW[j]*Z[i]
    
    deltaBw = alfa * deltinhaW

    for i in range(neuronHidden):
        for j in range(neuronOutput):
            deltinhaV[i] = deltinhaW[j]*W[i][j]*(Z[i]*(1-Z[i]))

    for i in range(neuronInput):
        for k in range(neuronHidden):
            deltaV[i][k] = alfa*deltinhaV[k]*Xpad[i]

    deltaBv = alfa*deltinhaV

    for i in range(neuronHidden):
        for k in range(neuronOutput):
            W[i][k] = W[i][k]+deltaW[i][k]

    Bw = Bw + deltaBw

    for i in range(neuronInput):
        for j in range(neuronHidden):
            V[i][j] = V[i][j] + deltaV[i][j]

    Bv = Bv + deltaBv

    print("Img: {}/60000\nConclusão: {}%\n".format(row,round((row/600),2)))

end = str(datetime.now())[0:19]

# Salvando o modelo no final do ciclo
model = {
    'architecture': {
        'neuron_input': neuronInput,
        'neuron_hidden': neuronHidden,
        'neuron_output': neuron_Output
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
model = json.dumps(model)

_json = open('model/'+name_model+'.json','wt')
try:
    _json.write(model)
finally:
    _json.close()
