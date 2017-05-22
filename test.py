import numpy as np

import arrayfire as af
from scipy import ndimage
import matplotlib.pyplot as plt
from LearnAF import *
import tqdm
import pandas as pd

def learner(X,W):
    
    # 784 -> 64
    X1 = tanh(add(matmul(X,W[0]),W[1]))
    # 64 -> 16
    X2 = tanh(add(matmul(X1,W[2]),W[3]))
    # 16 -> 10
    YP = softmax(add(matmul(X2,W[4]),W[5]))
    return YP

af.set_backend('cpu')
#af.set_device(1)

f = np.load('/home/narsi/Downloads/mnist.npz')
Data = f['x_train']
y_train = f['y_train']

Data = Data.reshape(60000, 784)
Data = Data.astype('float32')/255.0
classes = np.asarray(to_categorical(y_train), dtype = np.float32)

# initialize weights randomly with mean 0
syn0 = np.array(2*np.random.random((Data.shape[1],64)) - 1, dtype = np.float32)
W1 = Variable(af.np_to_af_array(syn0),name='W1')
b1 = Variable(af.constant(0,1),name='b1')

syn0 = np.array(2*np.random.random((64,16)) - 1, dtype = np.float32)
W2 = Variable(af.np_to_af_array(syn0),name='W2')
b2 = Variable(af.constant(0,1),name='b2')

syn0 = np.array(2*np.random.random((16,10)) - 1, dtype = np.float32)
W3 = Variable(af.np_to_af_array(syn0),name='W3')
b3 = Variable(af.constant(0,1),name='b3')

w = [W1,b1,W2,b2,W3,b3]

batch = 64
X_np = Data[0:batch,:]
Xin = Constant(X_np)
Y_np = classes[0:batch,:]
Y = Constant(Y_np)

YP = learner(Xin,w)
e = mse(Y,YP)
sgd = SGD(lr = 0.01,momentum=0.9, nesterov=True)

for i in range(10):
    epoch_acc = []
    epoch_loss = []
    total_batchs = int(Data.shape[0]/batch)
    for j in tqdm.tqdm(range(total_batchs)):
        X_np = Data[j*batch:(j+1)*batch,:]
        Xin.value = af.np_to_af_array(X_np)
        Y_np = classes[j*batch:(j+1)*batch,:]
        Y.value = af.np_to_af_array(Y_np)
        
        (l,w) = sgd.update(e, w, i)
        acc = accuracy(Y,YP)
        epoch_acc.append(acc)
        epoch_loss.append(l)
    print('Accuracy :'+str(np.mean(epoch_acc)))
    print('Loss :'+str(np.mean(epoch_loss)))