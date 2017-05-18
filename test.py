import numpy as np

import arrayfire as af
from scipy import ndimage
import matplotlib.pyplot as plt
from LearnAF import *
import tqdm
import pandas as pd

def learner(X,W):
    # 784 -> 64
    X1 = sigmoid(add(matmul(X,W[0]),W[1]))
    # 64 -> 16
    X2 = sigmoid(add(matmul(X1,W[2]),W[3]))
    # 16 -> 10
    YP = sigmoid(add(matmul(X2,W[4]),W[5]))
    return YP

af.set_backend('cpu')

DF = pd.read_csv('/home/narsi/Downloads/train.csv')
Data = np.asarray(DF, dtype = np.float32)/255.0
Data = Data[1:]
labels = np.asarray(DF['label'], dtype = np.float32)
classes = np.asarray(to_categorical(labels), dtype = np.float32)

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

Xin = Constant(af.np_to_af_array(np.zeros((32,Data.shape[1]),dtype = np.float32)))
Y = Constant(af.np_to_af_array(np.zeros((32,10),dtype = np.float32)))

YP = learner(Xin,w)
e = mse(Y,YP)
sgd = SGD(lr = 0.001,momentum=0.9)

for i in range(100):
    epoch_acc = []
    epoch_loss = []
    total_batchs = int(Data.shape[0]/32)
    for j in tqdm.trange(total_batchs):
        X_np = Data[j*32:(j+1)*32,:]
        Xin.set_value(af.np_to_af_array(X_np))
        Y_np = classes[j*32:(j+1)*32,:]
        Y.set_value(af.np_to_af_array(Y_np))
        
        acc = accuracy(Y,YP)
        (l,w) = sgd.update(e, w, i)
        epoch_acc.append(acc)
        epoch_loss.append(l)
        
    print('Accuracy :'+str(np.mean(epoch_acc)))
    print('Loss :'+str(np.mean(epoch_loss)))