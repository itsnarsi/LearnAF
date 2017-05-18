from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple
from operator import mul
from functools import reduce
import numpy as np
from sklearn.metrics import accuracy_score

def accuracy(Y,YP):
    P = np.argmax(np.asarray(YP.eval()), axis=1)
    T = np.argmax(np.asarray(Y.eval()), axis=1)

    return accuracy_score(T, P)