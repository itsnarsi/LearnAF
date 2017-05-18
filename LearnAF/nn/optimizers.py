from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple
from operator import mul
from functools import reduce
import numpy as np

class SGD:
    def __init__(self, lr=0.01, momentum=0., decay=0., nesterov=False):
        self.lr = lr
        self.momentum = momentum
        self.decay = decay
        self.nesterov = nesterov

    def update(self, loss, weights, epoch):
        error = loss.eval()

        weight_grads = loss.grad(weights)
        

        if self.decay > 0:
            self.lr *= (1. / (1. + self.decay * epoch))
        
        if epoch == 0:
            shapes = [W.value.shape for W in weights]
            self.moments = [af.np_to_af_array(np.zeros(shape, dtype = np.float32)) for shape in shapes]
            
        weight_updates = []

        for wk_id in range(len(weights)):
            wk = weights[wk_id].name
            m = self.moments[wk_id]
            v = self.momentum * m - self.lr * weight_grads[wk]
            if self.nesterov == True:
                v = self.momentum * v - self.lr * weight_grads[wk]
            self.moments[wk_id] = v
            weights[wk_id].value += v

            weight_updates.append(weights[wk_id])
        
        return (error,weight_updates)