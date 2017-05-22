from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple
from operator import mul
from functools import reduce
import numpy as np

class MeanSquareError(AG, namedtuple("mse", ["yTarge","yPredict"])):
    
    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.yTarge._eval(cache), self.yPredict._eval(cache)
            ops = af.mean(af.pow(eval1 - eval2,2))
            af.sync()
            cache[id(self)] = ops
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        m = float(reduce(mul, cache[id(self.yPredict)].shape))
        g = self.MUL_CAST((cache[id(self.yTarge)] - cache[id(self.yPredict)]), af.data.constant(2.0/m, 1, dtype=af.Dtype.f32))
        
        ops1 = self.MUL_CAST(g, adjoint)
        ops2 = self.MUL_CAST(g, -adjoint)

        self.yTarge._grad(ops1, gradient, cache)
        self.yPredict._grad(ops2, gradient, cache)
        af.sync()

class CrossEntropyError(AG, namedtuple("CrossEntropy", ["yTarge","yPredict"])):

    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.yTarge._eval(cache), self.yPredict._eval(cache)
            ops = -1.0 * af.sum(eval1 * af.arith.log(eval2 + 1e-7), dim = 1)
            cache[id(self)] = af.mean(ops)
            af.sync()
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        m = cache[id(self.yPredict)].shape[0]

        ops1 = af.arith.log(cache[id(self.yPredict)] + 1e-7)
        ops1 = self.MUL_CAST(ops1, af.data.constant(1.0/m, 1, dtype=af.Dtype.f32))
        ops2 = cache[id(self.yTarge)] / (cache[id(self.yPredict)] + 1e-7)
        ops2 = self.MUL_CAST(ops2, af.data.constant(1.0/m, 1, dtype=af.Dtype.f32))

        ops1 = self.MUL_CAST(ops1, -adjoint)
        ops2 = self.MUL_CAST(ops2, -adjoint)
        #print(ops1)
        self.yTarge._grad(ops1, gradient, cache)
        self.yPredict._grad(ops2, gradient, cache)
        af.sync()