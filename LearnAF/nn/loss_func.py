from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple
from operator import mul
from functools import reduce
import numpy as np

class mse(AG, namedtuple("mse", ["AG1","AG2"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = af.mean(af.pow2(eval1(cache) - eval2(cache)))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        m = reduce(mul, cache[id(self.AG2)].shape)
        g = (cache[id(self.AG1)] - cache[id(self.AG2)]) * 2.0/m
        
        self.AG1._grad(g * adjoint, gradient, cache)
        self.AG2._grad(g * -adjoint, gradient, cache)