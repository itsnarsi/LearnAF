from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple
from operator import mul
from functools import reduce
import numpy as np

class mse(AG, namedtuple("mse", ["AG1","AG2"])):
    
    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval(cache), self.AG2._eval(cache)
            ops = af.mean(af.pow(eval1 - eval2,2))
            #af.eval(ops)
            af.sync()
            cache[id(self)] = ops
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        m = float(reduce(mul, cache[id(self.AG2)].shape))
        g = (cache[id(self.AG1)] - cache[id(self.AG2)]) * 2.0/m
        ops1 = self.MUL_CAST(g, adjoint)
        ops2 = self.MUL_CAST(g, -adjoint)
        self.AG1._grad(ops1, gradient, cache)
        self.AG2._grad(ops2, gradient, cache)
        af.sync()