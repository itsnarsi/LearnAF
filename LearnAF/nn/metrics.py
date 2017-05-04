from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple
from operator import mul
from functools import reduce
import numpy as np

class accuracy(AG, namedtuple("accuracy", ["AG1","AG2"])):
    
    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = af.mean(eval1(cache) == af.arith.cast((eval2(cache) >= 0.5), dtype = af.Dtype.f32))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        return None