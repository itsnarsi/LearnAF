from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple

class sigmoid(AG, namedtuple("sigmoid", ["AG1"])):
    def _eval(self, point, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.sigmoid(eval1(point, cache))
        return cache[id(self)]

    def _grad(self, point, adjoint, gradient, cache):
        a = cache[id(self.AG1)]
        g = af.arith.sigmoid(a) * (1 - af.arith.sigmoid(a))
        self.AG1._grad(point, adjoint * g, gradient, cache)

class tanh(AG, namedtuple("tanh", ["AG1"])):
    def _eval(self, point, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.tanh(eval1(point, cache))
        return cache[id(self)]

    def _grad(self, point, adjoint, gradient, cache):
        a = cache[id(self.AG1)]
        g = 1 - af.arith.pow2(af.arith.tanh(a))
        self.AG1._grad(point, adjoint * g, gradient, cache)
