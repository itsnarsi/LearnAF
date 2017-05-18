from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple

class sigmoid(AG, namedtuple("sigmoid", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            ops = af.arith.sigmoid(eval1( cache))
            af.eval(ops)
            cache[id(self)] = ops
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        a = cache[id(self)]
        #g = af.arith.sigmoid(a) * (1 - af.arith.sigmoid(a))
        g = a * (1 - a)* adjoint
        af.eval(g)
        self.AG1._grad(g , gradient, cache)

class tanh(AG, namedtuple("tanh", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.tanh(eval1( cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        a = cache[id(self)]
        g = 1 - af.arith.pow2(af.arith.tanh(a))
        self.AG1._grad(g * adjoint, gradient, cache)
