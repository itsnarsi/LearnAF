from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple

class sigmoid(AG, namedtuple("sigmoid", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            ops = af.arith.sigmoid(eval1( cache))
            #af.eval(ops)
            cache[id(self)] = ops
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        a = cache[id(self)]
        #g = af.arith.sigmoid(a) * (1 - af.arith.sigmoid(a))
        g = a * (1 - a)* adjoint
        #af.eval(g)
        self.AG1._grad(g , gradient, cache)

class tanh(AG, namedtuple("tanh", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.tanh(eval1( cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        a = cache[id(self)]
        g = 1 - af.arith.pow(af.arith.tanh(a), 2)
        self.AG1._grad(g * adjoint, gradient, cache)


class relu(AG, namedtuple("relu", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval(cache)
            ops = eval1 > 0
            ops = eval1 * ops.as_type(af.Dtype.f32)
            cache[id(self)] = ops
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        a = cache[id(self)] > 0
        g = a.as_type(af.Dtype.f32)
        self.AG1._grad(g * adjoint, gradient, cache)

class softmax(AG, namedtuple("softmax", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval(cache)
            ops = af.arith.exp(eval1)
            ops = self.DIV_CAST(ops, af.sum(ops, dim=1))
            cache[id(self)] = ops
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        a = cache[id(self)]
        g = a * (1 - a)* adjoint
        self.AG1._grad(g, gradient, cache)