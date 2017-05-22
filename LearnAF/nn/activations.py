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
            x = list(eval1.shape)
            dims = [None,None,None,None]
            for i in range(len(x)):
                dims[i] = x[i]
            zeros = af.data.constant(0, dims[0], d1=dims[1], d2=dims[2], d3=dims[3], dtype = af.Dtype.f32)
            cond = eval1 > zeros
            ops = af.select(cond, eval1, zeros)
            ops = eval1 * ops
            cache[id(self)] = ops
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        eval1 = self.AG1._eval(cache)
        x = list(eval1.shape)
        dims = [None,None,None,None]
        for i in range(len(x)):
            dims[i] = x[i]
        zeros = af.data.constant(0, dims[0], d1=dims[1], d2=dims[2], d3=dims[3], dtype = af.Dtype.f32)
        cond = eval1 > zeros
        g = af.select(cond, eval1, zeros)
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
        g = af.arith.cast(a * (1 - a)* adjoint, af.Dtype.f32)
        self.AG1._grad(g, gradient, cache)