from .autograd import *
import arrayfire as af
from collections import namedtuple


class sin(AG, namedtuple("sin", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.sin(eval1(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(adjoint * af.arith.cos(cache[id(self.AG1)]), gradient, cache)

class asin(AG, namedtuple("asin", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.asin(eval1(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(adjoint * (1 / af.arith.sqrt(1 - af.arith.pow2(cache[id(self.AG1)]))), gradient, cache)

class cos(AG, namedtuple("cos", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.cos(eval1(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(-adjoint * af.arith.sin(cache[id(self.AG1)]), gradient, cache)

class acos(AG, namedtuple("acos", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.acos(eval1(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(adjoint * (-1 / af.arith.sqrt(1 - af.arith.pow2(cache[id(self.AG1)]))), gradient, cache)

class tan(AG, namedtuple("tan", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.tan(eval1(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(adjoint * (1 + af.arith.pow2(af.arith.tan(cache[id(self.AG1)]))), gradient, cache)

class atan(AG, namedtuple("atan", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.atan(eval1(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(adjoint * (1 / (1 + af.arith.pow2(cache[id(self.AG1)]))), gradient, cache)
