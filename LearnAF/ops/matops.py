from .autograd import *
import arrayfire as af
from collections import namedtuple

class matmul(AG, namedtuple("dotmul", ["AG1","AG2"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            eval2 = self.AG2._eval
            cache[id(self)] = af.blas.matmul(eval1(cache),eval2(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(af.blas.matmulTN(cache[id(self.AG2)],adjoint), gradient, cache)
        self.AG2._grad(af.blas.matmulNT(cache[id(self.AG1)],adjoint), gradient, cache)
