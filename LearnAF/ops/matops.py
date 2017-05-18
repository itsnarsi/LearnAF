from .autograd import *
import arrayfire as af
from collections import namedtuple
import time

class matmul(AG, namedtuple("matmul", ["AG1","AG2"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            eval2 = self.AG2._eval
            cache[id(self)] = af.blas.matmul(eval1(cache),eval2(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):

        if adjoint.shape == (1,):
            self.AG1._grad(cache[id(self.AG2)] * adjoint, gradient, cache)
            self.AG2._grad(cache[id(self.AG1)] * adjoint, gradient, cache)
        else:
            self.AG1._grad(af.blas.matmulNT(adjoint, cache[id(self.AG2)]), gradient, cache)
            self.AG2._grad(af.blas.matmulTN(cache[id(self.AG1)], adjoint), gradient, cache)

class add(AG, namedtuple("add", ["AG1","AG2"])):
    
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            eval2 = self.AG2._eval
            cache[id(self)] = self.ADD_CAST(eval1(cache), eval2(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(self.unbroadcast(adjoint, cache[id(self.AG1)].shape), gradient, cache)
        self.AG2._grad(self.unbroadcast(adjoint, cache[id(self.AG2)].shape), gradient, cache)

class sub(AG, namedtuple("sub", ["AG1","AG2"])):
    
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            eval2 = self.AG2._eval
            cache[id(self)] = self.SUB_CAST(eval1(cache),eval2(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        #print(cache[id(self.AG2)])
        self.AG1._grad(self.unbroadcast(adjoint, cache[id(self.AG1)].shape), gradient, cache)
        self.AG2._grad(-1 * self.unbroadcast(adjoint, cache[id(self.AG2)].shape), gradient, cache)
