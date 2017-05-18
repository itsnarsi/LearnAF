from .autograd import *
import arrayfire as af
from collections import namedtuple
import time

class matmul(AG, namedtuple("matmul", ["AG1","AG2"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            eval2 = self.AG2._eval
            ops = af.blas.matmul(eval1(cache),eval2(cache))
            af.eval(ops)
            cache[id(self)] = ops
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):

        if adjoint.shape == (1,):
            ops1 = cache[id(self.AG2)] * adjoint
            ops2 = cache[id(self.AG1)] * adjoint
        else:
            ops1 = af.blas.matmulNT(adjoint, cache[id(self.AG2)])
            ops2 = af.blas.matmulTN(cache[id(self.AG1)], adjoint)
        af.eval(ops1)
        af.eval(ops2)
        self.AG1._grad(ops1, gradient, cache)
        self.AG2._grad(ops2, gradient, cache)

class add(AG, namedtuple("add", ["AG1","AG2"])):
    
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            eval2 = self.AG2._eval
            ops = self.ADD_CAST(eval1(cache), eval2(cache))
            af.eval(ops)
            cache[id(self)] = ops
            
        #print('add<-'+str(cache[id(self)].dtype()))
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
