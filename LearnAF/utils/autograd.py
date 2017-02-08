import arrayfire as af
from collections import namedtuple

Point = "Dict[str, float]"

class AG:
    def eval(self) -> float:
        
        return self._eval({})
    
    def _eval(self, cache: dict) -> float:

        raise NotImplementedError
    
    def grad(self, variables) -> Point:
        cache = {}
        self._eval(cache)
        G = {}
        for i in range(len(variables)):
            G[variables[i].name] = 0
        self._grad(1, G, cache)
        return G

    def _grad(self, adjoint: float, gradient: Point, cache):

        raise NotImplementedError

    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Subtract(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __pow__(self, other):
        return Pow(self, other)

class Variable(AG):
    def __init__(self,value,name=None):
        self.value = value
        self.name = name
    def _eval(self, cache):
        cache[id(self)] = self.value
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        gradient[self.name] += adjoint

class Constant(AG, namedtuple("Constant", ["value"])):
    def _eval(self, cache):
        cache[id(self)] = self.value
        return self.value

    def _grad(self, ajoint, gradient, cache):
        pass

class Add(AG, namedtuple("Add", ["AG1", "AG2"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(cache) + eval2(cache)
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(adjoint, gradient, cache)
        self.AG2._grad(adjoint, gradient, cache)

class Subtract(AG, namedtuple("Subtract", ["AG1", "AG2"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(cache) - eval2(cache)
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        self.AG1._grad(adjoint, gradient, cache)
        self.AG2._grad(-adjoint, gradient, cache)

class Multiply(AG, namedtuple("Multiply", ["AG1", "AG2"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(cache) * eval2(cache)
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        lhs = cache[id(self.AG1)]
        rhs = cache[id(self.AG2)]
        self.AG1._grad(adjoint * rhs, gradient, cache)
        self.AG2._grad(adjoint * lhs, gradient, cache)

class Divide(AG, namedtuple("Divide", ["AG1", "AG2"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(cache) / eval2(cache)
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        high = cache[id(self.AG1)]
        low = cache[id(self.AG2)]
        self.AG1._grad(adjoint / low, gradient, cache)
        self.AG2._grad(-adjoint * high / low ** 2, gradient,
                                 cache)

class Pow(AG, namedtuple("Pow", ["AG1", "AG2"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(cache) ** eval2(cache)
        return cache[id(self)]
    def _grad(self, adjoint, gradient, cache):
        base = cache[id(self.AG1)]
        exp = cache[id(self.AG2)]

        self.AG1._grad(adjoint * exp * base ** (exp - 1), gradient, cache)
        self.AG2._grad(adjoint * af.arith.log(base) * base ** exp, gradient, cache)

class sin(AG, namedtuple("sin", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.sin(eval1(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        a = cache[id(self.AG1)]
        self.AG1._grad(adjoint * af.arith.cos(a), gradient, cache)
        
        
class cos(AG, namedtuple("cos", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.cos(eval1(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        a = cache[id(self.AG1)]
        self.AG1._grad(-adjoint * af.arith.sin(a), gradient, cache)


class exp(AG, namedtuple("exp", ["AG1"])):
    def _eval(self, cache):
        if id(self) not in cache:
            eval1 = self.AG1._eval
            cache[id(self)] = af.arith.exp(eval1(cache))
        return cache[id(self)]

    def _grad(self, adjoint, gradient, cache):
        a = cache[id(self.AG1)]
        self.AG1._grad(adjoint * af.arith.exp(a), gradient, cache)
