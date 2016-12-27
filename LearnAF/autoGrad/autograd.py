import arrayfire as af
from collections import namedtuple

Point = "Dict[str, float]"

class AG:
    def eval(self, point: Point) -> float:

        return self._eval(point, {})
    
    def _eval(self,point: Point, cache: dict) -> float:

        raise NotImplementedError
    
    def grad(self, point: Point) -> Point:

        cache = {}
        self._eval(point, cache)
        G = {key: 0 for key in point}
        self._grad(point, 1, G, cache)
        return G

    def _grad(self, point: Point, adjoint: float, gradient: Point, cache):

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

    
class Variable(AG, namedtuple("Variable", ["name"])):
    def _eval(self, point, cache):
        cache[id(self)] = point[self.name]
        return point[self.name]

    def _grad(self, point, adjoint, gradient, cache):
        gradient[self.name] += adjoint

class Constant(AG, namedtuple("Constant", ["value"])):
    def _eval(self, point, cache):
        cache[id(self)] = self.value
        return self.value

    def _grad(self, point, ajoint, gradient, cache):
        pass

class Add(AG, namedtuple("Add", ["AG1", "AG2"])):
    def _eval(self, point, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(point, cache) + eval2(point, cache)
        return cache[id(self)]

    def _grad(self, point, adjoint, gradient, cache):
        self.AG1._grad(point, adjoint, gradient, cache)
        self.AG2._grad(point, adjoint, gradient, cache)

class Subtract(AG, namedtuple("Subtract", ["AG1", "AG2"])):
    def _eval(self, point, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(point, cache) - eval2(point, cache)
        return cache[id(self)]

    def _grad(self, point, adjoint, gradient, cache):
        self.AG1._grad(point, adjoint, gradient, cache)
        self.AG2._grad(point, -adjoint, gradient, cache)

class Multiply(AG, namedtuple("Multiply", ["AG1", "AG2"])):
    def _eval(self, point, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(point, cache) * eval2(point, cache)
        return cache[id(self)]

    def _grad(self, point, adjoint, gradient, cache):
        lhs = cache[id(self.AG1)]
        rhs = cache[id(self.AG2)]
        self.AG1._grad(point, adjoint * rhs, gradient, cache)
        self.AG2._grad(point, adjoint * lhs, gradient, cache)

class Divide(AG, namedtuple("Divide", ["AG1", "AG2"])):
    def _eval(self, point, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(point, cache) / eval2(point, cache)
        return cache[id(self)]

    def _grad(self, point, adjoint, gradient, cache):
        high = cache[id(self.AG1)]
        low = cache[id(self.AG2)]
        self.AG1._grad(point, adjoint / low, gradient, cache)
        self.AG2._grad(point, -adjoint * high / low ** 2, gradient,
                                 cache)

class Pow(AG, namedtuple("Pow", ["AG1", "AG2"])):
    def _eval(self, point, cache):
        if id(self) not in cache:
            eval1, eval2 = self.AG1._eval, self.AG2._eval
            cache[id(self)] = eval1(point, cache) ** eval2(point, cache)
        return cache[id(self)]
    def _grad(self, point, adjoint, gradient, cache):
        base = cache[id(self.AG1)]
        exp = cache[id(self.AG2)]

        self.AG1._grad(point, adjoint * exp * base ** (exp - 1), gradient, cache)
        self.AG2._grad(point, adjoint * af.arith.log(base) * base ** exp, gradient, cache)
