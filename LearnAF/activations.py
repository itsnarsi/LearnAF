import arrayfire as af

class Linear(object):
    """docstring for Linear."""
    def __init__(self):
        super(Linear, self).__init__()

    def Fwd(self,x):
        return x

    def Bwd(self,x):
        return x * 0 + 1

class Sigmoid(object):
    """docstring for Linear."""
    def __init__(self):
        super(Sigmoid, self).__init__()
        self.sig = af.arith.sigmoid

    def Fwd(self,x):
        return self.sig(x)

    def Bwd(self,x):
        return self.sig(x) * (1 - self.sig(x))

class Tanh(object):
    """docstring for Linear."""
    def __init__(self):
        super(Tanh, self).__init__()
        self.tanh = af.arith.tanh
        self.pow2 = af.arith.pow2

    def Fwd(self,x):
        return self.tanh(x)

    def Bwd(self,x):
        return 1 - self.pow2(self.tanh(x))
