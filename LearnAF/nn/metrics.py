from LearnAF.ops import *
import arrayfire as af
from collections import namedtuple
from operator import mul
from functools import reduce
import numpy as np

def accuracy(Y,YP):
    return af.mean(Y.eval() * af.arith.round(YP.eval()))