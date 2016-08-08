# ActivationFunctions.py

import numpy as np
import arrayfire as af

class ActivationFunctions:
	def __init__(self):
		print("Message for init class")

	def linear(self, x):
	    return x

	def sigmoid(self, x):
	    return af.arith.sigmoid(x)

	def tanh(self, x):
	    return af.arith.tanh(x)

	def relu(self, x):
	    cond = af.Array(x > 0,dtype=af.Dtype.f32)
	    return cond * x

	def softmax(self, x):
	    exp_ = af.arith.exp(x) - af.algorithm.max(x)
	    sum_ = af.arith.sum(exp_)
	    return exp_ / sum_

