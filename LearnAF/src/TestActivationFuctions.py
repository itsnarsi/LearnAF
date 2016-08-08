# TestActivationFuctions.py

import numpy as np
import ActivationFunctions as activation
import arrayfire as af

funcs = activation.ActivationFunctions()

x_1D = np.arange(-4, 4,0.01, dtype=np.float32)
x = af.np_to_af_array(x_1D)

x_linear = funcs.linear(x)
x_sigmod = funcs.sigmoid(x)
x_tanh = funcs.tanh(x)
x_relu = funcs.relu(x)
x_softm = funcs.softmax(x)

print("LINEAR")
print(x_linear)
print(">>>")
print("SIGMOD")
print(x_sigmod)
print(">>>")
print("TANH")
print(x_tanh)
print(">>>")
print("RELU")
print(x_relu)
print(">>>")
print("SOFTMAX")
print(x_softm)
print(">>>")

