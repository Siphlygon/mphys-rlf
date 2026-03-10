import numpy as np

def sigmoid(x, x0, k, a, b):
    """Sigmoid function: a / (1 + exp(-k*(x-x0))) + b"""
    return a / (1 + np.exp(-k * (x - x0))) + b

def polynomial_deg4(x, a, b, c, d, e):
    """Quadratic polynomial: ax^2 + bx + c"""
    return a * x ** 4 + b * x ** 3 + c * x ** 2 + d * x + e

def polynomial(degree, x, *args):
    var = 0
    for i in range( degree + 1 ):
        var += args[ i ] * x ** i
    return var