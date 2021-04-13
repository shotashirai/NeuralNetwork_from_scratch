import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def softmax(x):
    c = np.max(x)
    exp_i = np.exp(x - c)
    sum_exp = np.sum(np.exp(x - c))
    return exp_i / sum_exp

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    
    batch_size = y.shape[0]

    return -np.sum(t * np.log(y + 1e-7)) / batch_size

def numerical_diff(f, x, h=1e-4):
    return (f(x+h) - f(x-h)) / 2*h
