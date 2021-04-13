#coding: utf-8
import os, sys
sys.path.append(os.pardir) # to import files in the parent directory
import numpy as np
import time 
from functions import softmax, sigmoid
from mnist_data.mnist import load_mnist
import pickle

tic = time.clock()

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# Network is initialized by pre-determined weight
def initialize_net():
    with open("../mnist_data/sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    w1, w2, w3 = network['W1'], network['W2'], network['W3'] # pre-determined weights
    b1, b2, b3 = network['b1'], network['b2'], network['b3'] # pre-determined biases
    
    # First layer
    a1 = np.dot(x, w1) + b1
    z1 = sigmoid(a1)
    # Second layer 
    a2 = np.dot(z1, w2) + b2
    z2 = sigmoid(a2)
    # Third layer
    a3 = np.dot(z2, w3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = initialize_net()  # Initialize network

batch_size = 1000  # the number of batch
accuracy_cnt = 0  # initial count of accurate prediction

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
toc = time.clock()
elapesed_time = toc - tic
print("Elapsed time (s): " + str(elapesed_time))