# Neural Network from scratch
Neural networks from scratch in Python, C++ and CUDA without using existing machine learning libraries. (Python uses Numpy library for calculations). While a variety of machine learning libraries are avalilable, especially in Python, C/C++ is still used in high performance computiong (HPC). CUDA is used for GPGPU (general purpose computing on graphic processing unit) computing. This project aims to build neural network from scratch. In C++ and CUDA codes, matrix operation is defined without using libraries while numpy is used in Python.

## Network structure
The network has three hidden layers. The signal propagates forward. Sigmoid function (hidden layers) and softmax function (output layer) are used as activation functions. In this project, the trained weight was used to initialize the network.

## Implement image recognition task
MNIST dataset was used for handwritten digit recognition tasks. The networks classified 10,000 test images with an accuracy of 93.5%. Compared to C++ code (CPU processing), CUDA (GPU processing) achieved ~50 times faster processing time.

## Environmental Configuraltion
OS: Ubuntu 18.04 LTS  
CPU: Intel Core i7-3770k 3.50GHz  
RAM: 32GB  
GPU: NVIDIA Geforce GTX1060 6GB  
CUDA 10.2

## Todo (in progress)
Wrap C++ code using **Cython** that provides C-like performace with Python code. This will allow to run C++ code (class or function) from Python.
