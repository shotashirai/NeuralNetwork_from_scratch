# Neural Network from scratch
This project aims to build a neural network from scratch in three different programming platforms (**Python**, **C++** and **CUDA**) without using existing machine learning libraries. While a variety of machine learning libraries are available in Python, C/C++ and CUDA are used in **high-performance computing (HPC)** platform and **GPGPU** (general-purpose computing on graphic processing unit) computing. In C++ and CUDA codes, a matrix operation is defined without using libraries while NumPy is used in Python for the calculation propose.

## Network structure
The network has three hidden layers. The signal propagates forward. Sigmoid function (hidden layers) and softmax function (output layer) are used as activation functions. In this project, the trained weights were used to initialize the network.

## Implement image recognition task
MNIST dataset was used for handwritten digit recognition tasks. The networks (Python, C++ and CUDA) classified 10,000 test images with **an accuracy of 93.5%**. Compared to C++ code (CPU processing), CUDA (GPU processing) achieved ~50 times faster processing time.

## Environmental Configuraltion
OS: Ubuntu 18.04 LTS  
CPU: Intel Core i7-3770k 3.50GHz  
RAM: 32GB  
GPU: NVIDIA Geforce GTX1060 6GB  
CUDA 10.2

## Todo (in progress)
Wrap C++ code using **Cython** that provides C-like performance with Python code. This will allow running C++ code (class or function) from Python, leading to faster processing.
