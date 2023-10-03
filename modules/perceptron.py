# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #2

import numpy as np

# Perceptron class with n input dimensions
class Perceptron:
    # constructor
    def __init__(self, dim, weight, activation, differential):
        self._dim = dim
        self._weight = weight
        self._activation = activation
        self._differential = differential

        self._invec = None # input vector
        self._out = None # output
        self._localgrad_weight = None # local gradient of weight
        self._localgrad_input = None # local gradient of input

    # input vector getter and setter
    @property
    def invec(self):
        return self._invec
    
    @invec.setter
    def invec(self, x):
        # input validation check
        if len(x) != self._dim:
            raise Exception('Input dimension is not matched.')
        self._invec = x

    # output getter
    @property
    def out(self):
        return self._out
    
    # weight getter and setter
    @property
    def weight(self):
        return self._weight
    
    @weight.setter
    def weight(self, w):
        # weight validation check
        if len(w) != self._dim:
            raise Exception('Weight dimension is not matched.')
        self._weight = w

    # forward propagation
    def forward(self, x=None):
        # input vector
        if x is not None:
            self.invec = x
        # calculate output vector
        self._out = self._activation(np.dot(self._weight, self._invec))
        # return output vector
        return self._out
    
    # local gradient of weight getter
    @property
    def localgrad_weight(self):
        return self._localgrad_weight
    
    # local gradient of input getter
    @property
    def localgrad_input(self):
        return self._localgrad_input

    # backward propagation : calculate local gradient
    def backward(self):
        # f'(z) where z = wx, f is activation function
        diff = self._differential(np.dot(self._weight, self._invec))
        # local gradient of weight = df/dz * dz/dw = f'(z) * x
        self._localgrad_weight = diff * self._invec
        # local gradient of input = df/dz * dz/dx = f'(z) * w
        self._localgrad_input = diff * self._weight

    # string representation
    def __str__(self):
        return f'Perceptron(invec={self._invec}, weight={self._weight}, outvec={self._out})'

# step activation function
def step(x):
    return 1 if x >= 0 else 0

# step activation differential (let's assume it is 1)
def step_diff(x):
    return 1
