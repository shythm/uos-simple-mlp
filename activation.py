# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #3

import numpy as np

# step activation function
def step(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1, 0)

# step activation differential (let's assume it is 1)
def step_diff(x: np.ndarray) -> np.ndarray:
    return np.ones(x.shape)

# relu activation function
def relu(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, x, 0)

# relu activation differential
def relu_diff(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1, 0)

# sigmoid activation function
def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))

# sigmoid activation differential
def sigmoid_diff(x: np.ndarray) -> np.ndarray:
    return sigmoid(x) * (1 - sigmoid(x))

# get activation function by name
def get_activation(name: str):
    if name == 'step':
        return step
    elif name == 'relu':
        return relu
    elif name == 'sigmoid':
        return sigmoid
    else:
        raise Exception('Unknown activation function.')
    
# get activation differential by name
def get_differential(name: str):
    if name == 'step':
        return step_diff
    elif name == 'relu':
        return relu_diff
    elif name == 'sigmoid':
        return sigmoid_diff
    else:
        raise Exception('Unknown activation function differential.')
