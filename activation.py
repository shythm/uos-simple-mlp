# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #3

import numpy as np

# step activation function
def step(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0, 1, 0)

# step activation differential (let's assume it is 1)
def step_diff(x: np.ndarray) -> np.ndarray:
    return np.ones(x.shape)

# get activation function by name
def get_activation(name: str):
    if name == 'step':
        return step
    else:
        raise Exception('Unknown activation function.')
    
# get activation differential by name
def get_differential(name: str):
    if name == 'step':
        return step_diff
    else:
        raise Exception('Unknown activation function differential.')
