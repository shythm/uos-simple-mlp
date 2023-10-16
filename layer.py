# 2019920037 컴퓨터과학부 이성호
# 2023-2 Artificial Intelligence Coding #3

## Requirements ##
# (1) Let the number of layers and the number of nodes per layer be variables.
#     -> This code can make an arbitrary number of layers and nodes per layer.
# (2) Implement the calculation of one layer as a function.
#     -> This code implements the calculation of one layer as a class.

import numpy as np
from activation import get_activation, get_differential

# Layer class
# It stores weights of perceptrons and provides forward and backward propagation.
# When it calculates the results, it performs matrix multiplication.
# This is for making it easy to maintain weights and fast to calculate.
class Layer:
    
    def __init__(self, input_dim: int, output_dim: int, 
                 activation_name: str, learning_rate: float):
        """
        Initializes a new instance of the Layer class.

        Args:
            input_dim (int): The input dimension of the layer.
            output_dim (int): The output dimension of the layer.
            activation_name (str): The name of the activation function to use.
            learning_rate (float): The learning rate of the layer.
        """
        self._input_dim = input_dim
        self._output_dim = output_dim
        self._activation_name = activation_name
        self._learning_rate = learning_rate

        # set activation and differential functions
        self._activation = get_activation(activation_name)
        self._differential = get_differential(activation_name)

        # initialize weights randomly, input_dim + 1 is for bias
        # weights are stored in a matrix, "each row" will be weights of a perceptron
        self._weights = np.random.randn(output_dim, input_dim + 1)
    
    # weights getter
    @property
    def weights(self) -> np.ndarray:
        return self._weights
    
    # weights setter
    @weights.setter
    def weights(self, weights: np.ndarray) -> None:
        # validate dimension of weights
        if weights.shape != self.weights.shape:
            raise ValueError("The dimension of the weights is not valid.")
        
        self._weights = weights

    # input vector getter
    @property
    def invec(self) -> np.ndarray:
        return self._invec
    
    # input vector setter
    @invec.setter
    def invec(self, invec: np.ndarray) -> None:
        """
        Sets the input vector of this layer.
        
        Args:
            invec (np.ndarray): The input vector.
        """
        # validate dimension of input vector
        if invec.shape != (self._input_dim, 1):
            raise ValueError("The dimension of the input vector is not valid.")
        
        # add bias and set input vector
        self._invec = np.append([[1]], invec, axis=0)

    # output vector getter (forward propagation)
    @property
    def outvec(self) -> np.ndarray:
        """
        Calculates and gets the output of this layer.

        Returns:
            np.ndarray: The output vector.
        """
        # calculate previous activation vector (matrix multiplication)
        self._pre_activation = self.weights @ self.invec
        # calculate output
        return self._activation(self._pre_activation).reshape(self._output_dim, 1)
    
    # forward propagation
    def forward(self, invec: np.ndarray) -> np.ndarray:
        """
        Performs forward propagation.

        Args:
            invec (np.ndarray): The input vector.

        Returns:
            np.ndarray: The output vector.
        """
        # set input vector
        self.invec = invec
        # return output vector
        return self.outvec

    # error vector getter (backward propagation)
    @property
    def error(self) -> np.ndarray:
        """
        Gets the error of this layer.
        This error vector will be propagated to the previous layer.
        
        Returns:
            np.ndarray: The error vector.
        """

        # to update the previous layer, we need to calculate the error of this layer.
        # the error of input node is as follows.
        # ei = prev_error * a'(zi) * sum(wij * dj)
        #    = delta * sum(wij * dj)
        # -> where zi is the ith pre-activation value of this layer
        # -> where dj is the jth delta of this layer
        return self.weights.T[1:] @ self._delta # remove bias

    # error vector setter (backward propagation)
    @error.setter
    def error(self, error: np.ndarray) -> None:
        """
        Sets the previous error vector which is from the next layer.
        If the next layer propagates the error vector to this layer,
        it will calculate the delta of this layer.
        
        Args:
            error (np.ndarray): The previous error vector.
        """
        # validate dimension of error which is propagated from the "next layer"
        if error.shape != (self._output_dim, 1):
            raise ValueError(
                "The dimension of the error is not valid."
                f"It must be ({self._output_dim}, 1). It receives {error.shape}"
            )

        # calculate delta of this layer
        # delta = dL/dz = dL/da * da/dz = error * a'(z)
        #   -> where a is a result of activation function which forwarded to the next layer
        #   -> whose dimension will be (output_dim x 1)
        self._delta = error * self._differential(self._pre_activation)

    # update weights
    def _update(self) -> None:
        # Let's see forward propagation of the ith perceptron.
        # Output Oi = activation(wi * x) where x is the input vector.
        # Then, the differential of the output Oi can be calculated as follows.
        # dOi/dwi = [dOi/dwi1, dOi/dwi2, ..., dOi/dwij, ..., dOi/dwin]
        #   where j is the index of the weight and n is the dimension of the input vector.
        # dOi/dwij = dOi/da * da/dzi * dzi/dwij
        #   where a is the activation function, zi is ith perceptron's pre-activation value.
        #   (zi = wi1 * x1 + wi2 * x2 + ... + wij * xj + ... + win * xn)
        # Therefore, dOi/dwij = 1 * a'(zi) * xj
        # Hence, dOi/dwi = a'(zi)[x1, x2, ..., xn]

        # calculate delta of weight to perform gradient descent
        delta_weight = (self._delta @ self.invec.T) * self._learning_rate

        # update weights
        self.weights -= delta_weight

    # backward propagation
    def backward(self, error: np.ndarray) -> None:
        """
        Performs backpropagation.

        Args:
            error (np.ndarray): The previous error vector.
        """
        # set error vector
        self.error = error
        # update weights
        self._update()
        # return error vector to propagate to the previous layer
        return self.error
