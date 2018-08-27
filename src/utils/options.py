import tensorflow as tf
from tensorflow.python.training import optimizer as Optimizer


class ANNOptions:

    def __init__(self, net_dims, learning_rate, activation_function, error_function, optimizer: Optimizer):
        self.net_dims = net_dims
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.error_function = error_function
        self.optimizer = optimizer
