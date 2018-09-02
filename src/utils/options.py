import tensorflow as tf
from tensorflow.python.training import optimizer as Optimizer
from src.utils.casemanager import CaseManager

class ANNOptions:

    def __init__(self, net_dims, activation_function, error_function, optimizer: Optimizer, case_manager: CaseManager,
                 epochs=1000, minibatch_size=10, learning_rate=0.001, cost_function=tf.nn.softmax):
        self.net_dims = net_dims
        self.learning_rate = learning_rate
        self.activation_function = activation_function
        self.error_function = error_function
        self.optimizer = optimizer
        self.epochs = epochs
        self.minibatch_size = minibatch_size
        self.case_manager = case_manager
        self.cost_function = cost_function
