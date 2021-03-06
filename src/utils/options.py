import tensorflow as tf
from tensorflow.python.training import optimizer as Optimizer
from src.utils.casemanager import CaseManager
from src.utils.sessiontracker import SessionTracker

class ANNOptions:

    def __init__(self, net_dims, optimizer: Optimizer, case_manager: CaseManager, session_tracker: SessionTracker,
                 map_case_func, steps=1000, minibatch_size=10, learning_rate=0.001, weight_range=[-1, 1],
                 vint=10, h_activation_function=tf.nn.relu, o_activation_function=tf.nn.softmax,
                 cost_function=tf.losses.mean_squared_error, map_case_count=10):
        self.net_dims = net_dims
        self.learning_rate = learning_rate
        self.h_activation_function = h_activation_function
        self.optimizer = optimizer
        self.steps = steps
        self.minibatch_size = minibatch_size
        self.case_manager = case_manager
        self.weight_range = weight_range
        self.o_activation_function = o_activation_function
        self.vint = vint
        self.cost_function = cost_function
        self.session_tracker = session_tracker
        self.map_case_count = map_case_count
        self.map_case_func = map_case_func
