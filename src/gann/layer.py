import tensorflow as tf
from src.utils.options import ANNOptions as Options
import numpy as np


class GannLayer:

    def __init__(self, net, options: Options, index, input_var, in_count, out_count):
        self.options = options
        self.net = net
        self.index = index
        self.input_var = input_var
        self.in_count = in_count
        self.out_count = out_count
        self.name = "Layer-" + index.__str__()

        # Avoid warnings
        #self.weights = None
        #self.biases = None
        #self.output = None

        # Build layer
        self.build_layer()

    def build_layer(self):
        w_range = self.options.weight_range
        self.weights = tf.Variable(np.random.uniform(w_range[0], w_range[1], size=(self.in_count, self.out_count)),
                                   name=self.name + '-wgt', trainable=True)  # True = default for trainable anyway
        self.biases = tf.Variable(np.random.uniform(w_range[0], w_range[1], size=self.out_count),
                                  name=self.name + '-bias', trainable=True)  # First bias vector
        self.output = tf.nn.relu(tf.matmul(self.input_var, self.weights) + self.biases, name=self.name + '-out')

    def get_output(self):
        return self.output
