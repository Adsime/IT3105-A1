import tensorflow as tf
from src.utils.tflowtools import *
from src.gann.layer import GannLayer as Layer
from src.utils.options import ANNOptions as Options


class Gann:

    def __init__(self, options: Options):
        self.options = options  # Holds all the user defined options
        self.net_dims = options.net_dims
        self.layers = []

        # Avoid warnings
        self.input = None
        self.output = None
        self.predictor = None
        self.error = None
        self.trainer = None
        self.target = None

        # Build ANN
        self.build_net()

    # Instantiates the neural net, defining each layer by the user specifications in options
    def build_net(self):
        tf.reset_default_graph()    # Useful for multiple runs
        in_count = self.net_dims[0]
        self.input = tf.placeholder(tf.float64, shape=(None, in_count), name='Input')
        in_iter = self.input
        for i, out_count in enumerate(self.net_dims[1:]):
            print(i, out_count)
            layer = Layer(self, self.options, i, in_iter, in_count, out_count)
            in_iter = layer.get_output()
            in_count = out_count
            self.add_layer(layer)
        self.output = in_iter
        self.target = tf.placeholder(tf.float64, shape=(None, out_count), name='Target')
        self.set_learning_options()

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def set_learning_options(self):
        self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = self.options.optimizer(self.options.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    def run(self):
        session = gen_initialized_session()
        close_session(session)





