import tensorflow as tf
from src.utils.tflowtools import *



class Gann():

    def __init__(self, net_dims):
        self.net_dims = net_dims
        self.build_net()


    def build_net(self):
        tf.reset_default_graph()    # Useful for multiple runs
        node_count = self.net_dims[0]
        self.input = tf.placeholder(tf.float64, shape=(None, node_count), name='Input')
        in_iter = self.input
        for i, output in enumerate(self.net_dims[1:]):

