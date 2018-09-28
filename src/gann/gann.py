import tensorflow as tf
from src.utils.tflowtools import *
from src.gann.layer import GannLayer as Layer
from src.utils.options import ANNOptions as Options
from src.utils.sessiontracker import SessionTracker
from src.utils.visualizer import *
from threading import Thread
import numpy as np

class Gann:

    def __init__(self, options: Options):
        self.options = options  # Holds all the user defined options
        self.net_dims = options.net_dims
        self.layers = []
        self.session_tracker = options.session_tracker
        self.global_training_step = 0
        self.cman = self.options.case_manager
        self.optimizer = options.optimizer


        # Avoid warnings
        self.input = None
        self.target = None
        self.output = None
        self.predictor = None
        self.error = None
        self.trainer = None
        self.current_session = None

    # Instantiates the neural net, defining each layer by the user specifications in options
    def build_net(self):
        tf.reset_default_graph()    # Useful for multiple runs
        self.input = tf.placeholder(tf.float64, shape=(None, self.net_dims[0]), name='Input')
        self.target = tf.placeholder(tf.float64, shape=(None, self.net_dims[-1]), name='Target')
        in_count = self.net_dims[0]
        in_iter = self.input
        for i, out_count in enumerate(self.net_dims[1:]):
            layer = Layer(self, self.options, i, in_iter, in_count, out_count)
            in_iter = layer.get_output()
            in_count = out_count
            self.add_layer(layer)
        self.output = self.options.o_activation_function(layer.output)
        self.set_learning_options()

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def set_learning_options(self):
        self.error = self.options.cost_function(self.target, self.output)
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        self.trainer = self.optimizer.minimize(self.error, name='Backprop')

    # Session methods

    def training_session(self, session=None, continued=False):
        self.current_session = session if session else gen_initialized_session()
        self.do_training()
        pass

    def validation_session(self):
        pass

    def testing_session(self):
        cases = self.cman.get_testing_cases()
        res = self.do_testing(cases)
        print("Test result: " + str(round(res * 100, 2)) + "%")
        cases = self.cman.get_training_cases()
        res = self.do_testing(cases)
        print("Training result: " + str(round(res * 100, 2)) + "%")

    def mapping_session(self):
        weights = []
        outputs = []
        for layer in self.layers:
            weights.append(layer.weights)
            outputs.append(layer.output)
        g_vars = [weights, outputs]
        cases = self.cman.get_n_random_cases(10, self.cman.get_testing_cases())
        feeder = self.generate_feeder(cases)
        res = self.run_step(self.predictor, g_vars, feeder)
        self.session_tracker.set_hinton_data(res[1][0])
        self.session_tracker.set_dendro_data(res[1][1])

    # Methods for doing work in given sessions

    def do_training(self):
        l_grab_vars = [self.error]
        for i in range(self.options.steps):
            minibatch = self.cman.get_n_random_cases(self.options.minibatch_size, self.cman.get_training_cases())
            result = self.run_step(self.trainer, l_grab_vars, self.generate_feeder(minibatch))
            error = result[1][0]
            self.session_tracker.error_tracker.gather_data(i, error, self, self.cman)

    def run_step(self, operators, grabbed_vars=None, feed_dict=None):
        return self.current_session.run([operators, grabbed_vars], feed_dict=feed_dict)

    def one_hots_to_ints(self, cases):
        return [one_hot_to_int(i[1]) for i in cases]

    def generate_feeder(self, cases):
        x, y = [[row[i] for row in cases] for i in range(len(cases[0]))]
        return {self.input: x, self.target: y}

    def generate_hit_counter(self, cases, k=1):
        correct = tf.nn.in_top_k(tf.cast(self.predictor, tf.float32), self.one_hots_to_ints(cases), k)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def do_validation(self):
        cases = self.cman.get_validation_cases()
        res = self.do_testing(cases)
        return res

    def do_testing(self, cases):
        feeder = self.generate_feeder(cases)
        res = self.run_step(self.generate_hit_counter(cases), [], feeder)
        return 1 - (res[0]/len(cases))


    # Main methods. Called by user.

    def run(self):
        # Build ANN
        self.build_net()
        self.training_session()
        self.testing_session()
        #visualize(self.session_tracker.history)
        #close_session(self.current_session)


class GannThread(Thread):

    def __init__(self, gann: Gann):
        Thread.__init__(self)
        #self.methods = methods
        self.gann = gann
        self.start()

    def run(self):
        self.gann.build_net()
        self.gann.training_session()
        self.gann.testing_session()
        self.gann.mapping_session()
        #self.gann.run()
        #Gann(self.options).run()
        return


