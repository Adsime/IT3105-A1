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
        self.do_training(self.cman.get_training_cases())
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

    def do_training(self, cases, continued=False):
        if not self.current_session:
            print("No active session detected. Please make sure to call method: training_session before using method: "
                  "do_training")
            exit(0)
        if not continued:
            pass
            #self.session_tracker.reset()
        minibatch_size = self.options.minibatch_size
        n_cases = len(cases)
        l_grab_vars = [self.error, self.output]
        n_batches = math.ceil(n_cases / minibatch_size)
        batch_start = 0
        batch_end = self.options.minibatch_size
        for i in range(self.options.steps):
            step = self.global_training_step + i
            if batch_end >= len(cases):
                batch_end = batch_end % len(cases)
                minibatch = cases[batch_start:] + cases[:batch_end]
            else:
                minibatch = cases[batch_start:batch_end]
            result = self.run_step(self.trainer, l_grab_vars, self.generate_feeder(minibatch))
            error = result[1][0]
            self.session_tracker.error_tracker.gather_data(step, error, self, self.cman)
            batch_start = batch_end
            batch_end = batch_end + self.options.minibatch_size

    def run_step(self, operators, grabbed_vars=None, feed_dict=None):
        return self.current_session.run([operators, grabbed_vars], feed_dict=feed_dict)

    def one_hots_to_ints(self, cases):
        return [one_hot_to_int(i[1]) for i in cases]

    def generate_feeder(self, cases):
        x, y = np.transpose(cases)
        return {self.input: x.tolist(), self.target: y.tolist()}

    def generate_hit_counter(self, cases, k=1):
        correct = tf.nn.in_top_k(tf.cast(self.predictor, tf.float32), self.one_hots_to_ints(cases), k)
        return tf.reduce_sum(tf.cast(correct, tf.int32))

    def consider_validation_testing(self, epoch):
        v_err = 1 - self.do_validation()
        self.session_tracker.append_error(epoch, v_err, self.session_tracker.v_err)

    def do_validation(self):
        cases = self.cman.get_validation_cases()
        res = self.do_testing(cases)
        return res

    def get_top_k_error(self, cases, k):
        feeder = self.generate_feeder(cases)
        res = self.run_step(self.generate_hit_counter(cases, k), [], feeder)
        return 1 - (res[0]/len(cases))

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


