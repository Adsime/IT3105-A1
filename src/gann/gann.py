import tensorflow as tf
from src.utils.tflowtools import *
from src.gann.layer import GannLayer as Layer
from src.utils.options import ANNOptions as Options
from src.utils.sessiontracker import SessionTracker


class Gann:

    def __init__(self, options: Options):
        self.options = options  # Holds all the user defined options
        self.net_dims = options.net_dims
        self.layers = []
        self.session_tracker = SessionTracker()
        self.global_training_step = 0


        # Avoid warnings
        self.input = None
        self.target = None
        self.output = None
        self.predictor = None
        self.error = None
        self.trainer = None
        self.current_session = None
        # Build ANN
        self.build_net()

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
        self.output = layer.output
        self.output = self.options.cost_function(self.output)
        self.set_learning_options()

    def add_layer(self, layer: Layer):
        self.layers.append(layer)

    def set_learning_options(self):
        self.error = tf.reduce_mean(tf.square(self.target - self.output), name='MSE')
        self.predictor = self.output  # Simple prediction runs will request the value of output neurons
        # Defining the training operator
        optimizer = self.options.optimizer(self.options.learning_rate)
        self.trainer = optimizer.minimize(self.error, name='Backprop')

    # Session methods

    def training_session(self, session=None, continued=False):
        self.current_session = session if session else gen_initialized_session()
        self.do_training(self.options.case_manager.get_training_cases())
        pass

    def validation_session(self):
        pass

    def testing_session(self):
        pass

    # Methods for doing work in given sessions

    def do_training(self, cases, continued=False):
        if not self.current_session:
            print("No active session detected. Please make sure to call method: training_session before using method: "
                  "do_training")
            exit(0)
        if not continued:
            self.session_tracker.reset()
        minibatch_size = self.options.minibatch_size
        n_cases = len(cases)
        for i in range(self.options.epochs):
            error = 0
            hits = 0
            step = self.global_training_step + i
            l_grab_vars = [self.error, self.output] + self.session_tracker.get_grab_variables()
            n_batches = math.ceil(n_cases/minibatch_size)
            for batch_start in range(0, n_cases, minibatch_size):
                batch_end = min(n_cases, batch_start+minibatch_size)
                minibatch = cases[batch_start:batch_end]   # Extract batch from cases
                inputs = [case[0] for case in minibatch]
                targets = [case[1] for case in minibatch]
                feeder = {self.input: inputs, self.target: targets}
                result = self.run_step(self.trainer, self.current_session,
                                       l_grab_vars, feeder)
                for res, tar in zip(result[1][1], targets):
                    hits += 1 if np.argmax(res) == np.argmax(tar) else 0
                #print("[" + batch_start.__str__() + ", " + batch_end.__str__() + "] - Error: " + result[1][0].__str__()
                #      + ". Output: " + result[1][1].__str__() + " " + np.sum(result[1][1][0]).__str__())
                error += result[1][0]
            print("Hit rate: " + (hits/n_cases).__str__())
            #print("Epoch: " + step.__str__() + " - Error: " + (error/n_batches).__str__())

    def run_step(self, operators, session, grabbed_vars=None, feed_dict=None):
        return session.run([operators, grabbed_vars], feed_dict=feed_dict)

    def do_validation(self):
        pass

    def do_testing(self):
        pass

    # Main methods. Called by user.

    def run(self):
        self.training_session()
        #close_session(self.current_session)
        exit()





