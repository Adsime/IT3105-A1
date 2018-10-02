from src.gann.gann import Gann
from src.utils.options import ANNOptions as Options
from src.utils.casemanager import CustomCaseManager
from src.utils.dataHandler import DataSets
import tensorflow as tf


def calc_net_dims(case, dims):
    return [len(case[0])] + dims + [len(case[1])]

class Scenarios:

    @staticmethod
    def get_mnist_options(session_tracker):
        cases = DataSets.Mnist()
        lrate = 0.001
        cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

        params = {
            'net_dims': calc_net_dims(cases[0], [80, 80]),
            'h_activation_function': tf.nn.relu,
            'o_activation_function': tf.nn.softmax,
            'cost_function': tf.losses.softmax_cross_entropy,
            'learning_rate': lrate,
            'weight_range': [-.1, .1],
            'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
            'case_manager': cman,
            'minibatch_size': 50,
            'steps': 1000,
            'vint': 10,
            'session_tracker': session_tracker,

            # MAP options
            'map_case_count': 10,
            'map_case_func': cman.get_testing_cases
        }

        # Options used throughout the program
        return Options(**params)

    @staticmethod
    def get_wine_options(session_tracker):
        cases = DataSets.Wine()
        lrate = 0.001
        cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

        params = {
            'net_dims': calc_net_dims(cases[0], [80, 80]),
            'h_activation_function': tf.nn.relu,
            'o_activation_function': tf.nn.softmax,
            'cost_function': tf.losses.softmax_cross_entropy,
            'learning_rate': lrate,
            'weight_range': [-.5, .5],
            'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
            'case_manager': cman,
            'minibatch_size': 100,
            'steps': 1000,
            'vint': 10,
            'session_tracker': session_tracker,

            # MAP options
            'map_case_count': 10,
            'map_case_func': cman.get_testing_cases
        }

        # Options used throughout the program
        return Options(**params)

    @staticmethod
    def get_yeast_options(session_tracker):
        cases = DataSets.Yeast()
        lrate = 0.001
        cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

        params = {
            'net_dims': calc_net_dims(cases[0], [80, 80]),
            'h_activation_function': tf.nn.relu,
            'o_activation_function': tf.nn.softmax,
            'cost_function': tf.losses.softmax_cross_entropy,
            'learning_rate': lrate,
            'weight_range': [-.5, .5],
            'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
            'case_manager': cman,
            'minibatch_size': 256,
            'steps': 1000,
            'vint': 10,
            'session_tracker': session_tracker,

            # MAP options
            'map_case_count': 10,
            'map_case_func': cman.get_testing_cases
        }

        # Options used throughout the program
        return Options(**params)

    @staticmethod
    def get_glass_options(session_tracker):
        cases = DataSets.Glass()
        lrate = 0.001
        cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

        params = {
            'net_dims': calc_net_dims(cases[0], [80, 80]),
            'h_activation_function': tf.nn.relu,
            'o_activation_function': tf.nn.softmax,
            'cost_function': tf.losses.softmax_cross_entropy,
            'learning_rate': lrate,
            'weight_range': [-.5, .5],
            'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
            'case_manager': cman,
            'minibatch_size': 75,
            'steps': 1000,
            'vint': 10,
            'session_tracker': session_tracker,

            # MAP options
            'map_case_count': 10,
            'map_case_func': cman.get_testing_cases
        }

        # Options used throughout the program
        return Options(**params)

    @staticmethod
    def get_hackers_choice_options(session_tracker):
        cases = DataSets.Hackers_Choice()
        lrate = 0.001
        cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

        params = {
            'net_dims': calc_net_dims(cases[0], [80, 80]),
            'h_activation_function': tf.nn.relu,
            'o_activation_function': tf.nn.softmax,
            'cost_function': tf.losses.softmax_cross_entropy,
            'learning_rate': lrate,
            'weight_range': [-.1, .1],
            'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
            'case_manager': cman,
            'minibatch_size': 50,
            'steps': 1000,
            'vint': 10,
            'session_tracker': session_tracker,

            # MAP options
            'map_case_count': 10,
            'map_case_func': cman.get_testing_cases
        }

        # Options used throughout the program
        return Options(**params)

    @staticmethod
    def get_parity_options(session_tracker):
        cases = DataSets.Parity(10)
        lrate = 0.001
        cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

        params = {
            'net_dims': calc_net_dims(cases[0], [80, 80]),
            'h_activation_function': tf.nn.relu,
            'o_activation_function': tf.nn.softmax,
            'cost_function': tf.losses.softmax_cross_entropy,
            'learning_rate': lrate,
            'weight_range': [-.5, .5],
            'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
            'case_manager': cman,
            'minibatch_size': 256,
            'steps': 2000,
            'vint': 100,
            'session_tracker': session_tracker,

            # MAP options
            'map_case_count': 10,
            'map_case_func': cman.get_testing_cases
        }

        # Options used throughout the program
        return Options(**params)

    @staticmethod
    def get_symmetry_options(session_tracker):
        cases = DataSets.Symmetry(101, 2000)
        lrate = 0.001
        cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

        params = {
            'net_dims': calc_net_dims(cases[0], [80, 80]),
            'h_activation_function': tf.nn.relu,
            'o_activation_function': tf.nn.softmax,
            'cost_function': tf.losses.softmax_cross_entropy,
            'learning_rate': lrate,
            'weight_range': [-.5, .5],
            'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
            'case_manager': cman,
            'minibatch_size': 256,
            'steps': 2000,
            'vint': 100,
            'session_tracker': session_tracker,

            # MAP options
            'map_case_count': 10,
            'map_case_func': cman.get_testing_cases
        }

        # Options used throughout the program
        return Options(**params)

    @staticmethod
    def get_bit_counter_options(session_tracker):
        cases = DataSets.Bit_Counter(500, 15)
        lrate = 0.1
        cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

        params = {
            'net_dims': calc_net_dims(cases[0], [80, 80]),
            'h_activation_function': tf.nn.relu,
            'o_activation_function': tf.nn.softmax,
            'cost_function': tf.losses.mean_squared_error,
            'learning_rate': lrate,
            'weight_range': [-.2, .2],
            'optimizer': tf.train.AdagradOptimizer(lrate, 0.001),
            'case_manager': cman,
            'minibatch_size': 100,
            'steps': 10000,
            'vint': 100,
            'session_tracker': session_tracker,

            # MAP options
            'map_case_count': 10,
            'map_case_func': cman.get_testing_cases
        }

        # Options used throughout the program
        return Options(**params)

    @staticmethod
    def get_segment_counter_options(session_tracker):
        cases = DataSets.Segment_Counter(25, 1000, 0, 8)
        lrate = 0.001
        cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

        params = {
            'net_dims': calc_net_dims(cases[0], [80, 80]),
            'h_activation_function': tf.nn.relu,
            'o_activation_function': tf.nn.softmax,
            'cost_function': tf.losses.softmax_cross_entropy,
            'learning_rate': lrate,
            'weight_range': [-.5, .5],
            'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
            'case_manager': cman,
            'minibatch_size': 256,
            'steps': 2000,
            'vint': 100,
            'session_tracker': session_tracker,

            # MAP options
            'map_case_count': 10,
            'map_case_func': cman.get_testing_cases
        }

        # Options used throughout the program
        return Options(**params)


