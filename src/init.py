from src.gann.gann import Gann as G, Options, GannThread as GT
from src.utils.casemanager import DefaultCaseManager, CustomCaseManager
from src.utils.dataHandler import *
from src.utils.visualizer import Visualizer
from src.utils.sessiontracker import SessionTracker
from src.presets.scenarios import Scenarios

session_tracker = SessionTracker()

"""
cases = DataSets.Parity(10)
cases = DataSets.Mnist()
lrate = 0.001

cman = CustomCaseManager(cases, None, cfrac=1, vfrac=0.1, tfrac=0.1)

params = {
    'net_dims': [len(cases[0][0]), 80, 80, len(cases[0][1])],
    'h_activation_function': tf.nn.relu,
    'o_activation_function': tf.nn.softmax,
    'cost_function': tf.losses.softmax_cross_entropy,
    'learning_rate': lrate,
    'weight_range': [-.1, .1],
    'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
    #'optimizer': tf.train.AdagradOptimizer(lrate, 0.001),
    #'optimizer': tf.train.GradientDescentOptimizer(lrate),
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
options = Options(**params)
"""

# Neural net init and run
#gann = G(options)

options = Scenarios.get_glass_options(session_tracker)
gann = G(options)
GT(gann.generate_full_run_sequence())

# Visualization tool
v = Visualizer(session_tracker, options, True)
v.insert_error_frame()
v.insert_dendro_frame(layers='all')
v.insert_output_hinton_frame(layers='all')
v.insert_bias_hinton_frame()
v.insert_weight_hinton_frame()
v.start()

#autoex()

#print(get_data("mnist_training"))
