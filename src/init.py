from src.gann.gann import Gann as G, Options, GannThread as GT
from src.utils.casemanager import DefaultCaseManager, CustomCaseManager
from src.utils.dataHandler import *
from src.utils.visualizer import Visualizer
from src.utils.sessiontracker import SessionTracker



lrate = 0.001
optimizer = \
    tf.train.AdamOptimizer(lrate, 0.9, 0.999)
    #tf.train.GradientDescentOptimizer(lrate)
session_tracker = SessionTracker()

cases = get_data('wine')

params = {
    'net_dims': [len(cases[0][0]), 100, 20, len(cases[0][1])],
    'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
    'case_manager': CustomCaseManager(cases, None, vfrac=0.1, tfrac=0.1),
    'steps': 20000,
    'minibatch_size': 10,
    'learning_rate': 0.001,
    'weight_range': [-.1, .1],
    'vint': 10,
    'h_activation_function': tf.nn.relu,
    'o_activation_function': tf.nn.softmax,
    'cost_function': tf.losses.mean_squared_error,
    'session_tracker': session_tracker
}

options = Options(**params)
print(options.net_dims)
#G(options).run()
GT(options)
Visualizer(session_tracker)

#autoex()

#print(get_data("mnist_training"))
