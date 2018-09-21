from src.gann.gann import Gann as G, Options, GannThread as GT
from src.utils.casemanager import DefaultCaseManager, CustomCaseManager
from src.utils.dataHandler import *
from src.utils.visualizer import Visualizer
from src.utils.sessiontracker import SessionTracker


session_tracker = SessionTracker()

cases = get_data('wine', mean_std)
lrate = 0.001

params = {
    'net_dims': [len(cases[0][0]), 10, 30, 80, 20, 10, len(cases[0][1])],
    'h_activation_function': tf.nn.relu,
    'o_activation_function': tf.nn.softmax,
    'cost_function': tf.losses.mean_squared_error,
    'learning_rate': lrate,
    'weight_range': [-.1, .1],
    'optimizer': tf.train.AdamOptimizer(lrate, 0.9, 0.999),
    'case_manager': CustomCaseManager(cases, None, cfrac=.1, vfrac=0.1, tfrac=0.1),
    'minibatch_size': 1000,
    'steps': 1000,
    'vint': 100,
    'session_tracker': session_tracker
}

options = Options(**params)
print(options.net_dims)
gann = G(options)
GT(gann)
Visualizer(session_tracker)

#autoex()

#print(get_data("mnist_training"))
