from src.gann.gann import Gann as G, Options, GannThread as GT
from src.utils.casemanager import DefaultCaseManager, CustomCaseManager
from src.utils.dataHandler import *
from src.utils.visualizer import Visualizer
from src.utils.sessiontracker import SessionTracker
from src.presets.scenarios import Scenarios

session_tracker = SessionTracker()

"""
cases = DataSets.Yeast()
lrate = 0.001
optimizer = tf.train.AdagradOptimizer(lrate)
cman = CustomCaseManager(cases, None, 1, .1, .1)
options = Scenarios.get_custom_options([80, 80], tf.nn.relu, tf.nn.softmax, tf.losses.softmax_cross_entropy,
                                       lrate, [-.5, .5], optimizer, 10, 1000, 10, session_tracker, cman, 10,
                                       cman.get_testing_cases)
"""


options = Scenarios.get_mnist_options(session_tracker)
gann = G(options)
GT(gann.generate_full_run_sequence())

# Visualization tool
v = Visualizer(session_tracker, options, False)
v.insert_error_frame()
v.insert_dendro_frame(layers='all')
v.insert_output_hinton_frame(layers='all')
v.insert_bias_frame(hinton=False)
v.insert_weight_frame(hinton=False)
v.insert_bias_frame()
v.insert_weight_frame()
v.start()

#autoex()

#print(get_data("mnist_training"))
