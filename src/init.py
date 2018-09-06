from src.utils.tflowtools import *
from src.utils.tutor1 import *
from src.gann.gann import Gann as G, Options
from src.utils.tutor2 import *
from src.utils.tutor3 import *
from src.utils.casemanager import DefaultCaseManager, CustomCaseManager
from src.data.mnist_basics import *
from src.utils.dataHandler import *

options = Options([11, 9, 9], tf.train.AdagradOptimizer,
                  CustomCaseManager(get_data('wine'), None, vfrac=0.2, tfrac=0.1), learning_rate=0.001,
                  weight_range=[-.1, .1], epochs=100, cost_function=tf.losses.mean_squared_error)
G(options).run()
#autoex()

#print(get_data("mnist_training"))