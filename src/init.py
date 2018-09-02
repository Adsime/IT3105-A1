from src.utils.tflowtools import *
from src.utils.tutor1 import *
from src.gann.gann import Gann as G, Options
from src.utils.tutor2 import *
from src.utils.tutor3 import *
from src.utils.casemanager import DefaultCaseManager, CustomCaseManager
from src.data.mnist_basics import *

options = Options([784, 100, 20, 10], tf.nn.relu, tf.reduce_min, tf.train.AdamOptimizer, CustomCaseManager(), cost_function=tf.nn.cross)
G(options).run()
autoex()