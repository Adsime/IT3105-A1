from src.utils.tflowtools import *
from src.utils.tutor1 import *
from src.gann.gann import Gann as G, Options
from src.utils.tutor2 import *
from src.utils.tutor3 import *
from src.utils.casemanager import DefaultCaseManager, CustomCaseManager
from src.data.mnist_basics import *
from src.utils.fileReader import *

options = Options([11, 100, 20, 10], tf.nn.relu, tf.reduce_min, tf.train.AdamOptimizer,
                  CustomCaseManager(get_csv_cases('./data/wine.txt')), cost_function=tf.nn.softmax, learning_rate=0.001)
G(options).run()
#autoex()
