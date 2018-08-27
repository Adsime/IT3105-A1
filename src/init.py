from src.utils.tflowtools import *
from src.utils.tutor1 import *
from src.gann.gann import Gann, Options
options = Options([28**2, 8, 10], 0.001, tf.nn.relu, tf.reduce_min, tf.train.AdamOptimizer)
Gann(options).run()