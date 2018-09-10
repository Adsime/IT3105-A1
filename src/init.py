from src.utils.tflowtools import *
from src.utils.tutor1 import *
from src.gann.gann import Gann as G, Options
from src.utils.tutor2 import *
from src.utils.tutor3 import *
from src.utils.casemanager import DefaultCaseManager, CustomCaseManager
from src.data.mnist_basics import *
from src.utils.dataHandler import *


lrate = 0.001
optimizer = \
    tf.train.AdamOptimizer(lrate, 0.9, 0.999)
    #tf.train.GradientDescentOptimizer(lrate)
cman = CustomCaseManager(get_data('wine'), None, vfrac=0.1, tfrac=0.1)

options = Options([11, 80, 80, 9], optimizer, cman,
                  weight_range=[-.1, .1], epochs=500, cost_function=tf.losses.mean_squared_error)
G(options).run()
#autoex()

#print(get_data("mnist_training"))
