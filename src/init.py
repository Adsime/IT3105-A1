from src.gann.gann import Gann as G, Options
from src.utils.casemanager import DefaultCaseManager, CustomCaseManager
from src.utils.dataHandler import *



lrate = 0.001
optimizer = \
    tf.train.AdamOptimizer(lrate, 0.9, 0.999)
    #tf.train.GradientDescentOptimizer(lrate)

cases = get_data('wine')
dims = [len(cases[0][0]), 80, 80, len(cases[0][1])]
cman = CustomCaseManager(cases, None, vfrac=0.1, tfrac=0.1)

options = Options(dims, optimizer, cman,
                  weight_range=[-.1, .1], steps=100000, cost_function=tf.losses.mean_squared_error)
G(options).run()
#autoex()

#print(get_data("mnist_training"))
