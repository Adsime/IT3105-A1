from src.utils.tflowtools import *
from src.utils.tutor1 import *
from src.gann.gann import Gann as G, Options
from src.utils.tutor2 import *
from src.utils.tutor3 import *
from src.utils.casemanager import DefaultCaseManager

case_func = (lambda: TFT.gen_all_one_hot_cases(12))
options = Options([12, 8, 12], tf.nn.relu, tf.reduce_min, tf.train.AdamOptimizer, DefaultCaseManager(case_func))
#Gann(options).do_training()
#autoex()
#print(int_to_one_hot(5, 9))
#autoex()
#autoex1()
G(options).run()
