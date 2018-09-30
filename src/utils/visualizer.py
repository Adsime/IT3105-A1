from matplotlib import pyplot as plt
from src.utils.sessiontracker import SessionTracker
from src.gui.window import Window
from src.gui.dendrogram import DendrogramFrame
from src.gui.error import ErrorFrame
from src.gui.hinton import HintonFrame
from src.utils.options import ANNOptions as Options

class Visualizer():

    def __init__(self, session_tracker: SessionTracker, options: Options):

        plt.ion()
        self.row, self.column = [0, 0]
        self.session_tracker = session_tracker
        self.options = options
        self.window = Window()

    def insert_error_frame(self):
        self.insert_frame(ErrorFrame(self.session_tracker.error_tracker, self.window.window, [self.row, self.column]))

    def insert_hinton_frame(self, layers='all'):
        if not layers:
            print("No layers specified for hinton tracking. No window will show.")
            return
        elif layers == 'all':
            layers = [i for i in range(len(self.options.net_dims) - 1)]
        else:
            invalid = []
            for i in layers:
                if i < 0 or i >= len(self.options.net_dims) - 1:
                    invalid.append(i)
            if len(invalid) > 0:
                print("Invalid values selected for hinton plot: " + invalid.__str__()
                      + ".\nAvailable layers are: " + [i for i in range(len(self.options.net_dims) - 1)].__str__())
                return

        self.insert_frame(HintonFrame(self.session_tracker, self.window.window, [self.row, self.column], layers))

    def insert_dendro_frame(self):
        self.insert_frame(DendrogramFrame(self.session_tracker, self.window.window, [self.row, self.column]))

    def insert_frame(self, frame):
        self.window.add_frame(frame)
        self.increment()

    def increment(self):
        self.row += 1 if self.column == 1 else 0
        self.column = self.column + 1 if self.column < 1 else 0

    def start(self):
        self.window.start()

