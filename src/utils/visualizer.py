from matplotlib import pyplot as plt
from src.utils.sessiontracker import SessionTracker
from src.gui.window import Window
from src.gui.dendrogram import DendrogramFrame
from src.gui.error import ErrorFrame
from src.gui.hinton import HintonFrame
from src.utils.options import ANNOptions as Options

class Visualizer():

    def __init__(self, session_tracker: SessionTracker, options: Options, flat_plotting=False):

        plt.ion()
        self.flat_plotting = flat_plotting
        self.row, self.column = [0, 0]
        self.session_tracker = session_tracker
        self.options = options
        self.window = Window()

    def insert_error_frame(self):
        self.insert_frame(ErrorFrame(self.session_tracker.error_tracker, self.window.window, [self.row, self.column]))

    def insert_output_hinton_frame(self, layers='all'):
        layers = self.check_layer_input(layers, "Output", 0)
        if not layers:
            return
        self.insert_frame(HintonFrame(self.session_tracker, self.window.window,
                                      [self.row, self.column], layers, "Outputs", "Layer output", "Case",
                                      self.session_tracker.get_output_data, self.session_tracker.output_data_updated))

    def insert_weight_hinton_frame(self, layers='all'):
        layers = self.check_layer_input(layers, "Weight")
        if not layers:
            return
        self.insert_frame(HintonFrame(self.session_tracker, self.window.window,
                                      [self.row, self.column], layers, "Weights", "", "",
                                      self.session_tracker.get_weight_data, self.session_tracker.weight_data_updated))

    def insert_bias_hinton_frame(self, layers='all'):
        layers = self.check_layer_input(layers, "Bias")
        if not layers:
            return
        self.insert_frame(HintonFrame(self.session_tracker, self.window.window,
                                      [self.row, self.column], layers, "Biases", "", "",
                                      self.session_tracker.get_bias_data, self.session_tracker.bias_data_updated))

    def insert_dendro_frame(self, layers='all'):
        layers = self.check_layer_input(layers, "hinton")
        if not layers:
            return
        self.insert_frame(DendrogramFrame(self.session_tracker, self.window.window, [self.row, self.column], layers))

    def check_layer_input(self, layers, name, max_index_correction=-1):
        if not layers:
            print("No layers specified for " + name + " tracking. No window will show.")
            return None
        elif layers == 'all':
            layers = [i for i in range(len(self.options.net_dims) + max_index_correction)]
        else:
            invalid = []
            for i in layers:
                if i < 0 or i >= len(self.options.net_dims) + max_index_correction:
                    invalid.append(i)
            if len(invalid) > 0:
                print("Invalid values selected for " + name + " plot: " + invalid.__str__()
                      + ".\nAvailable layers are: " + [i for i in range(len(self.options.net_dims))].__str__())
                return None
        return layers

    def insert_frame(self, frame):
        self.window.add_frame(frame)
        self.increment()

    def increment(self):
        if not self.flat_plotting:
            self.row += 1 if self.column == 1 else 0
            self.column = self.column + 1 if self.column < 1 else 0
        else:
            self.column += 1

    def start(self):
        self.window.start()

