from matplotlib import pyplot as plt
from src.utils.sessiontracker import SessionTracker
from src.gui.window import Window
from src.gui.dendrogram import DendrogramFrame
from src.gui.error import ErrorFrame
from src.gui.hinton import HintonFrame
from src.gui.quantitative import QuantitativeFrame
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
        self.insert_frame(ErrorFrame(self.session_tracker.error_tracker, self.window.windows[-1], [self.row, self.column]))

    def insert_output_hinton_frame(self, layers='all'):
        layers = self.check_layer_input(layers, "Output", 0)
        if not layers:
            return
        params = self.construct_params(layers, "Output", "", "", self.session_tracker.get_output_data)
        frame = HintonFrame(self.session_tracker.output_data_updated, **params)
        self.insert_frame(frame)
        """self.insert_frame(HintonFrame(self.session_tracker.output_data_updated, self.session_tracker, self.window.windows[-1],
                                      [self.row, self.column], layers, "Outputs", "Layer output", "Case",
                                      self.session_tracker.get_output_data))"""

    def insert_weight_frame(self, layers='all', hinton=True):
        layers = self.check_layer_input(layers, "Weight")
        if not layers:
            return
        params = self.construct_params(layers, "Weights", "", "", self.session_tracker.get_weight_data)
        frame = HintonFrame(self.session_tracker.weight_data_updated, **params) if hinton else \
                QuantitativeFrame(self.session_tracker.weight_data_updated, **params)
        self.insert_frame(frame)
        """self.insert_frame(HintonFrame(self.session_tracker.weight_data_updated, self.session_tracker, self.window.windows[-1],
                                      [self.row, self.column], layers, "Weights", "", "",
                                      self.session_tracker.get_weight_data))"""

    def insert_bias_frame(self, layers='all', hinton=True):
        layers = self.check_layer_input(layers, "Bias")
        if not layers:
            return
        params = self.construct_params(layers, "Biases", "", "", self.session_tracker.get_bias_data)
        frame = HintonFrame(self.session_tracker.bias_data_updated, **params) if hinton else \
                QuantitativeFrame(self.session_tracker.bias_data_updated, **params)
        self.insert_frame(frame)

    def insert_dendro_frame(self, layers='all'):
        layers = self.check_layer_input(layers, "hinton")
        if not layers:
            return
        self.insert_frame(DendrogramFrame(self.session_tracker, self.window.windows[-1], [self.row, self.column], layers))

    def construct_params(self, layers, title, xlabel, ylabel, data_func):
        params = {
            'session_tracker': self.session_tracker,
            'window': self.window.windows[-1],
            'location': [self.row, self.column],
            'layers': layers,
            'title': title,
            'xlabel': xlabel,
            'ylabel': ylabel,
            'data_func': data_func
        }
        return params

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
            self.column += 1
            if self.column >= 2:
                self.window.add_window()
                self.column = 0

    def start(self):
        self.window.start()

