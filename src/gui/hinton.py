from src.gui.frame import *
from src.gui.layer_change_frame import LayerChangeFrame
from src.utils.tflowtools import *


class HintonFrame(LayerChangeFrame):

    def __init__(self, session_tracker: SessionTracker, window, location, layers, title, xlabel, ylabel,
                 data_func, *updated):
        LayerChangeFrame.__init__(self, session_tracker, window, location, layers, title, xlabel, ylabel)
        self.updated = updated
        self.data_func = data_func
        self.name = title
        self.colors = ['gray','red','blue','white']
        self.build()

    def update(self, override=False):
        data = self.data_func()
        if (self.updated or override) and len(data) > 0:
            self.clear()
            data = data[self.layer]
            self.update_title(self.name)
            self.ax = self.get_hinton_plot(data, self.ax)
            self.draw()
            self.updated = False

    def get_hinton_plot(self, data, ax):
        data = np.array(data)
        colors = ['gray', 'red', 'blue', 'white']
        data = data.transpose()
        maxval = 1
        maxsize = 2 ** np.ceil(np.log(maxval) / np.log(2))

        ax.clear()
        ax.patch.set_facecolor(colors[0])  # This is the background color.  Hinton uses gray
        ax.set_aspect('auto',
                      'box')  # Options: ('equal'), ('equal','box'), ('auto'), ('auto','box')..see matplotlib docs
        ax.xaxis.set_major_locator(PLT.NullLocator())
        ax.yaxis.set_major_locator(PLT.NullLocator())

        ymax = (data.shape[1] - 1) * maxsize
        for (x, y), val in np.ndenumerate(data):
            color = colors[1] if val > 0 else colors[2]  # Hinton uses white = pos, black = neg
            size = max(0.01, np.sqrt(min(maxsize, maxsize * np.abs(val) / maxval)))
            bottom_left = [x - size / 2, (ymax - y) - size / 2]  # (ymax - y) to invert: row 0 at TOP of diagram
            blob = PLT.Rectangle(bottom_left, size, size, facecolor=color, edgecolor=colors[3])
            ax.add_patch(blob)
        ax.autoscale_view()
        return ax



