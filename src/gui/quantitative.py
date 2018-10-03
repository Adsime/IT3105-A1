from src.gui.frame import *
from src.gui.layer_change_frame import LayerChangeFrame
from src.utils.tflowtools import *


class QuantitativeFrame(LayerChangeFrame):

    def __init__(self, *updated, session_tracker: SessionTracker, window, location, layers, title, xlabel, ylabel,
                 data_func):
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
            self.ax = self.get_quantitative_plot(data, self.ax)
            self.draw()
            self.updated = False

    def get_quantitative_plot(self, data, ax, cutoff=0.1, tsize=12, tform='{:.3f}'):
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
            if val > 0:
                color = colors[0] if val > cutoff else colors[1]
            else:
                color = colors[3] if val < -cutoff else colors[2]
            botleft = [x - 1 / 2, (ymax - y) - 1 / 2]  # (ymax - y) to invert: row 0 at TOP of diagram
            # This is a hack, but I seem to need to add these blank blob rectangles first, and then I can add the text
            # boxes.  If I omit the blobs, I get just one plotted textbox...grrrrrr.
            blob = PLT.Rectangle(botleft, 1, 1, facecolor='white', edgecolor='white')
            ax.add_patch(blob)
            ax.text(botleft[0] + 0.5, botleft[1] + 0.5, tform.format(val),
                      bbox=dict(facecolor=color, alpha=0.5, edgecolor='white'), ha='center', va='center',
                      color='black', size=tsize)
        ax.autoscale_view()
        return ax



