from src.gui.frame import *
from src.utils.tflowtools import *


class HintonFrame(Frame):

    def __init__(self, session_tracker: SessionTracker, window, location, layers):
        Frame.__init__(self, session_tracker, window, location, "Hinton", "Weight", "Case")
        self.layers = layers
        self.layer = layers[0]
        self.colors = ['gray','red','blue','white']
        self.build()

    def update(self, override=False):
        if self.data_tracker.hinton_updated or override:
            self.clear()
            data = self.data_tracker.hinton[self.layer]
            self.update_title()
            self.ax = self.get_hinton_plot(data, self.ax)
            self.draw()
            self.data_tracker.hinton_updated = False

    def get_hinton_plot(self, data, ax):
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

    def build(self):
        fig = Figure()
        self.ax = fig.gca()

        self.canvas = FigureCanvasTkAgg(fig, master=self.box)
        self.canvas.get_tk_widget().pack(side=tk.TOP)

        buttonBox = tk.Frame(master=self.box)
        buttonBox.pack(side=tk.BOTTOM)


        self.next_button = tk.Button(master=buttonBox, text="Next", command=lambda : self.change_layer(next=True))
        self.next_button.pack(side=tk.RIGHT, padx=10)
        self.prev_button = tk.Button(master=buttonBox, text="Prev", command=lambda : self.change_layer(next=False))
        self.prev_button.pack(side=tk.LEFT, padx=10)
        self.draw()

    def update_title(self):
        self.set_title("Hinton (Layer: " + self.layer.__str__() + ")")

    def change_layer(self, next=True):
        currentIndex = self.layers.index(self.layer)
        if next and currentIndex + 1 < len(self.layers):
            self.layer = self.layers[currentIndex + 1]
        elif not next and currentIndex > 0:
            self.layer = self.layers[currentIndex - 1]
        else:
            return
        self.update(True)

