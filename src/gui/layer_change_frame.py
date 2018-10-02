from src.gui.frame import *
from src.utils.tflowtools import *

class LayerChangeFrame(Frame):

    def __init__(self, session_tracker: SessionTracker, window, location, layers, title, ylabel, xlabel):
        Frame.__init__(self, session_tracker, window, location, title, ylabel, xlabel)
        self.layers = layers
        self.layer = layers[0]
        pass

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

    def update_title(self, title):
        self.set_title(title + " (Layer: " + self.layer.__str__() + ")")

    def change_layer(self, next=True):
        currentIndex = self.layers.index(self.layer)
        if next and currentIndex + 1 < len(self.layers):
            self.layer = self.layers[currentIndex + 1]
        elif not next and currentIndex > 0:
            self.layer = self.layers[currentIndex - 1]
        else:
            return
        self.update(True)