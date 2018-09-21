from src.gui.frame import *


class ErrorFrame(Frame):

    def __init__(self, session_tracker: SessionTracker, window, location):
        Frame.__init__(self, session_tracker, window, location, "Error plot", "Minibatch", "Error")
        self.build()

    def update(self, override=False):
        if self.session_tracker.error_updated:
            data = self.session_tracker.history
            labels = []
            self.clear()
            for set in data:
                x = data[set][0]
                y = data[set][1]
                self.ax.plot(x, y)
                labels.append(set)
                self.ax.legend(labels)
            self.draw()
            self.session_tracker.error = False

    def build(self):
        fig = Figure()
        self.ax = fig.add_subplot(111)

        self.canvas = FigureCanvasTkAgg(fig, master=self.box)
        self.canvas.get_tk_widget().pack(side=tk.TOP)
        self.clear()



