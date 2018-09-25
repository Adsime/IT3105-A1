from src.gui.frame import *
import scipy.cluster.hierarchy as sch  # Needed for dendrograms

class DendrogramFrame(Frame):
    def __init__(self, session_tracker: SessionTracker, window, location):
        Frame.__init__(self, session_tracker, window, location, "Dendrogram", "Stuff", "stuff2")
        self.build()

    def update(self, override=False):
        if self.session_tracker.dendro_updated:
            data = self.session_tracker.dendro
            labels = []
            self.clear()
            self.dendrogram(data[0])
            self.draw()
            self.session_tracker.error = False

    def dendrogram(self, features, mode='average', metric='euclidean', orient='top', lrot=90.0):
        cluster_history = sch.linkage(features, method=mode, metric=metric)
        sch.dendrogram(cluster_history, orientation=orient, leaf_rotation=lrot, ax=self.ax)
        self.ax.set_ylabel(metric + " distance")


    def build(self):
        fig = Figure()
        self.ax = fig.gca()

        self.canvas = FigureCanvasTkAgg(fig, master=self.box)
        self.canvas.get_tk_widget().pack(side=tk.TOP)
        self.draw()