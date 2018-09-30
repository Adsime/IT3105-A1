from src.gui.frame import *
import scipy.cluster.hierarchy as sch  # Needed for dendrograms

class DendrogramFrame(Frame):
    def __init__(self, session_tracker: SessionTracker, window, location):
        Frame.__init__(self, session_tracker, window, location, "Dendrogram", "Label", "Distance")
        self.build()

    def update(self, override=False):
        if self.data_tracker.dendro_updated:
            data = self.data_tracker.dendro
            labels = []
            self.clear()
            self.dendrogram(data[0])
            self.draw()
            self.data_tracker.dendro_updated = False

    def dendrogram(self, features, mode='average', metric='euclidean', orient='top', lrot=90.0):
        cluster_history = sch.linkage(features, method=mode, metric=metric)
        sch.dendrogram(cluster_history, orientation=orient, leaf_rotation=lrot, ax=self.ax)
        self.ylabel= (metric + " distance")


    def build(self):
        fig = Figure()
        self.ax = fig.gca()

        self.canvas = FigureCanvasTkAgg(fig, master=self.box)
        self.canvas.get_tk_widget().pack(side=tk.TOP)
        self.draw()