from src.gui.layer_change_frame import *
import scipy.cluster.hierarchy as sch  # Needed for dendrograms


class DendrogramFrame(LayerChangeFrame):
    def __init__(self, session_tracker: SessionTracker, window, location, layers):
        LayerChangeFrame.__init__(self, session_tracker, window, location, layers, "Dendrogram", "Label", "Distance")
        self.build()

    def update(self, override=False):
        data = self.data_tracker.get_dendro_data()
        if (self.data_tracker.dendro_data_updated or override) and len(data) > 0:
            self.clear()
            self.update_title("Dendrogram")
            self.dendrogram(data[0][self.layer])
            self.draw()
            self.data_tracker.dendro_updated = False

    def dendrogram(self, features, mode='average', metric='euclidean', orient='top', lrot=90.0):
        cluster_history = sch.linkage(features, method=mode, metric=metric)
        sch.dendrogram(cluster_history, orientation=orient, leaf_rotation=lrot, ax=self.ax,
                       leaf_label_func=lambda x: "C: " + self.data_tracker.get_dendro_data()[1][x].__str__())
        self.ylabel=(metric + " distance")