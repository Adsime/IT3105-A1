import matplotlib
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
import tkinter as tk
from src.utils.sessiontracker import SessionTracker
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.backends.backend_agg import FigureCanvasAgg

import src.gui as gui
from gui.window import Window
from gui.dendrogram import DendrogramFrame
from gui.error import ErrorFrame
from gui.hinton import HintonFrame

class Visualizer():

    def __init__(self, session_tracker: SessionTracker):

        # Init window
        """self.window = tk.Tk()
        self.window.title("AI programmering")
        self.canvas = tk.Canvas(self.window, width=500, height=500)
        self.canvas.pack()

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)
        self.ax.set_ylabel("Error")
        self.ax.set_xlabel("Mini batch")
        """
        # [[case, target], [case, target]]

        plt.ion()

        self.row, self.column = [0, 0]

        self.session_tracker = session_tracker

        self.window = Window()

        self.insert_error_frame()
        #self.insert_hinton_frame()
        self.insert_dendro_frame()

        self.window.start()

    def insert_error_frame(self):
        self.insert_frame(ErrorFrame(self.session_tracker, self.window.window, [self.row, self.column]))

    def insert_hinton_frame(self):
        self.insert_frame(HintonFrame(self.session_tracker, self.window.window, [self.row, self.column]))

    def insert_dendro_frame(self):
        self.insert_frame(DendrogramFrame(self.session_tracker, self.window.window, [self.row, self.column]))

    def insert_frame(self, frame):
        self.window.add_frame(frame)
        self.increment()

    def increment(self):
        self.row += 1 if self.column == 1 else 0
        self.column = self.column + 1 if self.column < 1 else 0


    def add_figure(self, figure, loc=(0, 0)):
        canvas = FigureCanvasTkAgg(figure, master=self.window)
        canvas.get_tk_widget().grid(row=loc[0], column=loc[1])
        b = tk.Button(master=self.window, text="CLICK!", command=self.session_tracker.delete)
        b.grid(row=loc[0], column=loc[1])
        return canvas

