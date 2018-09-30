from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.utils.sessiontracker import SessionTracker, ErrorTracker
import tkinter as tk
import matplotlib.pyplot as plt
import numpy as np

class Frame():

    def __init__(self, data_tracker, window, location, title, xlabel, ylabel):
        self.data_tracker = data_tracker
        self.box = tk.Frame(master=window)
        self.window = window
        self.box.grid(row=location[0], column=location[1])
        self.canvas = None
        self.ax = None
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel

    def set_title(self, title):
        self.title = title

    def update(self, override=False):
        raise NotImplementedError("Could not find 'update' implementation of Frame")

    def build(self):
        raise NotImplementedError("Could not find 'build' implementation of Frame")

    def clear(self):
        self.ax.clear()

    def draw(self):
        self.ax.set_title(self.title)
        self.ax.set_xlabel(self.xlabel)
        self.ax.set_ylabel(self.ylabel)
        self.canvas.draw()