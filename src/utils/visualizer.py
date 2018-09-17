import matplotlib
from matplotlib.figure import Figure
from matplotlib import pyplot as plt
import numpy as np
import tkinter as tk
from src.utils.sessiontracker import SessionTracker
import time
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

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

        plt.ion()

        self.session_tracker = session_tracker

        self.lines = {}
        for set in session_tracker.history:
            self.lines[set] = None

        self.window = tk.Tk()
        self.window.wm_title("Embedding in TK")

        self.fig = Figure()
        self.ax = self.fig.add_subplot(111)

        self.ax.set_title('Stuff and things')
        self.ax.set_xlabel('Mini batches')
        self.ax.set_ylabel('Error')

        # a tk.DrawingArea
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.show()

        self.canvas._tkcanvas.pack(side=tk.TOP, fill=tk.BOTH, expand=1)


        self.start()

    def visualize(self, data):
        """with plt.style.context('seaborn-colorblind'):
            print(plt.style.available)
            labels = []
            for set in data:
                x = data[set][0]
                y = data[set][1]
                plt.plot(x, y, label=set)
                labels.append(set)
            plt.legend(labels)
            plt.title('Stuff and things')
            plt.xlabel('Epoch')
            plt.ylabel('Error')
            plt.show()"""
        pass

    def draw(self, data):
        """for set in data:
            x = data[set][0]
            y = data[set][1]
            self.ax.plot(x, y, label=set)
        data = data['training_error']
        self.ax.plot(data[0], data[1], label='training_error')

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()"""
        print("her")
        self.window.update()
        pass

    def updateGUI(self):
        if self.session_tracker.updated:
            data = self.session_tracker.history
            labels = []
            self.ax.clear()
            for set in data:
                if self.lines[set] is None:
                    sp, = self.ax.plot([], [], label=set)
                    self.lines[set] = sp
                x = data[set][0]
                y = data[set][1]
                self.ax.plot(x,y)
                labels.append(set)
                self.ax.legend(labels)
            self.canvas.draw()
            self.session_tracker.updated = False

    def start(self):
        while True:
            self.updateGUI()
            self.window.update_idletasks()
            self.window.update()