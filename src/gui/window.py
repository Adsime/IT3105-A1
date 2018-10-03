from src.gui.frame import Frame
import tkinter as tk


class Window:

    def __init__(self):
        self.frames = []
        self.windows = []
        self.window = tk.Tk()
        self.window.wm_title("AI Programming!")
        self.windows.append(self.window)
        pass

    def add_frame(self, frame: Frame):
        self.frames.append(frame)

    def updateGUI(self):
        for frame in self.frames:
            frame.update()

    def add_window(self):
        window = tk.Tk()
        window.wm_title("AI Programming!")
        self.windows.append(window)

    def start(self):
        while True:
            self.updateGUI()
            for window in self.windows:
                window.update_idletasks()
                window.update()
