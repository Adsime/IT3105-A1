from src.gui.frame import Frame
import tkinter as tk


class Window:

    def __init__(self):
        self.frames = []
        self.window = tk.Tk()
        self.window.wm_title("AI Programming!")
        pass

    def add_frame(self, frame: Frame):
        self.frames.append(frame)

    def updateGUI(self):
        for frame in self.frames:
            frame.update()

    def start(self):
        while True:
            self.updateGUI()
            self.window.update_idletasks()
            self.window.update()
