import matplotlib.pyplot as plt

class SessionTracker:

    t_err = "training_error"
    v_err = "validation_error"
    top_k_err = "top_1_error"

    def __init__(self):
        self.updated = False
        self.reset()


    def reset(self):
        #   error history. format: [ [step] , [error] ]
        self.history = {
            self.t_err: [[], []],
            self.v_err: [[], []],
            self.top_k_err: [[], []]
        }
        #self.visualizer = Visualizer()
        #self.visualizer.show()
        self.grab_vars = []

    def append_error(self, step, error, arr):
        try:
            self.history[arr][0].append(step)
            self.history[arr][1].append(error)
        except:
            print("" + arr + " is not a supported array. Terminating program.")
            exit(0)

    def append_grab_variable(self, variable):
        self.grab_vars.append(variable)

    def get_grab_variables(self):
        return self.grab_vars

    def draw(self):
        self.updated = True