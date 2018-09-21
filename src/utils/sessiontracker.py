import matplotlib.pyplot as plt

class SessionTracker:

    t_err = "training_error"
    v_err = "validation_error"
    top_k_err = "top_1_error"

    def __init__(self):
        self.error_updated = False
        self.hinton_updated = False
        self.reset()


    def reset(self):
        #   error history. format: [ [step] , [error] ]
        self.history = {
            self.t_err: [[], []],
            self.v_err: [[], []],
            self.top_k_err: [[], []]
        }
        self.hinton = []
        #self.visualizer = Visualizer()
        #self.visualizer.show()
        self.grab_vars = []

    def append_error(self, step, error, arr):
        try:
            self.history[arr][0].append(step)
            self.history[arr][1].append(error)
            self.error_updated = True
        except:
            print("" + arr + " is not a supported array. Terminating program.")
            exit(0)

    def append_grab_variable(self, variable):
        self.grab_vars.append(variable)

    def get_grab_variables(self):
        return self.grab_vars

    def delete(self):
        for arr in self.history:
            self.history[arr][0].pop()
            self.history[arr][1].pop()
        self.error_updated = True

    def set_hinton_data(self, data):
        self.hinton = data
        self.hinton_updated = True