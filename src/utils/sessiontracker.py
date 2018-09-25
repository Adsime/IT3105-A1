import matplotlib.pyplot as plt
from src.utils.casemanager import CustomCaseManager

class SessionTracker:

    t_err = "training_error"
    v_err = "validation_error"
    top_k_err = "top_1_error"

    def __init__(self):

        self.reset()


    def reset(self):
        self.error_tracker: ErrorTacker = ErrorTacker(1000)
        self.hinton_updated = False
        self.dendro_updated = False

        self.hinton = []
        self.dendro = []
        #self.visualizer = Visualizer()
        #self.visualizer.show()
        self.grab_vars = []


    def append_grab_variable(self, variable):
        self.grab_vars.append(variable)

    def get_grab_variables(self):
        return self.grab_vars

    def set_hinton_data(self, data):
        self.hinton = data
        self.hinton_updated = True

    def set_dendro_data(self, data):
        self.dendro = data
        self.dendro_updated = True

class ErrorTacker:
    t_err = "Training error"
    v_err = "validation error"
    top_k_err = "top 1 error"

    def __init__(self, interval):
        self.updated = False
        self.interval = interval
        self.history = {
            self.t_err: [[], []],
            self.v_err: [[], []],
            self.top_k_err: [[], []]
        }

    def gather_data(self, step, t_error, gann, cman: CustomCaseManager):
        self.append_error(step, t_error, self.t_err)    # Standard training error
        if not step % self.interval or step + 1 == gann.options.steps:
            self.append_error(step, gann.do_testing(cman.get_validation_cases()), self.v_err)   # Validation error
            self.append_error(step, gann.do_testing(cman.get_training_cases()), self.top_k_err)
            self.updated = True

    def append_error(self, step, error, arr):
        try:
            self.history[arr][0].append(step)
            self.history[arr][1].append(error)
        except:
            print("" + arr + " is not a supported array. Terminating program.")
            exit(0)