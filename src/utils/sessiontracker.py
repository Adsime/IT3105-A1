import matplotlib.pyplot as plt
from src.utils.casemanager import CustomCaseManager

class SessionTracker:

    t_err = "training_error"
    v_err = "validation_error"

    def __init__(self):
        self.reset()

    def set_options(self, options):
        self.options = options
        self.error_tracker: ErrorTracker = ErrorTracker(self.options.vint)

    def reset(self):
        self.output_data_updated = False
        self.bias_data_updated = False
        self.weight_data_updated = False
        self.dendro_data_updated = False

        self.output_data = []
        self.bias_data = []
        self.weight_data = []
        self.dendro_data = []

    def set_output_data(self, data):
        self.output_data = data
        self.output_data_updated = True

    def set_bias_data(self, data):
        self.bias_data = data
        self.bias_data_updated = True

    def set_weight_data(self, data):
        self.weight_data = data
        self.weight_data_updated = True

    def set_dendro_data(self, data, targets):
        self.dendro_data = [data, targets]
        self.dendro_data_updated = True

    def get_output_data(self):
        return self.output_data

    def get_bias_data(self):
        return self.bias_data

    def get_weight_data(self):
        return self.weight_data

    def get_dendro_data(self):
        return self.dendro_data


    def gather_data(self, step, t_error, gann, cman: CustomCaseManager):
        self.error_tracker.gather_data(step, t_error, gann, cman)

class ErrorTracker:
    t_err = "Training error"
    v_err = "validation error"

    def __init__(self, interval):
        self.updated = False
        self.interval = interval
        self.history = {
            self.t_err: [[], []],
            self.v_err: [[], []]
        }

    def gather_data(self, step, t_error, gann, cman: CustomCaseManager):

        if (not step % self.interval) or (step + 1 == gann.options.steps):
            self.append_error(step, t_error, self.t_err)  # Standard training error
            if len(cman.get_validation_cases()) > 0:
                self.append_error(step, gann.do_testing(cman.get_validation_cases()), self.v_err)   # Validation error
            self.updated = True

    def append_error(self, step, error, arr):
        try:
            self.history[arr][0].append(step)
            self.history[arr][1].append(error)
        except:
            print("" + arr + " is not a supported array. Terminating program.")
            exit(0)