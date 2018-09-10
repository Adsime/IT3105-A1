class SessionTracker:

    def __init__(self):
        self.reset()

    def reset(self):
        #   error history. format: [ [step] , [error] ]
        self.history = {
            "training_error": [[], []],
            "validation_error": [[], []]
        }
        self.grab_vars = []

    def append_training_error(self, step, error, arr="training_error"):
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