class SessionTracker:

    def __init__(self):
        self.error_history = []
        self.grab_vars = []

    def reset(self):
        self.error_history = []

    def append_error_trace(self, error):
        self.error_history.append(error)

    def append_grab_variable(self, variable):
        self.grab_vars.append(variable)

    def get_grab_variables(self):
        return self.grab_vars