from src.utils.tflowtools import *
from src.data.mnist_basics import *


class CaseManager:

    def generate_cases(self): pass

    def organize_cases(self): pass

    def get_training_cases(self): pass

    def get_validation_cases(self): pass

    def get_testing_cases(self): pass


class DefaultCaseManager(CaseManager):

    def __init__(self, cfunc, vfrac=0, tfrac=0):
        self.casefunc = cfunc
        self.validation_fraction = vfrac
        self.test_fraction = tfrac
        self.training_fraction = 1 - (vfrac + tfrac)
        self.generate_cases()
        self.organize_cases()

    def generate_cases(self):
        self.cases = self.casefunc()  # Run the case generator.  Case = [input-vector, target-vector]

    def organize_cases(self):
        ca = np.array(self.cases)
        np.random.shuffle(ca) # Randomly shuffle all cases
        separator1 = round(len(self.cases) * self.training_fraction)
        separator2 = separator1 + round(len(self.cases)*self.validation_fraction)
        self.training_cases = ca[0:separator1]
        self.validation_cases = ca[separator1:separator2]
        self.testing_cases = ca[separator2:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases


class CustomCaseManager(CaseManager):

    def __init__(self, cases, testing_cases, vfrac=0, tfrac=0):
        self.tfrac = 1 - (vfrac + tfrac)
        self.ca = cases
        separator1 = round(len(self.ca) * self.tfrac)
        separator2 = separator1 + round(len(self.ca) * vfrac)
        np.random.shuffle(self.ca)
        self.training_cases = self.ca[0:tfrac]
        self.validation_cases = self.ca[separator1:separator2]
        self.testing_cases = testing_cases if tfrac == 0 else ca[separator2:]

    def get_training_cases(self): return self.training_cases

    def get_validation_cases(self): return self.validation_cases

    def get_testing_cases(self): return self.testing_cases
