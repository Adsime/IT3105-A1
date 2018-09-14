import csv
from src.data.mnist_basics import *
from src.utils.tflowtools import *
import time

#__file_path__ = 'D:/GitProjects/IT3105-A1/src/data/'    # Home pc
__file_path__ = 'C:/Users/bruker/Desktop/gitProjects/IT3105-A1/src/data/'    # School pc


def get_csv_cases(file, scale_func, delimiter=';'):
    file = __file_path__ + file + ".txt"
    with open(file) as csv_file:
        data = csv.reader(csv_file, delimiter=delimiter)
        x, t = [[], []]
        for row in data:
            x.append([float(i) for i in row[:-1]])
            t.append(int(row[-1]))
    return to_case_format(x, t, scale_func)


def to_case_format(x_arr, t_arr, scale_func):
    c_count = get_class_count(t_arr)
    return [[x, t] for x, t in zip(x_arr if scale_func is None else scale_func(x_arr),
                                   [int_to_one_hot(i, c_count) for i in t_arr])]


def flat_to_case(file, scale_func):
    x, t = load_all_flat_cases(file)
    return to_case_format(x, t, scale_func)


def get_class_count(targets):
    return np.max(targets) + 1


def lambda_gen(file, delimiter=',', compiled=False): return lambda scale_func=None: \
    flat_to_case(file, scale_func) if compiled else get_csv_cases(file, scale_func, delimiter)


data_sources = {
    "wine": lambda_gen('wine', ';'),
    "yeast": lambda_gen('yeast'),
    "glass": lambda_gen('glass'),
    "mnist_training": lambda_gen('training', True),
    "mnist_testing": lambda_gen('testing', True)
}


def mean_std(cases):
    mean = np.mean(cases, axis=0)
    std_div = np.std(cases, axis=0)
    return [np.divide(np.subtract(case, mean), std_div) for case in cases]


def get_data(data_name, scale_func=mean_std):
    return data_sources[data_name](scale_func)
