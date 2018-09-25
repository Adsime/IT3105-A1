import csv
from src.data.mnist_basics import *
from src.utils.tflowtools import *
import time

__file_path__ = 'data/'


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
    "mnist_training": lambda_gen('training', ',', True),
    "mnist_testing": lambda_gen('testing', ',', True)
}


def mean_std(cases):
    mean = np.mean(cases, axis=0)
    std_div = np.std(cases, axis=0)
    std_div = [1 if val == 0 else val for val in std_div]
    return [np.divide(np.subtract(case, mean), std_div) for case in cases]


def direct_scale(cases):
    max = np.max(cases)
    cases = np.divide(cases, max)
    return cases


def get_data(data_name, scale_func=mean_std):
    return data_sources[data_name](scale_func)


def Wine(scale_func=mean_std):
    return get_csv_cases('wine', scale_func, ';')


def Yeast(scale_func=mean_std):
    return get_csv_cases('yeast', scale_func, ',')


def Glass(scale_func=mean_std):
    return get_csv_cases("glass", scale_func, ',')


def Mnist(scale_func=mean_std, dataset='training'):
    return flat_to_case(dataset, scale_func)
