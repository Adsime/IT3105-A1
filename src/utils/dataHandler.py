import csv
from src.data.mnist_basics import *
from src.utils.tflowtools import *
import time

__file_path__ = 'D:/GitProjects/IT3105-A1/src/data/'


def get_csv_cases(file, delimiter=';', one_hot=True):
    file = __file_path__ + file + ".txt"
    with open(file) as csv_file:
        data = csv.reader(csv_file, delimiter=delimiter)
        x, t = [[], []]
        for row in data:
            x.append([float(i) for i in row[0:len(row)-1]])
            t.append(int(row[-1]))
        c_count = get_class_count(t)
        if one_hot:
            t = [int_to_one_hot(i, c_count) for i in t]
        return to_case_format(x, t)


def get_mnist_cases(type='training', one_hot=True):
        x, t = load_all_flat_cases(type)
        if one_hot:
            t = [int_to_one_hot(i, get_class_count(t)) for i in t]
        return to_case_format(x, t)


def to_case_format(x_arr, t_arr):
    return [[x, t] for x, t in zip(x_arr, t_arr)]


def get_class_count(targets):
    return np.max(targets) + 1


data_sources = {
    "wine": lambda : get_csv_cases('wine'),
    "yeast": lambda : get_csv_cases('yeast'),
    "glass": lambda : get_csv_cases('glass'),
    "mnist_training": lambda : load_all_flat_cases('training'),
    "mnist_testing": lambda : load_all_flat_cases('testing')
}


def get_data(data_name):
    return data_sources[data_name]()