import csv
from src.utils.mnist_basics import *
from src.utils.tflowtools import *

__file_path__ = 'data/'


def get_csv_cases(file, scale_func, delimiter=';'):
    file = __file_path__ + file + ".txt"
    with open(file) as csv_file:
        data = csv.reader(csv_file, delimiter=delimiter)
        x, t = split_concat_case(data)
    return to_case_format(x, t, scale_func)

def split_concat_case(cases):
    x, t = [[], []]
    for row in cases:
        x.append([float(i) for i in row[:-1]])
        t.append(int(row[-1]))
    return [x, t]

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

class DataSets():

    @staticmethod
    def Wine(scale_func=mean_std):
        return get_csv_cases('wine', scale_func, ';')

    @staticmethod
    def Yeast(scale_func=mean_std):
        return get_csv_cases('yeast', scale_func, ',')

    @staticmethod
    def Glass(scale_func=mean_std):
        return get_csv_cases("glass", scale_func, ',')

    @staticmethod
    def Hackers_Choice(scale_func=mean_std):
        return get_csv_cases("hackers_choice", scale_func, ',')

    @staticmethod
    def Mnist(scale_func=mean_std, dataset='training'):
        return flat_to_case(dataset, scale_func)

    @staticmethod
    def Parity(n_bits):
        return gen_all_parity_cases(n_bits)

    @staticmethod
    def Symmetry(vlen, case_count):
        return to_case_format(*split_concat_case(gen_symvect_dataset(vlen, case_count)), scale_func=None)

    @staticmethod
    def One_Hot_Autoencoder(length, floats=False):
        return gen_all_one_hot_cases(length, floats)

    @staticmethod
    def Dense_Autoencoder(length, size, range=(0,1)):
        return gen_dense_autoencoder_cases(length, size, range)

    @staticmethod
    def Bit_Counter(num, size):
        return gen_vector_count_cases(num, size)

    @staticmethod
    def Segment_Counter(feature_count=25, case_count=1000, minsegs=0, maxsegs=8):
        return gen_segmented_vector_cases(feature_count, case_count, minsegs, maxsegs)