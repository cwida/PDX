import sys
import numpy as np
import h5py
from setup_settings import *


def read_hdf5_train_data(dataset):
    hdf5_file_name = os.path.join(RAW_DATA, dataset + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["train"], dtype=np.float32)


def read_hdf5_test_data(dataset):
    hdf5_file_name = os.path.join(RAW_DATA, dataset + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["test"], dtype=np.float32)


def read_hdf5_data(dataset):
    hdf5_file_name = os.path.join(RAW_DATA, dataset + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["train"], dtype=np.float32), np.array(hdf5_file["test"], dtype=np.float32)


def get_ground_truth_filename(file, k): return f"{file}_{k}.json"


def get_core_index_filename(file): return f"ivf_{file}.index"


def get_delta_d(ndim):
    delta_d = 32
    if ndim < 128:
        delta_d = int(ndim / 4)
    return delta_d


DIMENSIONALITIES = {
    'random-xs-20-angular': 20,
    'random-s-100-euclidean': 100,
    'har-561': 561,
    'nytimes-16-angular': 16,
    'nytimes-256-angular': 256,
    'mnist-784-euclidean': 784,
    'fashion-mnist-784-euclidean': 784,
    'glove-25-angular': 25,
    'glove-50-angular': 50,
    'glove-100-angular': 100,
    'glove-200-angular': 200,
    'sift-128-euclidean': 128,
    'trevi-4096': 4096,
    'msong-420': 420,
    'contriever-768': 768,
    'stl-9216': 9216,
    'gist-960-euclidean': 960,
    'deep-image-96-angular': 96,
    'instructorxl-arxiv-768': 768,
    'openai-1536-angular': 1536
}


if __name__ == '__main__':
    if not os.path.exists(RAW_DATA):
        os.makedirs(RAW_DATA)
    if not os.path.exists(NARY_DATA):
        os.makedirs(NARY_DATA)
