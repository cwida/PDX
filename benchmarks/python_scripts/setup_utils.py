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


def get_ground_truth_filename(file, k, norm=True):
    if norm:
        return f"{file}_{k}_norm.json"
    return f"{file}_{k}.json"


def get_core_index_filename(file, norm=True):
    if norm:
        return f"ivf_{file}_norm.index"
    return f"ivf_{file}.index"


def get_delta_d(ndim):
    delta_d = 32
    if ndim < 128:
        delta_d = int(ndim / 4)
    return delta_d


if __name__ == '__main__':
    if not os.path.exists(RAW_DATA):
        os.makedirs(RAW_DATA)
    if not os.path.exists(NARY_DATA):
        os.makedirs(NARY_DATA)
