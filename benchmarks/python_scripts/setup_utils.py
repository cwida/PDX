import numpy as np
import h5py
from setup_settings import *

# Abbreviated name -> (hdf5_dataset_name, num_dimensions)
# Matches RAW_DATASET_PARAMS in benchmark_utils.hpp
DATASET_INFO = {
    "sift":       ("sift-128-euclidean",                128),
    "yi":         ("yi-128-ip",                         128),
    "llama":      ("llama-128-ip",                      128),
    "glove200":   ("glove-200-angular",                 200),
    "yandex":     ("yandex-200-cosine",                 200),
    "yahoo":      ("yahoo-minilm-384-normalized",       384),
    "clip":       ("imagenet-clip-512-normalized",       512),
    "contriever": ("contriever-768",                    768),
    "gist":       ("gist-960-euclidean",                960),
    "mxbai":      ("agnews-mxbai-1024-euclidean",      1024),
    "openai":     ("openai-1536-angular",              1536),
    "arxiv":      ("arxiv-nomic-768-normalized",        768),
    "wiki":       ("simplewiki-openai-3072-normalized", 3072),
}


def read_hdf5_train_data(dataset):
    hdf5_file_name = os.path.join(RAW_DATA, dataset + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["train"], dtype=np.float32)


def read_ivecs(filename):
    a = np.fromfile(filename, dtype="int32")
    d = a[0]
    print(f"\t{filename} readed")
    return a.reshape(-1, d + 1)[:, 1:]


def read_fvecs(filename):
    return read_ivecs(filename).view("float32")


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


def get_core_index_filename(file, norm=True, balanced=False):
    if balanced:
        return f"ivf_{file}_norm.index.balanced"
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
