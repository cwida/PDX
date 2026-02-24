import os
import sys
import numpy as np
import h5py

SOURCE_DIR = os.getcwd()

# ── Directory layout ──────────────────────────────────────────────
DATA_DIRECTORY = os.path.join("benchmarks", "datasets")
if not os.path.exists(os.path.join(SOURCE_DIR, DATA_DIRECTORY)):
    os.makedirs(os.path.join(SOURCE_DIR, DATA_DIRECTORY))

RAW_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "downloaded")
GROUND_TRUTH_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "ground_truth")
FILTERED_GROUND_TRUTH_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "ground_truth_filtered")
SEMANTIC_GROUND_TRUTH_PATH = os.path.join(SOURCE_DIR, "benchmarks", "gt")
SEMANTIC_FILTERED_GROUND_TRUTH_PATH = os.path.join(SOURCE_DIR, "benchmarks", "gt_filtered")

QUERIES_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "queries")
PDX_DATA = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "pdx")
FILTER_SELECTION_VECTORS = os.path.join(SOURCE_DIR, DATA_DIRECTORY, "selection_vectors")

for _d in [RAW_DATA, GROUND_TRUTH_DATA, FILTERED_GROUND_TRUTH_DATA,
           SEMANTIC_GROUND_TRUTH_PATH, SEMANTIC_FILTERED_GROUND_TRUTH_PATH,
           QUERIES_DATA, PDX_DATA, FILTER_SELECTION_VECTORS]:
    os.makedirs(_d, exist_ok=True)

# ── Dataset registry ──────────────────────────────────────────────
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

# ── HDF5 I/O ─────────────────────────────────────────────────────

def read_hdf5_train_data(dataset_hdf5_name):
    hdf5_file_name = os.path.join(RAW_DATA, dataset_hdf5_name + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["train"], dtype=np.float32)


def read_hdf5_test_data(dataset_hdf5_name):
    hdf5_file_name = os.path.join(RAW_DATA, dataset_hdf5_name + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["test"], dtype=np.float32)


def read_hdf5_data(dataset_hdf5_name):
    hdf5_file_name = os.path.join(RAW_DATA, dataset_hdf5_name + ".hdf5")
    hdf5_file = h5py.File(hdf5_file_name, "r")
    return np.array(hdf5_file["train"], dtype=np.float32), np.array(hdf5_file["test"], dtype=np.float32)


# ── Helpers ───────────────────────────────────────────────────────

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
