import math
import numpy as np
import faiss

from sklearn import preprocessing
from setup_utils import *


def _compute_nbuckets(num_embeddings):
    if num_embeddings < 500_000:
        return math.ceil(2 * math.sqrt(num_embeddings))
    elif num_embeddings < 2_500_000:
        return math.ceil(4 * math.sqrt(num_embeddings))
    else:
        return math.ceil(8 * math.sqrt(num_embeddings))


def _load_and_prepare(dataset_abbrev, normalize):
    hdf5_name, dims = DATASET_INFO[dataset_abbrev]
    dims = int(dims)
    data = read_hdf5_train_data(hdf5_name)
    if normalize:
        data = preprocessing.normalize(data, axis=1, norm='l2')
    data = np.ascontiguousarray(data, dtype=np.float32)
    return hdf5_name, dims, data


def generate_faiss_index(dataset_abbrev: str, normalize=True):
    hdf5_name, dims, data = _load_and_prepare(dataset_abbrev, normalize)
    idx_path = os.path.join(FAISS_DATA, get_core_index_filename(hdf5_name, normalize))
    num_embeddings = len(data)
    nbuckets = _compute_nbuckets(num_embeddings)
    print(f'IVFFlat: {num_embeddings} embeddings, {dims}D, {nbuckets} buckets')

    quantizer = faiss.IndexFlatL2(dims)
    index = faiss.IndexIVFFlat(quantizer, dims, nbuckets)
    print('Training')
    index.train(data)
    print('Adding')
    index.add(data)
    print('Persisting')
    faiss.write_index(index, idx_path)


def generate_faiss_sq8_index(dataset_abbrev: str, normalize=True):
    hdf5_name, dims, data = _load_and_prepare(dataset_abbrev, normalize)
    idx_path = os.path.join(FAISS_DATA, get_core_index_filename(hdf5_name, sq8=True))
    num_embeddings = len(data)
    nbuckets = _compute_nbuckets(num_embeddings)
    print(f'IVFSQ8: {num_embeddings} embeddings, {dims}D, {nbuckets} buckets')

    quantizer = faiss.IndexFlatL2(dims)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dims, nbuckets, faiss.ScalarQuantizer.QT_8bit
    )
    print('Training')
    index.train(data)
    print('Adding')
    index.add(data)
    print('Persisting')
    faiss.write_index(index, idx_path)
