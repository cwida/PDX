import math
from numpy.random import default_rng
from pdxearch.index_core import IVF
from setup_utils import *
from setup_settings import *


# Generates core IVF index with FAISS
def generate_core_ivf(dataset_name: str):
    idx_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name))
    data = read_hdf5_train_data(dataset_name)
    num_embeddings = len(data)
    print('Num embeddings:', num_embeddings)
    if num_embeddings < 500_000:  # If collection is too small we better use only 1 * SQRT(n)
        nbuckets = math.ceil(1 * math.sqrt(num_embeddings))
    elif num_embeddings < 2_500_000:  # Faiss recommends 4*sqrt(n), pg_vector 1*sqrt(n), we will take the middle ground
        nbuckets = math.ceil(2 * math.sqrt(num_embeddings))
    else:  # Deep with 10m
        nbuckets = math.ceil(4 * math.sqrt(num_embeddings))
    print('N buckets:', nbuckets)
    core_idx = IVF(DIMENSIONALITIES[dataset_name], 'l2sq', nbuckets)
    training_points = nbuckets * 50  # Our collections do not need that many training points
    if training_points < num_embeddings:
        rng = default_rng()
        training_sample_idxs = rng.choice(num_embeddings, size=training_points, replace=False)
        training_sample_idxs.sort()
        print('Training with', training_points)
        core_idx.train(data[training_sample_idxs])
    else:
        print('Training with all points')
        core_idx.train(data)
    print('Building')
    core_idx.add(data)
    print('Persisting')
    core_idx.persist_core(idx_path)


if __name__ == "__main__":
    generate_core_ivf('fashion-mnist-784-euclidean')
