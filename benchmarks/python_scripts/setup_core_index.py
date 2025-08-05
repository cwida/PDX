import math
from numpy.random import default_rng
from pdxearch.index_core import IVF, IMI
from setup_utils import *
from setup_settings import *

from sklearn import preprocessing

# Generates core IVF index with FAISS
def generate_core_ivf(dataset_name: str, normalize=True):
    idx_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, normalize))
    data = read_hdf5_train_data(dataset_name)
    num_embeddings = len(data)
    print('Num embeddings:', num_embeddings)
    if dataset_name == "simplewiki-openai-3072-normalized": # Special case because it has too many dimensions!
        nbuckets = 2048
    elif num_embeddings < 500_000:  # If collection is too small we better use only 1 * SQRT(n)
        nbuckets = math.ceil(2 * math.sqrt(num_embeddings))
    elif num_embeddings < 2_500_000:  # Faiss recommends 4*sqrt(n), pg_vector 1*sqrt(n), we will take the middle ground
        nbuckets = math.ceil(4 * math.sqrt(num_embeddings))
    else:  # Deep with 10m
        nbuckets = math.ceil(8 * math.sqrt(num_embeddings))
    print('N buckets:', nbuckets)
    if normalize:
        data = preprocessing.normalize(data, axis=1, norm='l2')
    core_idx = IVF(DIMENSIONALITIES[dataset_name], 'l2sq', nbuckets)
    training_points = nbuckets * 300  # Our collections do not need that many training points
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


def generate_core_imi(dataset_name: str, normalize=True):
    idx_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, normalize))
    idx_path_l0 = os.path.join(CORE_INDEXES_FAISS_L0, get_core_index_filename(dataset_name, normalize))
    data = read_hdf5_train_data(dataset_name)
    num_embeddings = len(data)
    print('Num embeddings:', num_embeddings)
    if dataset_name == "simplewiki-openai-3072-normalized": # Special case because it has too many dimensions!
        nbuckets = 2048
    elif num_embeddings < 500_000:  # If collection is too small we better use only 1 * SQRT(n)
        nbuckets = math.ceil(2 * math.sqrt(num_embeddings))
    elif num_embeddings < 2_500_000:  # Faiss recommends 4*sqrt(n), pg_vector 1*sqrt(n), we will take the middle ground
        nbuckets = math.ceil(4 * math.sqrt(num_embeddings))
    else:  # Deep with 10m
        nbuckets = math.ceil(8 * math.sqrt(num_embeddings))
    nbuckets_l0 = 64 # math.ceil(math.sqrt(nbuckets))
    print('N buckets L1:', nbuckets)
    print('N buckets L0:', nbuckets_l0)
    if normalize:
        data = preprocessing.normalize(data, axis=1, norm='l2')
    core_idx = IMI(DIMENSIONALITIES[dataset_name], 'l2sq', nbuckets, nbuckets_l0)
    training_points = nbuckets * 300  # Our collections do not need that many training points
    if training_points < num_embeddings:
        rng = default_rng()
        training_sample_idxs = rng.choice(num_embeddings, size=training_points, replace=False)
        training_sample_idxs.sort()
        print('Training L1 with', training_points)
        core_idx.train(data[training_sample_idxs])
    else:
        print('Training L1 with all points')
        core_idx.train(data)
    print('Building L1')
    core_idx.add(data)
    print('Training and Building L0')
    core_idx.train_add_l0()
    print('Persisting L1 and L0')
    core_idx.persist_core(idx_path, idx_path_l0)

if __name__ == "__main__":
    generate_core_ivf('word2vec-300', normalize=True)
    generate_core_ivf('openai-1536-angular', normalize=True)
    generate_core_ivf('msong-420', normalize=True)
    generate_core_ivf('instructorxl-arxiv-768', normalize=True)
    generate_core_ivf('contriever-768', normalize=True)
    generate_core_ivf('gist-960-euclidean', normalize=True)
    generate_core_ivf('gooaq-distilroberta-768-normalized', normalize=True)
    generate_core_ivf('agnews-mxbai-1024-euclidean', normalize=True)
    generate_core_ivf('coco-nomic-768-normalized', normalize=True)
    generate_core_ivf('simplewiki-openai-3072-normalized', normalize=True)
    generate_core_ivf('imagenet-align-640-normalized', normalize=True)
    generate_core_ivf('yandex-200-cosine', normalize=True)
    generate_core_ivf('imagenet-clip-512-normalized', normalize=True)
    generate_core_ivf('laion-clip-512-normalized', normalize=True)
    generate_core_ivf('codesearchnet-jina-768-cosine', normalize=True)
    generate_core_ivf('yi-128-ip', normalize=True)
    generate_core_ivf('landmark-dino-768-cosine', normalize=True)
    generate_core_ivf('landmark-nomic-768-normalized', normalize=True)
    generate_core_ivf('arxiv-nomic-768-normalized', normalize=True)
    generate_core_ivf('ccnews-nomic-768-normalized', normalize=True)
    generate_core_ivf('celeba-resnet-2048-cosine', normalize=True)
    generate_core_ivf('llama-128-ip', normalize=True)
    generate_core_ivf('sift-128-euclidean', normalize=True)
    generate_core_ivf('yahoo-minilm-384-normalized', normalize=True)
    generate_core_ivf('glove-200-angular', normalize=True)

    generate_core_imi('coco-nomic-768-normalized', normalize=True)
    generate_core_imi('simplewiki-openai-3072-normalized', normalize=True)
    generate_core_imi('imagenet-align-640-normalized', normalize=True)
    generate_core_imi('imagenet-clip-512-normalized', normalize=True)
    generate_core_imi('laion-clip-512-normalized', normalize=True)
    generate_core_imi('codesearchnet-jina-768-cosine', normalize=True)
    generate_core_imi('yi-128-ip', normalize=True)
    generate_core_imi('landmark-dino-768-cosine', normalize=True)
    generate_core_imi('landmark-nomic-768-normalized', normalize=True)
    generate_core_imi('arxiv-nomic-768-normalized', normalize=True)
    generate_core_imi('ccnews-nomic-768-normalized', normalize=True)
    generate_core_imi('celeba-resnet-2048-cosine', normalize=True)
    generate_core_imi('llama-128-ip', normalize=True)
    generate_core_imi('yandex-200-cosine', normalize=True)
    generate_core_imi('word2vec-300', normalize=True)
    generate_core_imi('sift-128-euclidean', normalize=True)
    generate_core_imi('openai-1536-angular', normalize=True)
    generate_core_imi('msong-420', normalize=True)
    generate_core_imi('instructorxl-arxiv-768', normalize=True)
    generate_core_imi('contriever-768', normalize=True)
    generate_core_imi('gist-960-euclidean', normalize=True)
    generate_core_imi('yahoo-minilm-384-normalized', normalize=True)
    generate_core_imi('gooaq-distilroberta-768-normalized', normalize=True)
    generate_core_imi('agnews-mxbai-1024-euclidean', normalize=True)
    generate_core_imi('glove-200-angular', normalize=True)
