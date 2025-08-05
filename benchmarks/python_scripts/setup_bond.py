import faiss
from pdxearch.index_base import BaseIndexPDXIVF, BaseIndexPDXFlat
from pdxearch.constants import PDXConstants
from setup_utils import *
from setup_settings import *
from sklearn import preprocessing


def generate_bond_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.read_index(index_path)
    data = read_hdf5_train_data(dataset_name)
    if normalize:
        print('Normalizing')
        data = preprocessing.normalize(data, axis=1, norm='l2')
    print('Saving...')
    # PDX
    base_idx._to_pdx(data, _type='pdx', use_original_centroids=True, bond=True)
    base_idx._persist(os.path.join(PDX_DATA, dataset_name + '-ivf'))



def generate_bond_flat(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXFlat(DIMENSIONALITIES[dataset_name], 'l2sq')
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    if normalize:
        print('Normalizing')
        data = preprocessing.normalize(data, axis=1, norm='l2')
    print('Saving')
    # PDX FLAT
    base_idx._to_pdx(data, size_partition=PDXConstants.PDXEARCH_VECTOR_SIZE, _type='pdx', bond=True)
    base_idx._persist(os.path.join(PDX_DATA, dataset_name + '-flat'))




if __name__ == "__main__":
    generate_bond_ivf('coco-nomic-768-normalized', normalize=True)
    generate_bond_ivf('simplewiki-openai-3072-normalized', normalize=True)
    generate_bond_ivf('imagenet-align-640-normalized', normalize=True)
    generate_bond_ivf('imagenet-clip-512-normalized', normalize=True)
    generate_bond_ivf('laion-clip-512-normalized', normalize=True)
    generate_bond_ivf('codesearchnet-jina-768-cosine', normalize=True)
    generate_bond_ivf('yi-128-ip', normalize=True)
    generate_bond_ivf('landmark-dino-768-cosine', normalize=True)
    generate_bond_ivf('landmark-nomic-768-normalized', normalize=True)
    generate_bond_ivf('arxiv-nomic-768-normalized', normalize=True)
    generate_bond_ivf('ccnews-nomic-768-normalized', normalize=True)
    generate_bond_ivf('celeba-resnet-2048-cosine', normalize=True)
    generate_bond_ivf('llama-128-ip', normalize=True)
    generate_bond_ivf('yandex-200-cosine', normalize=True)
    generate_bond_ivf('word2vec-300', normalize=True)
    generate_bond_ivf('sift-128-euclidean', normalize=True)
    generate_bond_ivf('openai-1536-angular', normalize=True)
    generate_bond_ivf('msong-420', normalize=True)
    generate_bond_ivf('instructorxl-arxiv-768', normalize=True)
    generate_bond_ivf('contriever-768', normalize=True)
    generate_bond_ivf('gist-960-euclidean', normalize=True)
    generate_bond_ivf('yahoo-minilm-384-normalized', normalize=True)
    generate_bond_ivf('gooaq-distilroberta-768-normalized', normalize=True)
    generate_bond_ivf('agnews-mxbai-1024-euclidean', normalize=True)
    generate_bond_ivf('glove-200-angular', normalize=True)

    generate_bond_flat('coco-nomic-768-normalized', normalize=True)
    generate_bond_flat('simplewiki-openai-3072-normalized', normalize=True)
    generate_bond_flat('imagenet-align-640-normalized', normalize=True)
    generate_bond_flat('imagenet-clip-512-normalized', normalize=True)
    generate_bond_flat('laion-clip-512-normalized', normalize=True)
    generate_bond_flat('codesearchnet-jina-768-cosine', normalize=True)
    generate_bond_flat('yi-128-ip', normalize=True)
    generate_bond_flat('landmark-dino-768-cosine', normalize=True)
    generate_bond_flat('landmark-nomic-768-normalized', normalize=True)
    generate_bond_flat('arxiv-nomic-768-normalized', normalize=True)
    generate_bond_flat('ccnews-nomic-768-normalized', normalize=True)
    generate_bond_flat('celeba-resnet-2048-cosine', normalize=True)
    generate_bond_flat('llama-128-ip', normalize=True)
    generate_bond_flat('yandex-200-cosine', normalize=True)
    generate_bond_flat('word2vec-300', normalize=True)
    generate_bond_flat('sift-128-euclidean', normalize=True)
    generate_bond_flat('openai-1536-angular', normalize=True)
    generate_bond_flat('msong-420', normalize=True)
    generate_bond_flat('instructorxl-arxiv-768', normalize=True)
    generate_bond_flat('contriever-768', normalize=True)
    generate_bond_flat('gist-960-euclidean', normalize=True)
    generate_bond_flat('yahoo-minilm-384-normalized', normalize=True)
    generate_bond_flat('gooaq-distilroberta-768-normalized', normalize=True)
    generate_bond_flat('agnews-mxbai-1024-euclidean', normalize=True)
    generate_bond_flat('glove-200-angular', normalize=True)
