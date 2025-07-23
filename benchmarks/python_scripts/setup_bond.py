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
    generate_bond_ivf('sift-128-euclidean')
    generate_bond_flat('sift-128-euclidean')
