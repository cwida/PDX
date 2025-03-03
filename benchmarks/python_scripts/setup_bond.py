import faiss
from pdxearch.index_base import BaseIndexPDXIVF, BaseIndexPDXFlat
from pdxearch.constants import PDXConstants
from setup_utils import *
from setup_settings import *


def generate_bond_ivf(dataset_name: str):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    data = read_hdf5_train_data(dataset_name)
    print('Saving...')
    # PDX
    base_idx._to_pdx(data, _type='pdx', use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_DATA, dataset_name + '-ivf'))

    # DSM
    # base_idx._to_pdx(data, _type='dsm', use_original_centroids=True)
    # base_idx._persist(os.path.join(DSM_BOND_DATA, dataset_name + '-ivf'))

    # N-ARY: We also generate here the N-ary format
    base_idx._to_pdx(data, _type='n-ary', use_original_centroids=True)
    base_idx._persist(os.path.join(NARY_DATA, dataset_name + '-ivf'))

    # BLOCKIFYING: Tight blocks of 64, not used on IVF indexes for now
    # base_idx._to_pdx(data, _type='pdx', use_original_centroids=True, blockify=True)
    # base_idx._persist(os.path.join(PDX_DATA, dataset_name + '-ivf-blocks'))


def generate_bond_flat(dataset_name: str):
    base_idx = BaseIndexPDXFlat(DIMENSIONALITIES[dataset_name], 'l2sq')
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    print('Saving')
    # PDX FLAT
    base_idx._to_pdx(data, size_partition=PDXConstants.PDXEARCH_VECTOR_SIZE, _type='pdx')
    base_idx._persist(os.path.join(PDX_DATA, dataset_name + '-flat'))
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, size_partition=PDXConstants.PDX_VECTOR_SIZE, _type='pdx')
    base_idx._persist(os.path.join(PDX_DATA, dataset_name + '-flat-blocks'))
    # DSM
    # base_idx._to_pdx(data, size_partition=PDXConstants.PDXEARCH_VECTOR_SIZE, _type='dsm')
    # base_idx._persist(os.path.join(DSM_BOND_DATA, dataset_name + '-flat'))


if __name__ == "__main__":
    generate_bond_ivf('fashion-mnist-784-euclidean')
    generate_bond_flat('fashion-mnist-784-euclidean')
