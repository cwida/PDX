import sys
import os
import faiss
from pdxearch.index_base import BaseIndexPDXIVF
from pdxearch.preprocessors import BSA
from setup_utils import *


def generate_bsa_ivf(dataset_name: str, _types=('pdx', 'dual')):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    data = read_hdf5_train_data(dataset_name)
    preprocessor = BSA(DIMENSIONALITIES[dataset_name])
    data = preprocessor.preprocess(data)
    print('Saving...')
    # PDX
    base_idx._to_pdx(data, _type='pdx')
    base_idx._persist(os.path.join(PDX_BSA_DATA, dataset_name + '-ivf'))
    # DSM
    # base_idx._to_pdx(data, _type='dsm')
    # base_idx._persist(os.path.join(DSM_BSA_DATA, dataset_name + '-ivf'))
    # DUAL-BLOCK
    base_idx._to_pdx(data, _type='dual', delta_d=get_delta_d(len(data[0])))
    base_idx._persist(os.path.join(NARY_BSA_DATA, dataset_name + '-ivf-dual-block'))

    # Store metadata needed by BSA
    preprocessor.store_metadata(
        os.path.join(NARY_BSA_DATA, dataset_name + "-matrix"),
        os.path.join(NARY_BSA_DATA, dataset_name + "-dimension-variances"),
        os.path.join(NARY_BSA_DATA, dataset_name + "-dimension-means"),
        os.path.join(NARY_BSA_DATA, dataset_name + "-base-square")
    )


if __name__ == "__main__":
    generate_bsa_ivf('sift-128-euclidean')
