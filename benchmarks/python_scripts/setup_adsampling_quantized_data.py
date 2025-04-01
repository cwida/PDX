import faiss
from setup_utils import *
from setup_settings import *
from pdxearch.index_base import BaseIndexPDXIVF, BaseIndexPDXFlat
from pdxearch.preprocessors import ADSampling
from pdxearch.constants import PDXConstants


def generate_u8_ivf(dataset_name: str):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-4', lep=True, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u8x4-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u8-matrix'))


if __name__ == "__main__":
    # generate_u8_ivf('fashion-mnist-784-euclidean')
    # generate_u8_ivf('sift-128-euclidean')
    generate_u8_ivf('openai-1536-angular')
    # generate_adsampling_ivf('gist-960-euclidean')
