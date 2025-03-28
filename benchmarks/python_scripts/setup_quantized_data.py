import faiss
from setup_utils import *
from setup_settings import *
from pdxearch.index_base import BaseIndexPDXIVF, BaseIndexPDXFlat
from pdxearch.preprocessors import ADSampling
from pdxearch.constants import PDXConstants


def generate_flat(dataset_name: str):
    base_idx = BaseIndexPDXFlat(DIMENSIONALITIES[dataset_name], 'l2sq')
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, size_partition=PDXConstants.PDX_VECTOR_SIZE, _type='pdx')
    base_idx._persist(os.path.join(PDX_DATA, dataset_name + '-flat-blocks'))



def generate_u8(dataset_name: str):
    base_idx = BaseIndexPDXFlat(DIMENSIONALITIES[dataset_name], 'l2sq')
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    data = data.astype(dtype=np.uint8)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, size_partition=PDXConstants.PDX_VECTOR_SIZE, _type='pdx')
    base_idx._persist(os.path.join(PDX_DATA, dataset_name + '-u8-flat-blocks'))

    # N-ARY: We also generate here the N-ary format (only one partition)
    base_idx._to_pdx(data, _type='n-ary', size_partition=len(data), randomize=False)
    base_idx._persist(os.path.join(NARY_DATA, dataset_name + '-u8'))


def generate_chunk_flat(dataset_name: str):
    base_idx = BaseIndexPDXFlat(DIMENSIONALITIES[dataset_name], 'l2sq')
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    data = data.astype(dtype=np.uint8)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, size_partition=PDXConstants.PDX_VECTOR_SIZE, _type='pdx-4')
    base_idx._persist(os.path.join(PDX_DATA, dataset_name + '-u8x4-flat-blocks'))


if __name__ == "__main__":
    generate_flat('fashion-mnist-784-euclidean')
    generate_u8('fashion-mnist-784-euclidean')
    generate_chunk_flat('fashion-mnist-784-euclidean')
    # generate_adsampling_ivf('openai-1536-angular')
    # generate_adsampling_ivf('gist-960-euclidean')
