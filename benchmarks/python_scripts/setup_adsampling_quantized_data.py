import faiss
from setup_utils import *
from setup_settings import *
from pdxearch.index_base import BaseIndexPDXIVF, BaseIndexPDXFlat
from pdxearch.preprocessors import ADSampling
from pdxearch.constants import PDXConstants
from sklearn import preprocessing


def generate_u7_vh_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-v4-h', lep=True, lep_bw=7, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u7-v4-h64-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u7-v4-h64-matrix'))

def generate_u8_vh_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-v4-h', lep=True, lep_bw=8, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u8-v4-h64-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u8-v4-h64-matrix'))

def generate_u8_vh_ivf_symmetric(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-v4-h', lep=True, lep_bw=8, centroids_preprocessor=preprocessor, use_original_centroids=True, use_global_params=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u8-v4-h64-ivf-sym'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u8-v4-h64-matrix-sym'))

def generate_lep_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-v4-h', lep=True, lep_bw=4, centroids_preprocessor=preprocessor, use_original_centroids=True, use_exceptions=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-lep-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-lep-matrix'))

def generate_u7_vh_b64_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-v4-h', blockify=True, h_dims_block=64, lep=True, lep_bw=7, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u7-v4-h64-b64-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u7-v4-h64-b64-matrix'))


def generate_u7_vh128_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-v4-h', lep=True, lep_bw=7, h_dims_block=128, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u7-v4-h128-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u7-v4-h128-matrix'))

def generate_u8_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-4', lep=True, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u8x4-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u8-matrix'))


def generate_u7_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-4', lep=True, lep_bw=7, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u7x4-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u7-matrix'))

def generate_u6_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-4', lep=True, lep_bw=6, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u6x4-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u6-matrix'))


def generate_u4_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-4', lep=True, lep_bw=4, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u4x4-ivf'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u4-matrix'))


def generate_u4_symmetric_ivf(dataset_name: str, normalize=True):
    base_idx = BaseIndexPDXIVF(DIMENSIONALITIES[dataset_name], 'l2sq')
    # Core index IVF must exist
    index_path = os.path.join(CORE_INDEXES_FAISS, get_core_index_filename(dataset_name, norm=normalize))
    # Reads the core index created by faiss to generate the PDX index
    base_idx.core_index.index = faiss.read_index(index_path)
    print('Reading train data')
    data = read_hdf5_train_data(dataset_name)
    preprocessor = ADSampling(DIMENSIONALITIES[dataset_name])
    preprocessor.preprocess(data, inplace=True, normalize=True)
    print('Saving')
    # PDX FLAT BLOCKIFIED
    base_idx._to_pdx(data, _type='pdx-4', lep=True, lep_bw=4, centroids_preprocessor=preprocessor, use_original_centroids=True)
    base_idx._persist(os.path.join(PDX_ADSAMPLING_DATA, dataset_name + '-u4x4-ivf-s'))

    preprocessor.store_metadata(os.path.join(NARY_ADSAMPLING_DATA, dataset_name + '-u4-s-matrix'))


if __name__ == "__main__":
    # generate_u8_ivf('fashion-mnist-784-euclidean')
    # generate_u8_ivf('sift-128-euclidean')

    # generate_u7_vh_ivf('openai-1536-angular')
    # generate_u7_vh_ivf('instructorxl-arxiv-768')
    # generate_u7_vh128_ivf('openai-1536-angular')
    # generate_u7_vh_b64_ivf('openai-1536-angular')

    # The real one for lep
    # generate_lep_ivf('contriever-768')
    # generate_lep_ivf('gist-960-euclidean')
    # generate_lep_ivf('openai-1536-angular')
    # generate_lep_ivf('instructorxl-arxiv-768')
    # generate_lep_ivf('msong-420')
    # generate_lep_ivf('word2vec-300')

    # The real one for asymmetric
    # generate_u8_vh_ivf('openai-1536-angular')
    # generate_u8_vh_ivf('instructorxl-arxiv-768')
    # generate_u8_vh_ivf('gist-960-euclidean')
    # generate_u8_vh_ivf('contriever-768')
    # generate_u8_vh_ivf('msong-420')
    # generate_u8_vh_ivf('word2vec-300')

    # The real one for symmetric
    generate_u8_vh_ivf_symmetric('openai-1536-angular')
    generate_u8_vh_ivf_symmetric('instructorxl-arxiv-768')
    generate_u8_vh_ivf_symmetric('gist-960-euclidean')
    generate_u8_vh_ivf_symmetric('contriever-768')
    generate_u8_vh_ivf_symmetric('msong-420')
    generate_u8_vh_ivf_symmetric('gooaq-distilroberta-768-normalized')
    generate_u8_vh_ivf_symmetric('agnews-mxbai-1024-euclidean')
    generate_u8_vh_ivf_symmetric('coco-nomic-768-normalized')
    generate_u8_vh_ivf_symmetric('simplewiki-openai-3072-normalized')

    # generate_u7_ivf('openai-1536-angular')
    # generate_u6_ivf('openai-1536-angular')
    # generate_u4_ivf('openai-1536-angular')
    # generate_u4_symmetric_ivf('openai-1536-angular')

    # generate_u8_ivf('gist-960-euclidean')
    # generate_u6_ivf('gist-960-euclidean')
    # generate_u4_ivf('gist-960-euclidean')
    # generate_u4_symmetric_ivf('gist-960-euclidean')
    #
    # generate_u8_ivf('instructorxl-arxiv-768')
    # generate_u6_ivf('instructorxl-arxiv-768')
    # generate_u4_ivf('instructorxl-arxiv-768')
    # generate_u4_symmetric_ivf('instructorxl-arxiv-768')

    # generate_u8_ivf('contriever-768')
    # generate_u7_ivf('contriever-768')
    # generate_u6_ivf('contriever-768')
    # generate_u4_ivf('contriever-768')
    # generate_u4_symmetric_ivf('contriever-768')
